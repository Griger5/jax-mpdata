from functools import partial

import jax
import jax.numpy as jnp

import mpi4jax
from mpi4py import MPI

from .implementation import _pad_field, _solve_impl, _unpad_field, fill_halos as fill_periodic_halos

def _fill_halos_axis(psi, halo, axis, comm):
    psi = jnp.moveaxis(psi, axis, 0)

    lower_src, upper_dst = comm.Shift(axis, 1)
    upper_src, lower_dst = comm.Shift(axis, -1)

    send_upper = psi[-2 * halo:-halo]
    recv_lower = mpi4jax.sendrecv(
        send_upper,
        recvbuf=psi[:halo],
        source=lower_src,
        dest=upper_dst,
        sendtag=2 * axis,
        recvtag=2 * axis,
        comm=comm,
    )
    psi = psi.at[:halo].set(recv_lower)

    send_lower = psi[halo:2 * halo]
    recv_upper = mpi4jax.sendrecv(
        send_lower,
        recvbuf=psi[-halo:],
        source=upper_src,
        dest=lower_dst,
        sendtag=2 * axis + 1,
        recvtag=2 * axis + 1,
        comm=comm,
    )
    psi = psi.at[-halo:].set(recv_upper)

    return jnp.moveaxis(psi, 0, axis)

def _fill_halos(psi, halo, *, comm):
    for axis in range(psi.ndim):
        psi = _fill_halos_axis(psi, halo, axis, comm)

    return psi

def _domain_decomposition_slices(global_shape, dims, coords):
    starts = []
    local_sizes = []

    for axis in range(len(global_shape)):
        ranks_in_axis = dims[axis]
        rank_id = coords[axis]

        base, rem = divmod(global_shape[axis], ranks_in_axis)

        local_n = base + (1 if rank_id < rem else 0)
        start = rank_id * base + min(rank_id, rem)

        starts.append(start)
        local_sizes.append(local_n)

    return starts, local_sizes

def _partition_fields(psi_global, C_global, starts, local_sizes, halo):
    ndim = psi_global.ndim

    psi_slice = tuple(slice(starts[axis], starts[axis] + local_sizes[axis] + 2 * halo) for axis in range(ndim))

    psi_local = psi_global[psi_slice]

    C_local = []

    for vel_axis, C_axis_global in enumerate(C_global):
        C_slice = tuple(slice(starts[axis], starts[axis] + local_sizes[axis] + 2 * halo + (1 if axis == vel_axis else 0)) for axis in range(ndim))

        C_local.append(C_axis_global[C_slice])

    return psi_local, tuple(C_local)

def _gather_global(psi_local, halo, global_shape, comm):
    dims, _, coords = comm.Get_topo()
    size = comm.Get_size()
    ndim = psi_local.ndim

    interior = tuple(slice(halo, None if halo == 0 else -halo) for _ in range(ndim))

    psi_inner = psi_local[interior]

    _, local_sizes = _domain_decomposition_slices(global_shape, dims, coords)

    max_local_sizes = tuple(global_shape[axis] // dims[axis] + (1 if global_shape[axis] % dims[axis] else 0) for axis in range(ndim))
    pad_widths = tuple((0, max_local_sizes[axis] - local_sizes[axis]) for axis in range(ndim))

    psi_tile = jnp.pad(psi_inner, pad_widths)
    gathered = mpi4jax.allgather(psi_tile, comm=comm)

    padded_global_shape = tuple(global_shape[axis] + 2 * halo for axis in range(ndim))

    psi_full = jnp.zeros(padded_global_shape, dtype=psi_inner.dtype)

    for source in range(size):
        source_coords = comm.Get_coords(source)

        source_starts, source_sizes = _domain_decomposition_slices(global_shape, dims, source_coords)

        src_slice = tuple(slice(0, source_sizes[axis]) for axis in range(ndim))
        dst_slice = tuple(slice(halo + source_starts[axis], halo + source_starts[axis] + source_sizes[axis]) for axis in range(ndim))

        psi_full = psi_full.at[dst_slice].set(gathered[source][src_slice])

    return fill_periodic_halos(psi_full, halo)

def make_mpi_solver(n_dims):
    dims = MPI.Compute_dims(MPI.COMM_WORLD.Get_size(), [0] * n_dims)

    comm = MPI.COMM_WORLD.Create_cart(dims, periods=[True] * n_dims, reorder=False)
    fill_halos_fn = partial(_fill_halos, comm=comm)

    @partial(
        jax.jit,
        static_argnames=("halo", "n_iters"),
        donate_argnums=(0,),
    )
    def _solve_mpi_padded(psi_global, C_global, nt, halo=1, n_iters=1):
        dims, _, coords = comm.Get_topo()

        global_shape = tuple(psi_global.shape[axis] - 2 * halo for axis in range(psi_global.ndim))

        starts, local_sizes = _domain_decomposition_slices(global_shape, dims, coords)

        psi_local, C_local = _partition_fields(psi_global, C_global, starts, local_sizes, halo)

        psi_local = _solve_impl(psi_local, C_local, nt, halo, n_iters, fill_halos_fn=fill_halos_fn)

        return _gather_global(psi_local, halo, global_shape, comm)

    def solve_mpi(psi_global, C_global, nt, halo=1, n_iters=1):
        psi_global_padded = _pad_field(psi_global, halo)
        C_global_padded = _pad_field(C_global, halo)
        res_padded = _solve_mpi_padded(psi_global_padded, C_global_padded, nt, halo, n_iters)

        return _unpad_field(res_padded, halo)

    return comm, solve_mpi