from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

import mpi4jax
from mpi4py import MPI

COMM = None

def setup_comm(ndim, periods=True):
    size = MPI.COMM_WORLD.Get_size()
    dims = MPI.Compute_dims(size, [0] * ndim)

    COMM = MPI.COMM_WORLD.Create_cart(dims, periods=periods, reorder=False)

    return COMM

def donorcell(psi_l, psi_r, C):
    return ((C + jnp.abs(C)) * psi_l + (C - jnp.abs(C)) * psi_r) / 2

def mpdata_frac(nom, den, eps=1e-12):
    return nom / (den + eps)

def fill_halos_axis(psi, halo, axis, no_mpi=False):
    psi = jnp.moveaxis(psi, axis, 0)

    if COMM is None or no_mpi:
        psi = psi.at[:halo].set(psi[-2 * halo:-halo])
        psi = psi.at[-halo:].set(psi[halo:2 * halo])
        return jnp.moveaxis(psi, 0, axis)

    lower_src, upper_dst = COMM.Shift(axis, 1)
    upper_src, lower_dst = COMM.Shift(axis, -1)

    send_upper = psi[-2 * halo:-halo]
    recv_lower = mpi4jax.sendrecv(
        send_upper,
        recvbuf=psi[:halo],
        source=lower_src,
        dest=upper_dst,
        sendtag=2 * axis,
        recvtag=2 * axis,
        comm=COMM,
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
        comm=COMM,
    )
    psi = psi.at[-halo:].set(recv_upper)

    return jnp.moveaxis(psi, 0, axis)

def fill_halos(psi, halo, no_mpi=False):
    for axis in range(psi.ndim):
        psi = fill_halos_axis(psi, halo, axis, no_mpi)

    return psi

def compute_flux_axis(psi, C_self, halo, axis):
    psi_ax = jnp.moveaxis(psi, axis, 0)
    C_ax = jnp.moveaxis(C_self, axis, 0)

    n = psi_ax.shape[0]
    i = slice(halo, n - halo)
    i_plus = slice(halo + 1, n - halo + 1)
    i_minus = slice(halo - 1, n - halo - 1)

    flux_right = donorcell(psi_ax[i], psi_ax[i_plus], C_ax[i_plus])
    flux_left = donorcell(psi_ax[i_minus], psi_ax[i], C_ax[i])
    flux = flux_right - flux_left

    flux_full = jnp.zeros_like(psi_ax)
    flux_full = flux_full.at[i].set(flux)

    return jnp.moveaxis(flux_full, 0, axis)

def advop(psi, C, halo):
    total_flux = jnp.zeros_like(psi)
    for axis in range(psi.ndim):
        total_flux = total_flux + compute_flux_axis(psi, C[axis], halo, axis)
    return psi - total_flux

def compute_antidiff_axis(psi, C, halo, axis):
    ndim = psi.ndim
    N = [psi.shape[ax] - 2 * halo for ax in range(ndim)]

    base_slices = []
    for ax in range(ndim):
        if ax == axis:
            base_slices.append(slice(halo, halo + N[ax] + 1))
        else:
            base_slices.append(slice(halo, halo + N[ax]))

    cell_left_slices = list(base_slices)
    cell_left_slices[axis] = slice(halo - 1, halo + N[axis])

    psi_r = psi[tuple(base_slices)]
    psi_l = psi[tuple(cell_left_slices)]
    A = mpdata_frac(psi_r - psi_l, psi_r + psi_l)

    Cd_face = C[axis][tuple(base_slices)]
    cross_term = jnp.zeros_like(A)

    for dp in range(ndim):
        if dp == axis:
            continue

        up_slices = list(base_slices)
        up_slices[dp] = slice(halo + 1, halo + N[dp] + 1)

        down_slices = list(base_slices)
        down_slices[dp] = slice(halo - 1, halo + N[dp] - 1)

        up_left_slices = list(cell_left_slices)
        up_left_slices[dp] = slice(halo + 1, halo + N[dp] + 1)

        down_left_slices = list(cell_left_slices)
        down_left_slices[dp] = slice(halo - 1, halo + N[dp] - 1)

        psi_up = psi[tuple(up_slices)] + psi[tuple(up_left_slices)]
        psi_down = psi[tuple(down_slices)] + psi[tuple(down_left_slices)]

        B_dp = mpdata_frac(psi_up - psi_down, psi_up + psi_down) / 2

        c_up_slices = list(base_slices)
        c_up_slices[dp] = slice(halo + 1, halo + N[dp] + 1)

        c_down_slices = list(base_slices)
        c_down_slices[dp] = slice(halo, halo + N[dp])

        c_up_left_slices = list(cell_left_slices)
        c_up_left_slices[dp] = slice(halo + 1, halo + N[dp] + 1)

        c_down_left_slices = list(cell_left_slices)
        c_down_left_slices[dp] = slice(halo, halo + N[dp])

        C_dp_bar = (C[dp][tuple(c_up_slices)] + C[dp][tuple(c_up_left_slices)] + C[dp][tuple(c_down_slices)] + C[dp][tuple(c_down_left_slices)]) / 4

        cross_term = cross_term + Cd_face * C_dp_bar * B_dp

    C_antidiff = jnp.abs(Cd_face) * (1 - jnp.abs(Cd_face)) * A - cross_term

    return C_antidiff

def step(psi, C, halo, n_iters):
    psi = fill_halos(psi, halo)
    psi = advop(psi, C, halo)

    C_uncorr = C

    for _ in range(1, n_iters):
        psi = fill_halos(psi, halo)

        C_corr = []
        for d in range(psi.ndim):
            C_antidiff = compute_antidiff_axis(psi, C_uncorr, halo, d)
            
            pad_width = [(halo, halo)] * psi.ndim

            C_antidiff = jnp.pad(C_antidiff, pad_width)
            C_antidiff = fill_halos(C_antidiff, halo)
            
            C_corr.append(C_antidiff)

        C_corr = tuple(C_corr)
        psi = advop(psi, C_corr, halo)
        C_uncorr = C_corr

    return psi

def _solve_impl(psi0, C, nt, halo=1, n_iters=1):
    def body(n, psi):
        return step(psi, C, halo, n_iters)
    return lax.fori_loop(0, nt, body, psi0)

solve = jax.jit(
    _solve_impl,
    static_argnums=(3, 4),
    donate_argnums=(0,),
)

@partial(
    jax.jit,
    static_argnames=("halo", "n_iters", "batched_C"),
    donate_argnums=(0,),
)
def solve_batch(psi0_batch, C, nt, halo=1, n_iters=1, batched_C=False):
    C_in_axis = 0 if batched_C else None

    return jax.vmap(
        _solve_impl,
        in_axes=(0, C_in_axis, None, None, None),
        out_axes=0,
    )(psi0_batch, C, nt, halo, n_iters)

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

def _gather_global(psi_local, halo, global_shape):
    dims, _, coords = COMM.Get_topo()
    size = COMM.Get_size()
    ndim = psi_local.ndim

    interior = tuple(slice(halo, None if halo == 0 else -halo) for _ in range(ndim))

    psi_inner = psi_local[interior]

    _, local_sizes = _domain_decomposition_slices(global_shape, dims, coords)

    max_local_sizes = tuple(global_shape[axis] // dims[axis] + (1 if global_shape[axis] % dims[axis] else 0) for axis in range(ndim))
    pad_widths = tuple((0, max_local_sizes[axis] - local_sizes[axis]) for axis in range(ndim))

    psi_tile = jnp.pad(psi_inner, pad_widths)
    gathered = mpi4jax.allgather(psi_tile, comm=COMM)

    padded_global_shape = tuple(global_shape[axis] + 2 * halo for axis in range(ndim))

    psi_full = jnp.zeros(padded_global_shape, dtype=psi_inner.dtype)

    for source in range(size):
        source_coords = COMM.Get_coords(source)

        source_starts, source_sizes = _domain_decomposition_slices(global_shape, dims, source_coords)

        src_slice = tuple(slice(0, source_sizes[axis]) for axis in range(ndim))
        dst_slice = tuple(slice(halo + source_starts[axis], halo + source_starts[axis] + source_sizes[axis]) for axis in range(ndim))

        psi_full = psi_full.at[dst_slice].set(gathered[source][src_slice])

    return fill_halos(psi_full, halo, no_mpi=True)

@partial(
    jax.jit,
    static_argnums=(3, 4),
    donate_argnums=(0,)
)
def solve_mpi(psi_global, C_global, nt, halo=1, n_iters=1):
    dims, _, coords = COMM.Get_topo()

    global_shape = tuple(psi_global.shape[axis] - 2 * halo for axis in range(psi_global.ndim))

    starts, local_sizes = _domain_decomposition_slices(global_shape, dims, coords)

    psi_local, C_local = _partition_fields(psi_global, C_global, starts, local_sizes, halo)

    psi_local = _solve_impl(psi_local, C_local, nt, halo, n_iters)

    return _gather_global(psi_local, halo, global_shape)