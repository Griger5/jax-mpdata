import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

import mpi4jax
from mpi4py import MPI

import matplotlib.pyplot as plt

real_t = "float32"

cpu_device = jax.devices("cpu")[0]

COMM = None

try:
    gpu_device = jax.devices("gpu")[0]
except:
    gpu_device = None

def setup_comm(ndim, periods=True):
    # MPI.Init_thread(3)
    size = MPI.COMM_WORLD.Get_size()
    dims = MPI.Compute_dims(size, [0] * ndim)

    COMM = MPI.COMM_WORLD.Create_cart(dims, periods=periods, reorder=False)

    return COMM

def donorcell(psi_l, psi_r, C):
    return ((C + jnp.abs(C)) * psi_l + (C - jnp.abs(C)) * psi_r) / 2

def mpdata_frac(nom, den, eps=1e-12):
    return nom / (den + eps)

# def fill_halos_axis(psi, halo, axis):
#     psi = jnp.moveaxis(psi, axis, 0)

#     psi = psi.at[:halo].set(psi[-2 * halo:-halo])
#     psi = psi.at[-halo:].set(psi[halo:2 * halo])

#     return jnp.moveaxis(psi, 0, axis)

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

def gaussian_2d(h: int, w: int) -> jax.Array:
    y = jnp.arange(h)[:, None]
    x = jnp.arange(w)[None, :]
    
    cy, cx = (h - 1) / 2, (w - 1) / 2
    sigma = min(h, w) / 6

    return jnp.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def init(nx, ny, halo):
    psi0 = jnp.zeros((nx + 2*halo, ny + 2*halo), dtype=real_t)

    Cx = jnp.full((nx + 1 + 2*halo, ny + 2*halo), -0.2, dtype=real_t)
    Cy = jnp.full((nx + 2*halo, ny + 1 + 2*halo), 0.5, dtype=real_t)

    psi0 = psi0.at[halo:halo+nx, halo:halo+ny].set(gaussian_2d(nx, ny))

    return psi0, Cx, Cy

def quicklook(arg, halo):
	fig, ax = plt.subplots()
	
	im = ax.imshow(arg[halo:-halo, halo:-halo], vmax=1)
	fig.colorbar(im, ax=ax)

if __name__ == "__main__":
    # nx, ny = 20, 30
    # nt = 50
    # halo = 1

    # psi0, Cx, Cy = init(nx, ny, halo)
    # quicklook(psi0, halo)
    # plt.show()

    # start = time.perf_counter()
    # psi_final = solve(psi0, Cx, Cy, nt, halo)
    # end = time.perf_counter()

    # quicklook(psi_final, halo)

    # psi0, Cx, Cy = init(nx, ny, halo)
    # psi_mp = solve(psi0, Cx, Cy, nt, halo, 3)
    # quicklook(psi_mp, halo)

    # plt.show()

    # print(f"Time: {end - start:.6f} s")

    times = {cpu_device: [], gpu_device : []}

    with jax.default_device(cpu_device):
        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            nx = int(20 * 1)
            ny = int(30 * 1)
            halo = 1
            n_iters = 3
            nt = 50

            # psi0, Cx, Cy = init(nx, ny, halo)

            # quicklook(psi0, halo)

            # print_jax_threads()

            # psi_final = solve(psi0, (Cx, Cy), nt, halo, n_iters).block_until_ready()

            # print_jax_threads()

            # quicklook(psi_final, halo)
            # plt.show()

            COMM = setup_comm(2, periods=True)

            psi0, Cx, Cy = init(nx, ny, halo)
            quicklook(psi0, halo)

            psi_final = solve_mpi(psi0, (Cx, Cy), nt, halo, n_iters)

            # jax.debug.visualize_array_sharding(psi_final)

            # if COMM.Get_rank() == 0:
            # print(MPI.Query_thread())

            # psi_final.block_until_ready()
            quicklook(psi_final, halo)
            plt.show()

                # MPI.Finalize()

    # for _ in range(10):
    #     for device, name in zip([cpu_device, gpu_device], ["CPU", "GPU"]):
    #         print("#########################      " + name + "     ##########################")
    #         with jax.default_device(device):
    #             nx = int(200 * 0.5)
    #             ny = int(300 * 0.5)
    #             halo = 1
    #             n_iters = 3
    #             nt = 100

    #             psi0, Cx, Cy = init(nx, ny, halo)

    #             start = time.perf_counter()
    #             psi_final = solve(psi0, Cx, Cy, nt, halo, n_iters).block_until_ready()
    #             end = time.perf_counter()

    #             print(f"Time: {end - start:.6f} seconds")

    #             times[device].append(end - start)

    #         print(f"{min(times[device])=}")

# if __name__ == "__main__":
#     GRID_SCALE_X = 2.0
#     GRID_SCALE_Y = 2.0

#     BATCH_SIZE = 64
#     NT = 250
#     HALO = 1
#     N_ITERS = 3

#     USE_GPU = False
#     SHOW_PLOTS = False
#     CHECK_RESULTS = True
#     DIFFERENT_FIELDS = True

#     nx = max(1, int(20 * GRID_SCALE_X))
#     ny = max(1, int(30 * GRID_SCALE_Y))

#     device = gpu_device if (USE_GPU and gpu_device is not None) else cpu_device

#     print("==================================================")
#     print("JAX MPDATA batched vs sequential quick test")
#     print("==================================================")
#     print(f"device       = {device}")
#     print(f"grid         = {nx} x {ny}")
#     print(f"halo         = {HALO}")
#     print(f"steps        = {NT}")
#     print(f"MPDATA iters = {N_ITERS}")
#     print(f"batch size   = {BATCH_SIZE}")
#     print(f"different fields = {DIFFERENT_FIELDS}")
#     print("==================================================")

#     with jax.default_device(device):
#         psi0, Cx, Cy = init(nx, ny, HALO)
#         C = (Cx, Cy)

#         if DIFFERENT_FIELDS:
#             amps = jnp.linspace(0.9, 1.1, BATCH_SIZE, dtype=real_t)
#             amps = amps[:, None, None]
#             psi0_batch = psi0[None, ...] * amps
#         else:
#             psi0_batch = jnp.repeat(psi0[None, ...], BATCH_SIZE, axis=0)

#         print(f"psi0.shape      = {psi0.shape}")
#         print(f"psi0_batch.shape = {psi0_batch.shape}")

#         print("Warming up solve_batch...")
#         _ = solve_batch(
#             psi0_batch.copy(),
#             C,
#             NT,
#             halo=HALO,
#             n_iters=N_ITERS,
#         ).block_until_ready()

#         print("Warming up solve...")
#         _ = solve(
#             psi0.copy(),
#             C,
#             NT,
#             HALO,
#             N_ITERS,
#         ).block_until_ready()

#         print("Running timed batched solve...")

#         batch_in = psi0_batch.copy()

#         start = time.perf_counter()
#         psi_batch_out = solve_batch(
#             batch_in,
#             C,
#             NT,
#             halo=HALO,
#             n_iters=N_ITERS,
#         ).block_until_ready()
#         batch_time = time.perf_counter() - start

#         batch_in = None

#         print("Preparing sequential inputs...")
#         sequential_inputs = [
#             psi0_batch[i].copy()
#             for i in range(BATCH_SIZE)
#         ]

#         print("Running timed sequential solves...")

#         sequential_outputs = []

#         start = time.perf_counter()
#         for i in range(BATCH_SIZE):
#             psi_i = sequential_inputs[i]

#             psi_out_i = solve(
#                 psi_i,
#                 C,
#                 NT,
#                 HALO,
#                 N_ITERS,
#             ).block_until_ready()

#             psi_i = None
#             sequential_inputs[i] = None

#             sequential_outputs.append(psi_out_i)

#         sequential_time = time.perf_counter() - start

#         psi_sequential_out = jnp.stack(sequential_outputs, axis=0)

#         print("==================================================")
#         print("Timing results")
#         print("==================================================")
#         print(f"Batched total time      = {batch_time:.6f} s")
#         print(f"Sequential total time   = {sequential_time:.6f} s")
#         print()
#         print(f"Batched time / field    = {batch_time / BATCH_SIZE:.6f} s")
#         print(f"Sequential time / field = {sequential_time / BATCH_SIZE:.6f} s")
#         print()
#         print(f"Batched throughput      = {BATCH_SIZE / batch_time:.3f} fields/s")
#         print(f"Sequential throughput   = {BATCH_SIZE / sequential_time:.3f} fields/s")
#         print()
#         print(f"Sequential / batched wall-clock ratio = {sequential_time / batch_time:.3f}x")
#         print("==================================================")

#         if CHECK_RESULTS:
#             max_abs_diff = float(jnp.max(jnp.abs(psi_batch_out - psi_sequential_out)))
#             close = bool(
#                 jnp.allclose(
#                     psi_batch_out,
#                     psi_sequential_out,
#                     rtol=1e-5,
#                     atol=1e-6,
#                 )
#             )

#             print("Result comparison:")
#             print(f"  max abs diff = {max_abs_diff:.6e}")
#             print(f"  allclose     = {close}")

#             if not close:
#                 print("WARNING: batched and sequential results differ more than expected.")
#             print("==================================================")

#         if SHOW_PLOTS:
#             quicklook(psi0, HALO)
#             plt.title("Initial field")

#             quicklook(psi_batch_out[0], HALO)
#             plt.title("Batched output, member 0")

#             quicklook(psi_sequential_out[0], HALO)
#             plt.title("Sequential output, member 0")

#             if BATCH_SIZE > 1:
#                 quicklook(psi_batch_out[-1], HALO)
#                 plt.title("Batched output, last member")

#                 quicklook(psi_sequential_out[-1], HALO)
#                 plt.title("Sequential output, last member")
            
#             plt.show()