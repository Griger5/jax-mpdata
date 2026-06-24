import time

import jax
import jax.numpy as jnp
from jax import lax

import matplotlib.pyplot as plt

real_t = "float32"

cpu_device = jax.devices("cpu")[0]

try:
    gpu_device = jax.devices("gpu")[0]
except:
    gpu_device = None

def donorcell(psi_l, psi_r, C):
    return ((C + jnp.abs(C)) * psi_l + (C - jnp.abs(C)) * psi_r) / 2

def mpdata_frac(nom, den, eps=1e-12):
    return nom / (den + eps)

def fill_halos_axis(psi, halo, axis):
    psi = jnp.moveaxis(psi, axis, 0)

    psi = psi.at[:halo].set(psi[-2 * halo:-halo])
    psi = psi.at[-halo:].set(psi[halo:2 * halo])

    return jnp.moveaxis(psi, 0, axis)

def fill_halos(psi, halo):
    for axis in range(psi.ndim):
        psi = fill_halos_axis(psi, halo, axis)

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

@jax.jit(static_argnums=(3, 4), donate_argnums=(0,))
def solve(psi0, C, nt, halo=1, n_iters=1):
    def body(n, psi):
        return step(psi, C, halo, n_iters)
    return lax.fori_loop(0, nt, body, psi0)

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

            psi0, Cx, Cy = init(nx, ny, halo)

            quicklook(psi0, halo)

            psi_final = solve(psi0, (Cx, Cy), nt, halo, n_iters).block_until_ready()

            quicklook(psi_final, halo)
            plt.show()

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