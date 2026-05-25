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

def fill_halos(psi, halo):
    nx, ny = psi.shape

    left_edge = lax.dynamic_slice(psi, (halo, 0), (halo, ny))
    psi = lax.dynamic_update_slice(psi, left_edge, (nx - halo, 0))

    right_edge = lax.dynamic_slice(psi, (nx - 2*halo, 0), (halo, ny))
    psi = lax.dynamic_update_slice(psi, right_edge, (0, 0))

    top_edge = lax.dynamic_slice(psi, (0, halo), (nx, halo))
    psi = lax.dynamic_update_slice(psi, top_edge, (0, ny - halo))

    bottom_edge = lax.dynamic_slice(psi, (0, ny - 2*halo), (nx, halo))
    psi = lax.dynamic_update_slice(psi, bottom_edge, (0, 0))

    return psi

def mpdata_frac(nom, den):
    return jnp.where(den > 0, nom / den, 0.0)

def mpdata_C_antidiff_x(psi, Cx, Cy, x_faces, j):
    cell_left = slice(x_faces.start - 1, x_faces.stop - 1)
    j_up = slice(j.start + 1, j.stop + 1)
    j_down = slice(j.start - 1, j.stop - 1)

    psi_r = psi[x_faces, j]
    psi_l = psi[cell_left, j]

    A = mpdata_frac(psi_r - psi_l, psi_r + psi_l)

    B = mpdata_frac(
        psi[x_faces, j_up] + psi[cell_left, j_up] - psi[x_faces, j_down] - psi[cell_left, j_down],
        psi[x_faces, j_up] + psi[cell_left, j_up] + psi[x_faces, j_down] + psi[cell_left, j_down],
    ) / 2

    Cy_bar = (Cy[x_faces, j_up] + Cy[cell_left, j_up] + Cy[x_faces, j] + Cy[cell_left, j]) / 4
    Cx_face = Cx[x_faces, j]

    return jnp.abs(Cx_face) * (1 - jnp.abs(Cx_face)) * A - Cx_face * Cy_bar * B

def mpdata_C_antidiff_y(psi, Cx, Cy, i, y_faces):
    cell_below = slice(y_faces.start - 1, y_faces.stop - 1)
    i_right = slice(i.start + 1, i.stop + 1)
    i_left = slice(i.start - 1, i.stop - 1)

    psi_r = psi[i, y_faces]
    psi_l = psi[i, cell_below]

    A = mpdata_frac(psi_r - psi_l, psi_r + psi_l)

    B = mpdata_frac(
        psi[i_right, y_faces] + psi[i_right, cell_below] - psi[i_left, y_faces] - psi[i_left, cell_below],
        psi[i_right, y_faces] + psi[i_right, cell_below] + psi[i_left, y_faces] + psi[i_left, cell_below],
    ) / 2

    Cx_bar = (Cx[i_right, y_faces] + Cx[i_right, cell_below] + Cx[i, y_faces] + Cx[i, cell_below]) / 4
    Cy_face = Cy[i, y_faces]

    return jnp.abs(Cy_face) * (1 - jnp.abs(Cy_face)) * A - Cy_face * Cx_bar * B

def advop(psi, Cx, Cy, halo):
    nx = psi.shape[0] - 2 * halo
    ny = psi.shape[1] - 2 * halo

    i = slice(halo, halo + nx)
    j = slice(halo, halo + ny)

    flux_x_right = donorcell(psi[i, j], psi[i.start+1:i.stop+1, j], Cx[i.start+1:i.stop+1, j])
    flux_x_left = donorcell(psi[i.start-1:i.stop-1, j], psi[i, j], Cx[i, j])

    flux_y_right = donorcell(psi[i, j], psi[i, j.start+1:j.stop+1], Cy[i, j.start+1:j.stop+1])
    flux_y_left = donorcell(psi[i, j.start-1:j.stop-1], psi[i, j], Cy[i, j])
    
    return psi.at[i, j].set(psi[i, j] - (flux_x_right - flux_x_left) - (flux_y_right - flux_y_left))

def step(psi, Cx, Cy, halo, n_iters):
    nx = psi.shape[0] - 2 * halo
    ny = psi.shape[1] - 2 * halo

    i = slice(halo, halo + nx)
    j = slice(halo, halo + ny)
    x_faces = slice(halo, halo + nx + 1)
    y_faces = slice(halo, halo + ny + 1)

    psi = fill_halos(psi, halo)
    psi = advop(psi, Cx, Cy, halo)

    Cx_uncorr, Cy_uncorr = Cx, Cy

    for _ in range(1, n_iters):
        psi = fill_halos(psi, halo)

        Cx_corr = jnp.pad(
            mpdata_C_antidiff_x(psi, Cx_uncorr, Cy_uncorr, x_faces, j),
            ((halo, halo), (halo, halo))
        )
        Cx_corr = fill_halos(Cx_corr, halo)

        Cy_corr = jnp.pad(
            mpdata_C_antidiff_y(psi, Cx_uncorr, Cy_uncorr, i, y_faces),
            ((halo, halo), (halo, halo)),
        )
        Cy_corr = fill_halos(Cy_corr, halo)

        psi = advop(psi, Cx_corr, Cy_corr, halo)
        Cx_uncorr, Cy_uncorr = Cx_corr, Cy_corr

    return psi

@jax.jit(static_argnums=(4, 5), donate_argnums=(0,))
def solve(psi0, Cx, Cy, nt, halo = 1, n_iters = 1):
    def body(n, psi):
        return step(psi, Cx, Cy, halo, n_iters)
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

    with jax.default_device(gpu_device):
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
                nx = int(200 * 0.5)
                ny = int(300 * 0.5)
                halo = 1
                n_iters = 3
                nt = 100

                psi0, Cx, Cy = init(nx, ny, halo)

                psi_final = solve(psi0, Cx, Cy, nt, halo, n_iters).block_until_ready()

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