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

def advop(psi, Cx, Cy, halo):
    nx = psi.shape[0] - 2*halo
    ny = psi.shape[1] - 2*halo

    i = slice(halo, halo+nx)
    j = slice(halo, halo+ny)

    flux_x_right = donorcell(psi[i, j], psi[i.start+1:i.stop+1, j], Cx[i.start+1:i.stop+1, j])
    flux_x_left = donorcell(psi[i.start-1:i.stop-1, j], psi[i, j], Cx[i.start-1:i.stop-1, j])

    flux_y_right = donorcell(psi[i, j], psi[i, j.start+1:j.stop+1], Cy[i, j.start+1:j.stop+1])
    flux_y_left = donorcell(psi[i, j.start-1:j.stop-1], psi[i, j], Cy[i, j.start-1:j.stop-1])

    return psi.at[i, j].set(psi[i, j] - (flux_x_right - flux_x_left) - (flux_y_right - flux_y_left))

def step(psi, Cx, Cy, halo):
    psi = fill_halos(psi, halo)
    psi = advop(psi, Cx, Cy, halo)
    return psi

@jax.jit(static_argnums=(4,))
def solve(psi0, Cx, Cy, nt, halo = 1):
    def body(n, psi):
        return step(psi, Cx, Cy, halo)

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
    # nx, ny = 200, 300
    # nt = 500
    # halo = 1

    # psi0, Cx, Cy = init(nx, ny, halo)
    # quicklook(psi0, halo)
    # plt.show()

    # start = time.perf_counter()
    # psi_final = solve(psi0, Cx, Cy, nt, halo)
    # end = time.perf_counter()

    # quicklook(psi_final, halo)
    # plt.show()

    # print(f"Time: {end - start:.6f} s")

    times = {cpu_device: [], gpu_device : []}

    for _ in range(10):
        for device, name in zip([cpu_device, gpu_device], ["CPU", "GPU"]):
            print("#########################      " + name + "     ##########################")
            with jax.default_device(device):
                nx = 200
                ny = 300
                halo = 1
                nt = 500

                psi0, Cx, Cy = init(nx, ny, halo)

                start = time.perf_counter()
                psi_final = solve(psi0, Cx, Cy, nt, halo)
                end = time.perf_counter()

                print(f"Time: {end - start:.6f} seconds")

                times[device].append(end - start)

            print(f"{min(times[device])=}")