from jumbax_api.mpdata import setup_mpdata, solve_mpdata

import jax_mpdata

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from jumbax_core.common import Dtype
from jumbax_core.mpdata import BoundaryCondition

def gaussian_2d(h: int, w: int) -> jax.Array:
    y = jnp.arange(h)[:, None]
    x = jnp.arange(w)[None, :]
    
    cy, cx = (h - 1) / 2, (w - 1) / 2
    sigma = min(h, w) / 6

    return jnp.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def init(nx, ny, halo):
    psi0 = jnp.zeros((nx + 2*halo, ny + 2*halo), dtype=jnp.float32)

    Cx = jnp.full((nx + 1 + 2*halo, ny + 2*halo), -0.2, dtype=jnp.float32)
    Cy = jnp.full((nx + 2*halo, ny + 1 + 2*halo), 0.5, dtype=jnp.float32)

    psi0 = psi0.at[halo:halo+nx, halo:halo+ny].set(gaussian_2d(nx, ny))

    return psi0, Cx, Cy

def quicklook(arg, halo):
	fig, ax = plt.subplots()
	
	im = ax.imshow(arg[halo:-halo, halo:-halo], vmax=1)
	fig.colorbar(im, ax=ax)

if __name__ == "__main__":
    nx = 20
    ny = 30
    halo = 1
    nt = 50

    psi0, Cx, Cy = init(nx, ny, halo)

    with setup_mpdata("jax", n_dims=2, n_iters=3, epsilon=1e-12, dtype=Dtype.float32, halo=1, boundary_conditions=[BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC], mpi=False, n_threads=None):
        quicklook(psi0, halo)
        psi = solve_mpdata(psi0, (Cx, Cy), nt)
        # psi = solve_mpdata(psi0[halo:-halo, halo:-halo], (Cx[halo:-halo, halo:-halo], Cy[halo:-halo, halo:-halo]), nt)
        quicklook(psi, halo)
        plt.show()
