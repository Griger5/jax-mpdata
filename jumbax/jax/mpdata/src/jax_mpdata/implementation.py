from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

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

def step(psi, C, halo, n_iters, fill_halos_fn=fill_halos):
    psi = fill_halos_fn(psi, halo)
    psi = advop(psi, C, halo)

    C_uncorr = C

    for _ in range(1, n_iters):
        psi = fill_halos_fn(psi, halo)

        C_corr = []
        for d in range(psi.ndim):
            C_antidiff = compute_antidiff_axis(psi, C_uncorr, halo, d)
            
            pad_width = [(halo, halo)] * psi.ndim

            C_antidiff = jnp.pad(C_antidiff, pad_width)
            C_antidiff = fill_halos_fn(C_antidiff, halo)
            
            C_corr.append(C_antidiff)

        C_corr = tuple(C_corr)
        psi = advop(psi, C_corr, halo)
        C_uncorr = C_corr

    return psi

def _solve_impl(psi0, C, nt, halo=1, n_iters=1, fill_halos_fn=fill_halos):
    def body(n, psi):
        return step(psi, C, halo, n_iters, fill_halos_fn)
    return lax.fori_loop(0, nt, body, psi0)

_solve_padded = jax.jit(
    _solve_impl,
    static_argnums=(3, 4),
    donate_argnums=(0,),
)

@partial(
    jax.jit,
    static_argnames=("halo", "n_iters", "batched_C"),
    donate_argnums=(0,),
)
def _solve_batch_padded(psi0_batch, C, nt, halo=1, n_iters=1, batched_C=False):
    C_in_axis = 0 if batched_C else None

    return jax.vmap(
        _solve_impl,
        in_axes=(0, C_in_axis, None, None, None),
        out_axes=0,
    )(psi0_batch, C, nt, halo, n_iters)

def _pad_field(field, halo):
    if isinstance(field, (tuple, list)):
        return type(field)(_pad_field(f, halo) for f in field)
    if halo == 0:
        return field
    
    pad_width = [(halo, halo)] * field.ndim

    return jnp.pad(field, pad_width, mode='edge')

def _unpad_field(field, halo):
    if isinstance(field, (tuple, list)):
        return type(field)(_unpad_field(f, halo) for f in field)
    if halo == 0:
        return field
    
    slices = tuple(slice(halo, -halo) for _ in range(field.ndim))

    return field[slices]

def solve(psi0, C, nt, halo=1, n_iters=1):
    psi0_padded = _pad_field(psi0, halo)
    C_padded = _pad_field(C, halo)
    res_padded = _solve_padded(psi0_padded, C_padded, nt, halo, n_iters)

    return _unpad_field(res_padded, halo)

@partial(
    jax.jit,
    static_argnames=("halo", "n_iters", "batched_C"),
    donate_argnums=(0,),
)
def solve_batch(psi0_batch, C, nt, halo=1, n_iters=1, batched_C=False):
    C_in_axis = 0 if batched_C else None

    return jax.vmap(
        solve,
        in_axes=(0, C_in_axis, None, None, None),
        out_axes=0,
    )(psi0_batch, C, nt, halo, n_iters)