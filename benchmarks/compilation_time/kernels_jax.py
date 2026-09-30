import jax
import jax.numpy as jnp

def jax_loop_unrolled_kernel(x, iterations=100):
    y = x
    for _ in range(iterations):
        y = jnp.sin(y) * jnp.cos(y) + jnp.exp(-jnp.abs(y))
    return y

def jax_loop_fori_kernel(x, iterations):
    def body(_, y):
        return jnp.sin(y) * jnp.cos(y) + jnp.exp(-jnp.abs(y))

    return jax.lax.fori_loop(0, iterations, body, x)

def jax_stencil_5pt(grid):
    out = jnp.zeros_like(grid)

    interior = (
        grid[1:-1, 1:-1]
        + grid[:-2, 1:-1]
        + grid[2:, 1:-1]
        + grid[1:-1, :-2]
        + grid[1:-1, 2:]
    ) / 5.0

    return out.at[1:-1, 1:-1].set(interior)

def jax_scalar_piecewise(x):
    return jax.lax.cond(
        x > 0,
        lambda x: jnp.sin(x),
        lambda x: jnp.cos(x) * x**2,
        x,
    )

def jax_array_piecewise(x):
    return jnp.where(
        x > 0,
        jnp.sin(x),
        jnp.cos(x) * x**2,
    )