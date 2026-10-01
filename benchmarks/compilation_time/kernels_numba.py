import os
import platform

import numpy as np
import numba
from numba import njit, prange

@njit(cache=False)
def numba_loop_kernel(x, iterations):
    y = x.copy()
    for _ in range(iterations):
        y = np.sin(y) * np.cos(y) + np.exp(-np.abs(y))
    return y

@njit(parallel=True, cache=False)
def numba_loop_parallel_kernel(x, iterations):
    y = x.copy()
    for _ in range(iterations):
        out = np.empty_like(y)
        for i in prange(y.size):
            out[i] = np.sin(y[i]) * np.cos(y[i]) + np.exp(-np.abs(y[i]))
        y = out
    return y

@njit(cache=False)
def numba_stencil_5pt(grid):
    out = np.zeros_like(grid)

    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            out[i, j] = (
                grid[i, j]
                + grid[i - 1, j]
                + grid[i + 1, j]
                + grid[i, j - 1]
                + grid[i, j + 1]
            ) / 5.0

    return out

@njit(parallel=True, cache=False)
def numba_stencil_5pt_parallel(grid):
    out = np.zeros_like(grid)

    for i in prange(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            out[i, j] = (
                grid[i, j]
                + grid[i - 1, j]
                + grid[i + 1, j]
                + grid[i, j - 1]
                + grid[i, j + 1]
            ) / 5.0

    return out

@njit(cache=False)
def numba_scalar_piecewise(x):
    if x > 0:
        return np.sin(x)
    else:
        return np.cos(x) * x * x

@njit(cache=False)
def numba_array_piecewise(x):
    out = np.empty_like(x)

    x_flat = x.ravel()
    out_flat = out.ravel()

    for i in range(x_flat.size):
        xi = x_flat[i]

        if xi > 0:
            out_flat[i] = np.sin(xi)
        else:
            out_flat[i] = np.cos(xi) * xi * xi

    return out

@njit(parallel=True, cache=False)
def numba_array_piecewise_parallel(x):
    out = np.empty_like(x)

    x_flat = x.ravel()
    out_flat = out.ravel()

    for i in prange(x_flat.size):
        xi = x_flat[i]

        if xi > 0:
            out_flat[i] = np.sin(xi)
        else:
            out_flat[i] = np.cos(xi) * xi * xi

    return out