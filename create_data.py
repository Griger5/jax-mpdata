import numpy as np
import xarray as xr

def gaussian_2d(height, width):
    y, x = np.ogrid[:height, :width]
    cy, cx = (height - 1) / 2, (width - 1) / 2
    sigma = min(height, width) / 6

    return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2), dtype="float32")

def create_gaussian_benchmark(size_x, size_y, steps, halo, n_iters, name):
    psi = gaussian_2d(size_x, size_y)
    Cx = np.full((size_x + 1 + 2 * halo, size_y + 2 * halo), -0.2, dtype="float32")
    Cy = np.full((size_x + 2 * halo, size_y + 1 + 2 * halo), 0.5, dtype="float32")

    ds = xr.Dataset(
        {
            "psi": (("x", "y"), psi),
            "Cx": (("x_u", "y_u"), Cx),
            "Cy": (("x_v", "y_v"), Cy),
        },
        coords={
            "x": np.arange(size_x),
            "y": np.arange(size_y),
            "x_u": np.arange(Cx.shape[0]),
            "y_u": np.arange(Cx.shape[1]),
            "x_v": np.arange(Cy.shape[0]),
            "y_v": np.arange(Cy.shape[1]),
        }
    )

    ds.attrs["steps"] = steps
    ds.attrs["size_x"] = size_x
    ds.attrs["size_y"] = size_y
    ds.attrs["halo"] = halo
    ds.attrs["n_iters"] = n_iters

    ds.to_netcdf(f"benchmarks/data/{name}.nc", format="NETCDF4")

if __name__ == "__main__":
    # create_gaussian_benchmark(200, 300, 500, 1, 1, "gaussian2d_big_upwind")
    # create_gaussian_benchmark(20, 30, 50, 1, 1, "gaussian2d_small_upwind")

    # create_gaussian_benchmark(200, 300, 500, 1, 3, "gaussian2d_big_mpdata")
    # create_gaussian_benchmark(20, 30, 50, 1, 3, "gaussian2d_small_mpdata")

    for n_iters in range(1,4):
        for n in np.logspace(5, 7, num = 3, base = 2).astype(int):
            create_gaussian_benchmark(n, n, 100, 1, n_iters, f"gaussian2d_{n}_{n_iters}")
        
