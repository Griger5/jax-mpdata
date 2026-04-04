import numpy as np
import xarray as xr

def gaussian_2d(height, width):
    y, x = np.ogrid[:height, :width]
    cy, cx = (height - 1) / 2, (width - 1) / 2
    sigma = min(height, width) / 6

    return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

if __name__ == "__main__":
    size_x = 200
    size_y = 300
    halo = 1
    
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

    ds.attrs["steps"] = 500
    ds.attrs["size_x"] = size_x
    ds.attrs["size_y"] = size_y
    ds.attrs["halo"] = halo

    ds.to_netcdf("benchmarks/data/gaussian2d_big.nc", format="NETCDF4")

    size_x = 20
    size_y = 30
    halo = 1
    
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

    ds.attrs["steps"] = 50
    ds.attrs["size_x"] = size_x
    ds.attrs["size_y"] = size_y
    ds.attrs["halo"] = halo

    ds.to_netcdf("benchmarks/data/gaussian2d_small.nc", format="NETCDF4")