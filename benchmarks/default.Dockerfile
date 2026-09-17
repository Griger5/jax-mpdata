FROM python:3.13-slim
RUN apt-get update && apt-get install -y libnetcdf-dev libhdf5-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy xarray netCDF4 jax[cpu] PyMPDATA numba tqdm
