FROM python:3.13-slim

# Install system dependencies required by netCDF4 (used by worker.py)
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the standard scientific and benchmark stack
RUN pip install --no-cache-dir \
    numpy \
    xarray \
    netCDF4 \
    jax[cpu] \
    PyMPDATA \
    numba \
    tqdm

CMD ["python", "-c", "print('Ready')"]