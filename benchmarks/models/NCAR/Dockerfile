FROM python:3.13-slim

# 1. Install C++/Fortran build tools and system dependencies
# Note the addition of 'gfortran' for the legacy and modern Fortran files.
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    python3-dev \
    libnetcdf-dev \
    libhdf5-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy the project files into the image
COPY . .

RUN rm -rf build/ dist/ *.egg-info

# 3. Build and install the nanobind extension
# CMake will automatically compile both the C++ and Fortran code.
RUN pip install --no-cache-dir .

# 4. Install data-parsing dependencies required by worker.py
RUN pip install --no-cache-dir xarray netCDF4

CMD ["python", "-c", "print('Ready')"]