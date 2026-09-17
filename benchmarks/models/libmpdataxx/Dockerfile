FROM python:3.13-slim

# 1. Install C++ build tools, Blitz++, Boost, and NetCDF dependencies (mirroring your CI)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    libnetcdf-dev \
    libhdf5-dev \
    libblitz0-dev \
    libboost-thread-dev \
    libboost-date-time-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-timer-dev \
    libboost-filesystem-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# 2. Clone and install the underlying C++ libmpdata++ library system-wide
RUN git clone https://github.com/igfuw/libmpdataxx.git && \
    cd libmpdataxx/libmpdata++ && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    make install && \
    cd /tmp && rm -rf libmpdataxx

# Update shared library cache so Python can find the newly installed C++ libs
RUN ldconfig

WORKDIR /app
COPY . .

RUN rm -rf build/ dist/ *.egg-info

# 3. Build and install the nanobind Python bindings
RUN pip install --no-cache-dir .

# 4. Install data-parsing dependencies required by worker.py
RUN pip install --no-cache-dir xarray netCDF4

CMD ["python", "-c", "print('Ready')"]