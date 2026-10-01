import time

import jax
import numpy as np

from benchmarks.compilation_time.kernels_jax import jax_array_piecewise, jax_scalar_piecewise
from benchmarks.compilation_time.kernels_numba import numba_array_piecewise, numba_scalar_piecewise, numba_array_piecewise_parallel
from benchmarks.compilation_time.measure import enable_jax_dtype, first_warm, get_dtype, jax_compile_time

def jax_control_scalar(spec):
    dtype = get_dtype(spec)
    enable_jax_dtype(dtype)

    x_spec = jax.ShapeDtypeStruct((), dtype)
    compile_time = jax_compile_time(jax_scalar_piecewise, args=(x_spec,))
    return {"compile_time": compile_time}

def numba_control_scalar(spec):
    dtype = get_dtype(spec)
    warm = int(spec.get("warm", 3))
    x = dtype(1.0)

    def call():
        start = time.perf_counter()
        numba_scalar_piecewise(x)
        return time.perf_counter() - start

    return first_warm(call, warm)

def jax_control_array(spec):
    dtype = get_dtype(spec)
    enable_jax_dtype(dtype)

    shape = tuple(spec.get("shape", (1024,)))
    x_spec = jax.ShapeDtypeStruct(shape, dtype)
    compile_time = jax_compile_time(jax_array_piecewise, args=(x_spec,))
    return {"compile_time": compile_time}

def numba_control_array(spec):
    dtype = get_dtype(spec)
    call_shape = tuple(spec.get("call_shape", (8,)))
    warm = int(spec.get("warm", 3))
    x = np.ones(call_shape, dtype=dtype)

    def call():
        start = time.perf_counter()
        numba_array_piecewise(x)
        return time.perf_counter() - start

    return first_warm(call, warm)

def numba_control_array_parallel(spec):
    import numba

    dtype = get_dtype(spec)
    call_shape = tuple(spec.get("call_shape", (8,)))
    warm = int(spec.get("warm", 3))

    threads = spec.get("threads")
    if threads is not None:
        numba.set_num_threads(int(threads))

    x = np.ones(call_shape, dtype=dtype)

    def call():
        start = time.perf_counter()
        numba_array_piecewise_parallel(x)
        return time.perf_counter() - start

    return first_warm(call, warm)