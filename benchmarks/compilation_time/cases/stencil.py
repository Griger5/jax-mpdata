import time

import jax
import numpy as np
import numba

from benchmarks.compilation_time.kernels_jax import jax_stencil_5pt
from benchmarks.compilation_time.kernels_numba import numba_stencil_5pt, numba_stencil_5pt_parallel
from benchmarks.compilation_time.measure import enable_jax_dtype, first_warm, get_dtype, jax_compile_time

def jax_stencil(spec):
    dtype = get_dtype(spec)
    enable_jax_dtype(dtype)

    shape = tuple(spec.get("shape", (256, 256)))
    x_spec = jax.ShapeDtypeStruct(shape, dtype)

    compile_time = jax_compile_time(jax_stencil_5pt, args=(x_spec,))

    return {"compile_time": compile_time}

def _numba_stencil(spec, parallel):
    dtype = get_dtype(spec)
    call_shape = tuple(spec.get("call_shape", (4, 4)))
    warm = int(spec.get("warm", 3))

    if parallel:
        threads = spec.get("threads")
        if threads is not None:
            numba.set_num_threads(int(threads))
        kernel = numba_stencil_5pt_parallel
    else:
        kernel = numba_stencil_5pt

    x = np.ones(call_shape, dtype=dtype)

    def call():
        start = time.perf_counter()
        kernel(x)
        return time.perf_counter() - start

    return first_warm(call, warm)

def numba_stencil_serial(spec):
    return _numba_stencil(spec, parallel=False)

def numba_stencil_parallel(spec):
    return _numba_stencil(spec, parallel=True)