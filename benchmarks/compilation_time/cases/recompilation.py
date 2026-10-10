import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.compilation_time.kernels_jax import jax_stencil_5pt
from benchmarks.compilation_time.kernels_numba import numba_stencil_5pt, numba_stencil_5pt_parallel
from benchmarks.compilation_time.measure import configure_jax, estimate_compile_time, get_dtype, set_numba_threads

def recompile_jax_stencil(spec):
    configure_jax(spec)

    dtype = get_dtype(spec)
    warm = int(spec.get("warm", 1))
    f = jax.jit(jax_stencil_5pt)

    times = {}
    details = {}

    for shape in spec.get("shapes", []):
        shape = tuple(shape)
        x = jnp.ones(shape, dtype=dtype)

        def call(x=x):
            start = time.perf_counter()
            y = f(x)
            y.block_until_ready()
            return time.perf_counter() - start

        measured = estimate_compile_time(call, warm)
        key = "x".join(str(d) for d in shape)
        times[key] = measured["compile_time"]
        details[key] = measured["details"]

    return {**times, "details": details}

def recompile_numba_stencil(spec):
    dtype = get_dtype(spec)
    warm = int(spec.get("warm", 1))

    times = {}
    details = {}

    for shape in spec.get("shapes", []):
        shape = tuple(shape)
        x = np.ones(shape, dtype=dtype)

        def call(x=x):
            start = time.perf_counter()
            numba_stencil_5pt(x)
            return time.perf_counter() - start

        measured = estimate_compile_time(call, warm)
        key = "x".join(str(d) for d in shape)
        times[key] = measured["compile_time"]
        details[key] = measured["details"]

    return {**times, "details": details}

def recompile_numba_stencil_parallel(spec):
    dtype = get_dtype(spec)
    warm = int(spec.get("warm", 1))

    threads = spec.get("threads")
    if threads is not None:
        set_numba_threads(threads)

    times = {}
    details = {}

    for shape in spec.get("shapes", []):
        shape = tuple(shape)
        x = np.ones(shape, dtype=dtype)

        def call(x=x):
            start = time.perf_counter()
            numba_stencil_5pt_parallel(x)
            return time.perf_counter() - start

        measured = estimate_compile_time(call, warm)
        key = "x".join(str(d) for d in shape)
        times[key] = measured["compile_time"]
        details[key] = measured["details"]

    return {**times, "details": details}