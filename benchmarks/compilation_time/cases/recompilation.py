import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.compilation_time.kernels_jax import jax_stencil_5pt
from benchmarks.compilation_time.kernels_numba import numba_stencil_5pt, numba_stencil_5pt_parallel
from benchmarks.compilation_time.measure import enable_jax_dtype, first_warm, get_dtype

def recompile_jax_stencil(spec):
    dtype = get_dtype(spec)
    enable_jax_dtype(dtype)

    f = jax.jit(jax_stencil_5pt)
    warm = int(spec.get("warm", 1))
    sequence = []

    for shape in spec.get("shapes", []):
        shape = tuple(shape)
        x = jnp.ones(shape, dtype=dtype)

        def call(x=x):
            start = time.perf_counter()
            y = f(x)
            y.block_until_ready()
            return time.perf_counter() - start

        entry = {"shape": list(shape)}
        entry.update(first_warm(call, warm))
        sequence.append(entry)

    return {"sequence": sequence}

def recompile_numba_stencil(spec):
    dtype = get_dtype(spec)
    warm = int(spec.get("warm", 1))
    sequence = []

    for shape in spec.get("shapes", []):
        shape = tuple(shape)
        x = np.ones(shape, dtype=dtype)

        def call(x=x):
            start = time.perf_counter()
            numba_stencil_5pt(x)
            return time.perf_counter() - start

        entry = {"shape": list(shape)}
        entry.update(first_warm(call, warm))
        sequence.append(entry)

    return {"sequence": sequence}

def recompile_numba_stencil_parallel(spec):
    import numba

    dtype = get_dtype(spec)
    warm = int(spec.get("warm", 1))

    threads = spec.get("threads")
    if threads is not None:
        numba.set_num_threads(int(threads))

    sequence = []

    for shape in spec.get("shapes", []):
        shape = tuple(shape)
        x = np.ones(shape, dtype=dtype)

        def call(x=x):
            start = time.perf_counter()
            numba_stencil_5pt_parallel(x)
            return time.perf_counter() - start

        entry = {"shape": list(shape)}
        entry.update(first_warm(call, warm))
        sequence.append(entry)

    return {"sequence": sequence}