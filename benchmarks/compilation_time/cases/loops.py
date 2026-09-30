import time

import jax
import jax.numpy as jnp
import numpy as np
import numba

from benchmarks.compilation_time.kernels_jax import jax_loop_fori_kernel, jax_loop_unrolled_kernel
from benchmarks.compilation_time.kernels_numba import numba_loop_kernel, numba_loop_parallel_kernel
from benchmarks.compilation_time.measure import count_tokens, enable_jax_dtype, first_warm, get_dtype, jax_compile_time

def jax_loop_unrolled(spec):
    dtype = get_dtype(spec)
    enable_jax_dtype(dtype)

    iterations = int(spec.get("iterations", 100))
    shape = tuple(spec.get("shape", (1024,)))

    x_spec = jax.ShapeDtypeStruct(shape, dtype)

    compile_time = jax_compile_time(
        lambda x: jax_loop_unrolled_kernel(x, iterations=iterations),
        args=(x_spec,),
    )

    x_small = jnp.ones((4,), dtype=dtype)
    jaxpr = jax.make_jaxpr(
        lambda x: jax_loop_unrolled_kernel(x, iterations=iterations)
    )(x_small)

    counts = count_tokens(
        str(jaxpr),
        ["sin", "cos", "exp", "abs", "while"],
    )

    return {
        "compile_time": compile_time,
        "jaxpr_counts": counts,
    }

def jax_loop_fori(spec):
    dtype = get_dtype(spec)
    enable_jax_dtype(dtype)

    iterations = int(spec.get("iterations", 100))
    shape = tuple(spec.get("shape", (1024,)))

    x_spec = jax.ShapeDtypeStruct(shape, dtype)
    n_spec = jax.ShapeDtypeStruct((), np.int32)

    compile_time = jax_compile_time(
        jax_loop_fori_kernel,
        args=(x_spec, n_spec),
    )

    x_small = jnp.ones((4,), dtype=dtype)
    jaxpr = jax.make_jaxpr(jax_loop_fori_kernel)(x_small, iterations)

    counts = count_tokens(
        str(jaxpr),
        ["sin", "cos", "exp", "abs", "while"],
    )

    return {
        "compile_time": compile_time,
        "jaxpr_counts": counts,
    }

def numba_loop(spec):
    dtype = get_dtype(spec)
    iterations = int(spec.get("iterations", 100))
    call_shape = tuple(spec.get("call_shape", (8,)))
    warm = int(spec.get("warm", 3))

    x = np.ones(call_shape, dtype=dtype)

    def call():
        start = time.perf_counter()
        numba_loop_kernel(x, iterations)
        return time.perf_counter() - start

    return first_warm(call, warm)

def numba_loop_parallel(spec):
    dtype = get_dtype(spec)
    iterations = int(spec.get("iterations", 100))
    call_shape = tuple(spec.get("call_shape", (8,)))
    warm = int(spec.get("warm", 3))

    threads = spec.get("threads")
    if threads is not None:
        numba.set_num_threads(int(threads))

    x = np.ones(call_shape, dtype=dtype)

    def call():
        start = time.perf_counter()
        numba_loop_parallel_kernel(x, iterations)
        return time.perf_counter() - start

    return first_warm(call, warm)