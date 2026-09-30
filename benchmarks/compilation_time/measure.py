import re
import statistics
import time

import jax
import numpy as np

DEFAULT_DTYPE = "float32"

DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
}

def get_dtype(spec):
    return DTYPE_MAP[spec.get("dtype", DEFAULT_DTYPE)]

def enable_jax_dtype(dtype):
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)

def median(xs):
    return statistics.median(xs) if xs else None

def count_tokens(text, tokens):
    return {
        token: len(re.findall(rf"\b{token}\b", text))
        for token in tokens
    }

def jax_compile_time(fn, args=(), kwargs=None, jit_kwargs=None):
    kwargs = kwargs or {}
    jit_kwargs = jit_kwargs or {}

    jitted = jax.jit(fn, **jit_kwargs)

    start = time.perf_counter()
    lowered = jitted.lower(*args, **kwargs)
    lowered.compile()
    return time.perf_counter() - start

def first_warm(call, warm):
    first = call()
    warm_times = [call() for _ in range(warm)]
    warm_median = median(warm_times)

    overhead_estimate = (
        max(first - warm_median, 0.0)
        if warm_median is not None
        else None
    )

    return {
        "first": first,
        "warm_times": warm_times,
        "warm_median": warm_median,
        "overhead_estimate": overhead_estimate,
    }