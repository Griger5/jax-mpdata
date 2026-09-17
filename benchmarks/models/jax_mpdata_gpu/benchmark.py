import jax
import numpy as np

from benchmarks.models.jax_mpdata_cpu_serial.benchmark import _setup, _compute, result_to_numpy

gpu_device = jax.devices("gpu")[0]

psi = None

def setup(data, metadata: dict):
    _setup(data, metadata, gpu_device)

def compute(data, metadata: dict):
    return _compute(data, metadata, gpu_device)