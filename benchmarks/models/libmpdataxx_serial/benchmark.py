import copy

import libmpdataxx
import numpy as np

from benchmarks.env_manager import EnvContextManager
from benchmarks.models.libmpdataxx_4threads.benchmark import setup, _compute as imported_compute, result_to_numpy

def compute(data, metadata: dict):
    with EnvContextManager("OMP_NUM_THREADS", "1"):
        return imported_compute(data, metadata)

