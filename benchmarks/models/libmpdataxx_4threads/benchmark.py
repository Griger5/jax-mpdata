import copy

import libmpdataxx
import numpy as np

from benchmarks.env_manager import EnvContextManager

def _compute(data, metadata):
    psi = copy.deepcopy(data[0])

    psi_in = np.array(data[0], dtype=np.float32)

    psi_ext = np.empty((metadata["size_x"] + 1, metadata["size_y"] + 1), dtype=np.float32)
    psi_ext[:metadata["size_x"], :metadata["size_y"]] = psi_in
    psi_ext[metadata["size_x"], :metadata["size_y"]] = psi_in[0, :]
    psi_ext[:metadata["size_x"], metadata["size_y"]] = psi_in[:, 0]
    psi_ext[metadata["size_x"], metadata["size_y"]] = psi_in[0, 0]
    psi = psi_ext

    libmpdataxx.mpdata_2d(psi, data[1], data[2], 0.1, metadata["steps"], metadata["n_iters"])

    return psi

def setup(data, metadata: dict):
    pass  

def compute(data, metadata: dict):
    with EnvContextManager("OMP_NUM_THREADS", "4"):
        return _compute(data, metadata)

def result_to_numpy(result, metadata: dict):
    return result[:metadata["size_x"], :metadata["size_y"]]
