import copy

import libmpdataxx
import numpy as np

def setup(data, metadata: dict):
    pass

def compute(data, metadata: dict):
    global res
    psi = copy.deepcopy(data[0])

    psi_in = np.array(data[0], dtype=np.float32)

    psi_ext = np.empty((metadata["size_x"] + 1, metadata["size_y"] + 1), dtype=np.float32)
    psi_ext[:metadata["size_x"], :metadata["size_y"]] = psi_in
    psi_ext[metadata["size_x"], :metadata["size_y"]] = psi_in[0, :]
    psi_ext[:metadata["size_x"], metadata["size_y"]] = psi_in[:, 0]
    psi_ext[metadata["size_x"], metadata["size_y"]] = psi_in[0, 0]
    psi = psi_ext

    libmpdataxx.mpdata_2d(psi, data[1], data[2], 0.1, metadata["steps"], 1)

    return psi

def result_to_numpy(result, metadata: dict):
    return result[:metadata["size_x"], :metadata["size_y"]]
