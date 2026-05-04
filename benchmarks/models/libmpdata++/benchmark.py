import copy

import libmpdataxx
import numpy as np

res = None

def setup(data, metadata: dict):
    pass

def compute(data, metadata: dict):
    global res
    psi = copy.deepcopy(data[0])
    halo = metadata["halo"]

    psi_in = np.array(data[0], dtype=np.float32)

    psi_ext = np.empty((metadata["size_x"] + 1, metadata["size_y"] + 1), dtype=np.float32)
    psi_ext[:metadata["size_x"], :metadata["size_y"]] = psi_in
    psi_ext[metadata["size_x"], :metadata["size_y"]] = psi_in[0, :]
    psi_ext[:metadata["size_x"], metadata["size_y"]] = psi_in[:, 0]
    psi_ext[metadata["size_x"], metadata["size_y"]] = psi_in[0, 0]
    psi = psi_ext

    res = libmpdataxx.mpdata_2d(psi, data[1][halo:-halo, halo:-halo], data[2][halo:-halo, halo:-halo], 0.1, metadata["steps"], 1)

    res = copy.deepcopy(psi)

    return psi

def result_to_numpy(result, metadata: dict):
    global res
    # return res
    return res[:metadata["size_x"], :metadata["size_y"]]
