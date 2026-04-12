import ncar_mpdata

import numpy as np

psi = None
h = None
u1 = None
u2 = None

def setup(data, metadata: dict = {}):
    global psi, h, u1, u2

    psi_in = np.array(data[0], dtype=np.float32)

    h = np.asfortranarray(np.ones((metadata["size_x"] + 1, metadata["size_y"] + 1), dtype=np.float32))

    psi_ext = np.empty((metadata["size_x"] + 1, metadata["size_y"] + 1), dtype=np.float32)
    psi_ext[:metadata["size_x"], :metadata["size_y"]] = psi_in
    psi_ext[metadata["size_x"], :metadata["size_y"]] = psi_in[0, :]
    psi_ext[:metadata["size_x"], metadata["size_y"]] = psi_in[:, 0]
    psi_ext[metadata["size_x"], metadata["size_y"]] = psi_in[0, 0]
    psi = np.asfortranarray(psi_ext)

    u1 = np.asfortranarray(
        np.full((metadata["size_x"] + 2, metadata["size_y"] + 1), data[1][metadata["halo"], metadata["halo"]], dtype=np.float32)
    )
    u2 = np.asfortranarray(
        np.full((metadata["size_x"] + 1, metadata["size_y"] + 2), data[2][metadata["halo"], metadata["halo"]], dtype=np.float32)
    )

def compute(data, metadata: dict = {}):
    global psi, h, u1, u2

    for _ in range(metadata["steps"]):
        ncar_mpdata.mpdata_2d(u1, u2, psi, h, 0, 0)

    return psi.copy()

def result_to_numpy(result, metadata):
    return np.ascontiguousarray(result[:metadata["size_x"], :metadata["size_y"]])
