from jax_mpdata.procedural import solve

import jax
import jax.numpy as jnp
import numpy as np

cpu_device = jax.devices("cpu")[0]

psi = None

def _setup(data, metadata, device):
    global psi
    with jax.default_device(device):
        psi = jnp.zeros((metadata["size_x"] + 2*metadata["halo"], metadata["size_y"] + 2*metadata["halo"]))
        psi = psi.at[metadata["halo"]:metadata["halo"]+metadata["size_x"], metadata["halo"]:metadata["halo"]+metadata["size_y"]].set(data[0])

def _compute(data, metadata, device):
    with jax.default_device(device):
        return solve(psi, data[1], data[2], metadata["steps"], metadata["halo"], metadata["n_iters"])

def setup(data, metadata: dict):
    _setup(data, metadata, cpu_device)

def compute(data, metadata: dict):
    return _compute(data, metadata, cpu_device)
    
def result_to_numpy(result, metadata):
    return np.asarray(result[metadata["halo"]:-metadata["halo"], metadata["halo"]:-metadata["halo"]])