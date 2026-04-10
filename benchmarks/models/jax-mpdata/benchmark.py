from jax_mpdata.procedural import solve

import jax
import jax.numpy as jnp
import numpy as np

cpu_device = jax.devices("cpu")[0]

psi = None

def setup(data, metadata: dict = {}):
    global psi
    with jax.default_device(cpu_device):
        psi = jnp.zeros((metadata["size_x"] + 2*metadata["halo"], metadata["size_y"] + 2*metadata["halo"]))
        psi = psi.at[metadata["halo"]:metadata["halo"]+metadata["size_x"], metadata["halo"]:metadata["halo"]+metadata["size_y"]].set(data[0])

def compute(data, metadata: dict = {}):
    global psi
    with jax.default_device(cpu_device):
        return solve(psi, data[1], data[2], metadata["steps"], metadata["halo"])
    
def result_to_numpy(result, metadata):
    return np.asarray(result[metadata["halo"]:-metadata["halo"], metadata["halo"]:-metadata["halo"]])