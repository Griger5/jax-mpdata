from jax_mpdata.procedural import solve

import jax

cpu_device = jax.devices("cpu")[0]

def compute(data, metadata: dict = {}):
    with jax.default_device(cpu_device):
        solve(data[0], data[1], data[2], metadata["steps"], metadata["halo"])