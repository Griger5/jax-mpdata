from PyMPDATA import Options
from PyMPDATA import ScalarField
from PyMPDATA import VectorField
from PyMPDATA import Stepper
from PyMPDATA import Solver
from PyMPDATA.boundary_conditions import Periodic

import numpy as np

solver = None

def _setup(data, metadata, n_threads):
    options = Options(n_iters=metadata["n_iters"], dtype=np.float32)
    stepper = Stepper(options=options, n_dims=2, n_threads=n_threads)

    advectee = ScalarField(
        data=data[0],
        halo=metadata["halo"],
        boundary_conditions=(Periodic(), Periodic())
    )
    advector = VectorField(
        data=(data[1][metadata["halo"]:-metadata["halo"], metadata["halo"]:-metadata["halo"]], data[2][metadata["halo"]:-metadata["halo"], metadata["halo"]:-metadata["halo"]]),
        halo=metadata["halo"],
        boundary_conditions=(Periodic(), Periodic())
    )

    solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
    solver.advance(n_steps=0)

    return solver

def setup(data, metadata: dict):
    global solver

    solver = _setup(data, metadata, 4)

def compute(data, metadata: dict):
    solver.advance(n_steps=metadata["steps"])

def result_to_numpy(result, metadata: dict):
    return solver.advectee.get()