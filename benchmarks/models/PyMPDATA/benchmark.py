from PyMPDATA import Options
from PyMPDATA import ScalarField
from PyMPDATA import VectorField
from PyMPDATA import Stepper
from PyMPDATA import Solver
from PyMPDATA.boundary_conditions import Periodic

import numpy as np

advectee = None
advector = None
options = Options(n_iters=1, dtype=np.float32)
stepper = Stepper(options=options, n_dims=2, n_threads=1)
solver = None

def setup(data, metadata: dict):
    global advectee, advector, options, stepper, solver

    options = Options(n_iters=metadata["n_iters"], dtype=np.float32)
    stepper = Stepper(options=options, n_dims=2, n_threads=1)

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

def compute(data, metadata: dict):
    solver.advance(n_steps=metadata["steps"])

def result_to_numpy(result, metadata: dict):
    return solver.advectee.get()