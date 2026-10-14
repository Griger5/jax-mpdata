import os
import time

import numpy as np

from PyMPDATA import Options, ScalarField, VectorField, Stepper, Solver
from PyMPDATA.boundary_conditions import Periodic

def _run_pympdata(case: dict, config: dict) -> list:
    metadata = case["metadata"]

    nt = metadata["steps"]
    halo = metadata["halo"]
    n_iters = metadata["n_iters"]

    psi_batch_np = case["psi_batch"]
    Cx_int, Cy_int = case["C_int"]

    batch_size = psi_batch_np.shape[0]

    repeats = int(config["repeats"])
    warmup = int(config["warmup_pympdata"])

    n_threads = os.cpu_count()

    def make_solvers():
        solvers = []

        for i in range(batch_size):
            options = Options(
                n_iters=n_iters,
                dtype=np.float32,
                epsilon=1e-12,
            )

            stepper = Stepper(
                options=options,
                n_dims=2,
                n_threads=n_threads,
            )

            advectee = ScalarField(
                data=psi_batch_np[i].copy(),
                halo=halo,
                boundary_conditions=(Periodic(), Periodic()),
            )

            advector = VectorField(
                data=(Cx_int, Cy_int),
                halo=halo,
                boundary_conditions=(Periodic(), Periodic()),
            )

            solver = Solver(
                stepper=stepper,
                advectee=advectee,
                advector=advector,
            )

            solver.advance(n_steps=0)
            solvers.append(solver)

        return solvers

    for _ in range(warmup):
        solvers = make_solvers()

        for solver in solvers:
            solver.advance(n_steps=nt)

        del solvers

    times = []

    for _ in range(repeats):
        solvers = make_solvers()

        start = time.perf_counter()

        for solver in solvers:
            solver.advance(n_steps=nt)

        times.append(time.perf_counter() - start)

        del solvers

    return times

def get_benchmarks(config: dict) -> list:
    return [
        {
            "name": "pympdata_sequential",
            "run": _run_pympdata,
        }
    ]