import numpy as np
from array_api.latest import Array

from jumbax_core.mpdata.backend import MPDATABackend
from jumbax_core.mpdata.config import MPDATAConfig, BoundaryCondition
from jumbax_core.common.dtypes import Dtype

from PyMPDATA import Options, ScalarField, VectorField, Stepper, Solver
from PyMPDATA.boundary_conditions import Periodic

class NumbaMPDATABackend(MPDATABackend):
    name = "numba_mpdata"
    supports_mpi = True
    supports_n_threads = False

    def __init__(self):
        self.stepper = None
        self.advectee = None
        self.advector = None
        self.solver = None

    def setup(self, options: MPDATAConfig):
        super().setup(options)

        if self.options.mpi:
            raise NotImplementedError()

        pympdata_options = Options(n_iters=self.options.n_iters, dtype=Dtype.resolve_dtype(self.options.dtype, np), epsilon=self.options.epsilon)

        self.stepper = Stepper(options=pympdata_options, n_dims=self.options.n_dims, n_threads=self.options.n_threads)

    def solve(self, scalar_field: Array, vector_field: Array, steps: int) -> Array:
        if not self.solver:
            self.advectee = ScalarField(data=scalar_field, halo=self.options.halo, boundary_conditions=self.resolve_boundary_conditions())
            self.advector = VectorField(data=vector_field, halo=self.options.halo, boundary_conditions=self.resolve_boundary_conditions())

            self.solver = Solver(stepper=self.stepper, advectee=self.advectee, advector=self.advector)

        self.solver.advance(n_steps=steps)

        return self.solver.advectee.get()

    def teardown(self):
        self.stepper = None
        self.advectee = None
        self.advector = None
        self.solver = None

    def resolve_boundary_conditions(self):
        pympdata_conditions = []
        
        for condition in self.options.boundary_conditions:
            match condition:
                case BoundaryCondition.PERIODIC:
                    pympdata_conditions.append(Periodic())

        return tuple(pympdata_conditions)