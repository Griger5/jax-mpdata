from array_api.latest import Array

from jumbax_core.mpdata.backend import MPDATABackend
from jumbax_core.mpdata.config import MPDATAConfig

from .impl import solve, solve_mpi, setup_comm

class JaxMPDATABackend(MPDATABackend):
    name = "jax_mpdata"
    supports_mpi = True
    supports_n_threads = False

    def setup(self, options: MPDATAConfig):
        super().setup(options)

        if self.options.mpi:
            setup_comm(self.options.n_dims)

    def solve(self, scalar_field: Array, vector_field: Array, steps: int) -> Array:
        if self.options.mpi:
            return solve_mpi(scalar_field, vector_field, steps, self.options.halo, self.options.n_iters)

        return solve(scalar_field, vector_field, steps, self.options.halo, self.options.n_iters)