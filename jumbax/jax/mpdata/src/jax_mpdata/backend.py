from collections.abc import Sequence

import jax
from array_api.latest import Array

from jumbax_core.common.dtypes import Dtype
from jumbax_core.mpdata import MPDATABackend, MPDATAConfig
from jumbax_core.mpdata.available_backends import add_mpdata_backend

from .implementation import solve

@add_mpdata_backend("jax")
class JaxMPDATABackend(MPDATABackend):
    name = "jax_mpdata"
    supports_mpi = True
    supports_n_threads = False
    supported_dtypes = (Dtype.float32, Dtype.float64)

    def __init__(self):
        super().__init__()
        self._comm = None
        self._solver = solve

    def setup(self, options: MPDATAConfig):
        super().setup(options)

        if Dtype.coerce(options.dtype) == Dtype.float64 and not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 before requesting float64")

        if options.mpi:
            try:
                from .implementation_mpi import make_mpi_solver
            except ModuleNotFoundError as exc:
                if exc.name not in {"mpi4py", "mpi4jax"}:
                    raise

                raise ImportError("MPI mode requires mpi4py and mpi4jax. Install the JAX backend with its [mpi] extra.") from exc

            self._comm, self._solver = make_mpi_solver(options.n_dims)

    def solve(self, scalar_field: Array, vector_field: Sequence[Array], steps: int) -> Array:
        return self._solver(scalar_field, vector_field, steps, self.options.halo, self.options.n_iters)

    def teardown(self):
        if self._comm is not None:
            jax.effects_barrier()
            self._comm.Free()
            self._comm = None

        self._solver = solve