from array_api.latest import Array
import numpy as np

from jumbax_core.common.env_manager import EnvContextManager
from jumbax_core.mpdata.backend import MPDATABackend
from jumbax_core.mpdata.config import MPDATAConfig

from libmpdataxx import mpdata_2d

class CppMPDATABackend(MPDATABackend):
    name = "cpp_mpdata"
    supports_mpi = False
    supports_n_threads = True

    def setup(self, options: MPDATAConfig):
        super().setup(options)

        if self.options.n_dims != 2:
            raise NotImplementedError("C++ backend currently supports only a 2D MPDATA scheme")

    def solve(self, scalar_field: Array, vector_field: Array, steps: int) -> Array:
        xp = scalar_field.__array_namespace__()

        psi_np = np.array(scalar_field, dtype=np.float32, copy=True, order='C')
        u1_np = np.array(vector_field[0], dtype=np.float32, copy=True, order='C')
        u2_np = np.array(vector_field[1], dtype=np.float32, copy=True, order='C')

        if self.options.n_threads:
            with EnvContextManager("OMP_NUM_THREADS", str(self.options.n_threads)):
                result_np = mpdata_2d(psi_np, u1_np, u2_np, 0.1, steps, self.options.n_iters)
        else:
            result_np = mpdata_2d(psi_np, u1_np, u2_np, 0.1, steps, self.options.n_iters)

        return xp.asarray(result_np)