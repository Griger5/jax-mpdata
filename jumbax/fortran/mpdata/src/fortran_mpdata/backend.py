from array_api.latest import Array
import numpy as np

from jumbax_core.mpdata.backend import MPDATABackend
from jumbax_core.mpdata.config import MPDATAConfig

from ncar_mpdata import mpdata_2d

class FortranMPDATABackend(MPDATABackend):
    name = "fortran_mpdata"
    supports_mpi = False
    supports_n_threads = False

    def setup(self, options: MPDATAConfig):
        super().setup(options)

        if self.options.n_dims != 2:
            raise NotImplementedError("Fortran backend currently supports only a 2D MPDATA scheme")

    def solve(self, scalar_field: Array, vector_field: Array, steps: int) -> Array:
        xp = scalar_field.__array_namespace__()

        nx, nz = scalar_field.shape

        psi_np = np.asarray(scalar_field)
        u1_np = np.asarray(vector_field[0])
        u2_np = np.asarray(vector_field[1])

        psi_ext = np.empty((nx + 1, nz + 1), dtype=psi_np.dtype)
        psi_ext[:nx, :nz] = psi_np
        psi_ext[nx, :nz] = psi_ext[0, :nz]
        psi_ext[:nx, nz] = psi_ext[:nx, 0]
        psi_ext[nx, nz] = psi_ext[0, 0]

        h_ext = np.ones((nx + 1, nz + 1), dtype=psi_np.dtype)

        u1_ext = np.empty((nx + 2, nz + 1), dtype=u1_np.dtype)
        u1_ext[:nx + 1, :nz] = u1_np
        u1_ext[nx + 1, :nz] = u1_ext[1, :nz]
        u1_ext[:, nz] = u1_ext[:, 0]

        u2_ext = np.empty((nx + 1, nz + 2), dtype=u2_np.dtype)
        u2_ext[:nx, :nz + 1] = u2_np
        u2_ext[:nx, nz + 1] = u2_ext[:nx, 1]
        u2_ext[nx, :] = u2_ext[0, :]

        psi_f = np.asfortranarray(psi_ext)
        u1_f = np.asfortranarray(u1_ext)
        u2_f = np.asfortranarray(u2_ext)
        h_f = np.asfortranarray(h_ext)

        mpdata_2d(u1_f, u2_f, psi_f, h_f, 0, 0, steps, self.options.n_iters)

        result_np = np.ascontiguousarray(psi_f[:nx, :nz])

        return xp.asarray(result_np)