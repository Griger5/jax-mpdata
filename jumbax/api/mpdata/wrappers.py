from contextlib import contextmanager
from typing import Optional

from array_api.latest import Array

from jumbax_api.mpdata.context import MPDATAConfig, MPDATAContext
from jumbax_core.mpdata.available_backends import _MPDATA_CURRENT_BACKEND
from jumbax_core.mpdata.config import BoundaryCondition
from jumbax_core.common.dtypes import Dtype

@contextmanager
def setup_mpdata(backend: str, *, n_dims: int, n_iters: int, epsilon: float, dtype: Dtype, halo: int, boundary_conditions: list[BoundaryCondition], mpi: bool = False, n_threads: Optional[int] =  None):
    with MPDATAContext(backend, config=MPDATAConfig(n_dims, n_iters, epsilon, dtype, halo, boundary_conditions, mpi, n_threads)):
        yield

def solve_mpdata(scalar_field: Array, vector_field: Array, steps: int) -> Array:
    return _MPDATA_CURRENT_BACKEND.get().solve(scalar_field, vector_field, steps)