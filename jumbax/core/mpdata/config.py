from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.common import dtypes

class BoundaryCondition(Enum):
    PERIODIC = 1

@dataclass
class MPDATAConfig:
    n_dims: int
    n_iters: int
    epsilon: float
    dtype: dtypes.Dtype
    halo: int
    boundary_conditions: list[BoundaryCondition]

    mpi: bool
    n_threads: Optional[int]