from contextlib import contextmanager
from array_api.latest import Array

from api.initialization import MPDATAConfig, MPDATAContext
from core.mpdata.available_backends import _MPDATA_CURRENT_BACKEND

@contextmanager
def setup_mpdata(backend: str, *, config: MPDATAConfig):
    with MPDATAContext(backend, config=config):
        yield

def solve_mpdata(scalar_field: Array, vector_field: Array, steps: int) -> Array:
    return _MPDATA_CURRENT_BACKEND.get().solve(scalar_field, vector_field, steps)