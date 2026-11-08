from abc import abstractmethod
from typing import ClassVar

from array_api.latest import Array

from core.mpdata.config import MPDATAConfig

class MPDATABackend():
    supports_mpi: ClassVar[bool]
    supports_n_threads: ClassVar[bool]

    def __init__(self):
        self.options = None

    def require_mpi(self):
        if not self.supports_mpi:
            raise ValueError(f"Backend {self.name!r} does not support MPI")

    def require_n_threads(self):
        if not self.supports_n_threads:
            raise ValueError(f"Backend {self.name!r} does not support setting the thread number")

    def setup(self, options: MPDATAConfig):
        self.options = options

    @abstractmethod
    def solve(self, scalar_field: Array, vector_field: Array, steps: int) -> Array:
        ...

    def teardown(self):
        pass