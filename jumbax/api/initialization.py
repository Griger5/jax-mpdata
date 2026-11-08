from core.mpdata.config import MPDATAConfig
from core.mpdata.available_backends import _MPDATA_AVAILABLE_BACKENDS, _MPDATA_CURRENT_BACKEND

class MPDATAContext:
    def __init__(self, backend: str, *, config: MPDATAConfig):
        if not isinstance(backend, str):
            raise TypeError("Backend argument should be a string")

        if not isinstance(config, MPDATAConfig):
            raise TypeError("Config argument should be an MPDATAConfig")

        if backend not in _MPDATA_AVAILABLE_BACKENDS.keys():
            raise ValueError(f"Unknown backend {backend!r}. Expected one of: {_MPDATA_AVAILABLE_BACKENDS.keys()}")
        
        self.backend_name = backend
        self.config = config
        self.backend_token = None

    def __enter__(self):
        self.backend_token = _MPDATA_CURRENT_BACKEND.set(_MPDATA_AVAILABLE_BACKENDS[self.backend_name])

        backend = _MPDATA_CURRENT_BACKEND.get()

        if self.config.mpi:
            backend.require_mpi()

        if self.config.n_threads:
            backend.require_n_threads()

        backend.setup(self.config)

    def __exit__(self, exc_type, exc_value, traceback):
        _MPDATA_CURRENT_BACKEND.reset(self.backend_token)

        return False