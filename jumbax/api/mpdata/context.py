from contextlib import contextmanager

from jumbax_core.mpdata.available_backends import MPDATA_BACKENDS, _MPDATA_CURRENT_BACKEND
from jumbax_core.mpdata import MPDATAConfig

@contextmanager
def MPDATAContext(backend: str, *, config: MPDATAConfig):
    available = MPDATA_BACKENDS.available()

    if backend not in available:
        raise ValueError(f"Unknown backend {backend!r}; choose from {list(available)}")

    instance = available[backend]()
    token = _MPDATA_CURRENT_BACKEND.set(instance)
    
    try:
        if config.mpi:
            instance.require_mpi()
        if config.n_threads is not None:
            instance.require_n_threads()

        instance.setup(config)
        yield instance
    finally:
        try:
            instance.teardown()
        finally:
            _MPDATA_CURRENT_BACKEND.reset(token)