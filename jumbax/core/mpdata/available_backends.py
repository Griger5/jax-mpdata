from contextvars import ContextVar

from jumbax_core.common import BackendRegistry
from jumbax_core.mpdata.backend import MPDATABackend

MPDATA_BACKENDS = BackendRegistry("mpdata", MPDATABackend)
add_mpdata_backend = MPDATA_BACKENDS.register

_MPDATA_CURRENT_BACKEND: ContextVar[MPDATABackend | None] = ContextVar("mpdata_current_backend", default=None)