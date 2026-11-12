from contextvars import ContextVar
from typing import Optional

from core.mpdata.backend import MPDATABackend

_MPDATA_AVAILABLE_BACKENDS: dict[str, MPDATABackend] = {}
_MPDATA_CURRENT_BACKEND: ContextVar[Optional[MPDATABackend]] = ContextVar("mpdata_current_backend", default=None)

try:
    from jax_mpdata import JaxMPDATABackend

    _MPDATA_AVAILABLE_BACKENDS["jax"] = JaxMPDATABackend()
except:
    pass

try:
    from numba_mpdata import NumbaMPDATABackend

    _MPDATA_AVAILABLE_BACKENDS["numba"] = NumbaMPDATABackend()
except:
    pass

try:
    from fortran_mpdata import FortranMPDATABackend

    _MPDATA_AVAILABLE_BACKENDS["fortran"] = FortranMPDATABackend()
except:
    pass

# use a decorator wrapper