from importlib.metadata import entry_points
from typing import Callable, Generic, TypeVar

B = TypeVar("B")

class BackendRegistry(Generic[B]):
    def __init__(self, algorithm: str, base: type[B]):
        self.group = f"jumbax.backends.{algorithm}"
        self.base = base
        self._backends: dict[str, type[B]] = {}
        self._loaded = False

    def register(self, name: str) -> Callable[[type[B]], type[B]]:
        def decorator(cls: type[B]) -> type[B]:
            if not issubclass(cls, self.base):
                raise TypeError(f"{cls.__name__} must inherit {self.base.__name__}")

            if name in self._backends and self._backends[name] is not cls:
                raise ValueError(f"Duplicate backend {name!r} in {self.group}")

            self._backends[name] = cls
            return cls

        return decorator

    def available(self) -> dict[str, type[B]]:
        if not self._loaded:
            for entry in entry_points(group=self.group):
                entry.load()
            self._loaded = True

        return self._backends.copy()