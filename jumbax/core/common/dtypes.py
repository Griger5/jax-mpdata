from enum import StrEnum
from typing import Any

class Dtype(StrEnum):
    boolean = "bool"

    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"

    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    float32 = "float32"
    float64 = "float64"

    complex64 = "complex64"
    complex128 = "complex128"

    def resolve(self, xp: Any):
        try:
            return getattr(xp, self.value)
        except AttributeError as e:
            xp_name = getattr(xp, "__name__", repr(xp))
            raise TypeError(f"Array namespace {xp_name} does not provide dtype {self.value!r}. \nEnsure `xp` is an Array API-compatible namespace.") from e

    @classmethod
    def coerce(cls, dtype: "Dtype | str") -> "Dtype":
        if isinstance(dtype, cls):
            return dtype

        try:
            return cls(dtype)
        except ValueError:
            valid = ", ".join(sorted(member.value for member in cls))
            raise ValueError(f"Unknown dtype {dtype!r}. Expected one of: {valid}") from None

    @classmethod
    def resolve_dtype(cls, dtype: "Dtype | str", xp: Any):
        return cls.coerce(dtype).resolve(xp)