from __future__ import annotations
from typing import Protocol, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

__all__ = ["DataFrameNormalizer"]


class DataFrameNormalizer(Protocol):
    def reset(self) -> None:
        ...

    def fit(self, data: T):
        ...

    def transform(self, data: T) -> T:
        ...

    def fit_transform(self, data: T) -> T:
        ...
