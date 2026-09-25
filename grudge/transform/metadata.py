from __future__ import annotations

from dataclasses import dataclass

from pytools.tag import Tag, UniqueTag


@dataclass(frozen=True)
class OutputIsTensorProductDOFArrayOrdered(Tag): ...


@dataclass(frozen=True)
class TensorProductDOFAxisTag(UniqueTag):
    axis: int
