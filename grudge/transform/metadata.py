from __future__ import annotations

from dataclasses import dataclass

from pytools.tag import Tag


@dataclass(frozen=True)
class OutputIsTensorProductDOFArrayOrdered(Tag): ...
