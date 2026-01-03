from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class POI:
    """
    Point of interconnection limits (at grid interface).
    """

    p_import_max_mw: float
    p_export_max_mw: float
    s_max_mva: float
    pf_min: float  # minimum PF magnitude at POI (screening)


@dataclass(frozen=True)
class Transformer:
    """
    Simplified transformer bank definition.

    count: number of identical transformers in parallel.
    mva_rating: nameplate per transformer.
    n_plus_one: if True, enforce (count-1) available for screening.
    """

    name: str
    count: int
    mva_rating: float
    n_plus_one: bool

    def effective_mva_capacity(self) -> float:
        if self.count <= 0:
            return 0.0
        if self.n_plus_one and self.count >= 2:
            return (self.count - 1) * self.mva_rating
        return self.count * self.mva_rating


@dataclass(frozen=True)
class Topology:
    """
    Minimal explicit topology:

    Grid (POI bus) -> Transformer(s) -> Microgrid bus (resources + load)

    This is extendable to multi-bus later; we keep it explicit and auditable now.
    """

    poi: POI
    transformers: List[Transformer]

    def validate(self) -> None:
        if self.poi.s_max_mva <= 0:
            raise ValueError("POI s_max_mva must be > 0")
        if self.poi.p_import_max_mw < 0 or self.poi.p_export_max_mw < 0:
            raise ValueError("POI import/export MW caps must be >= 0")
        if self.poi.pf_min <= 0 or self.poi.pf_min > 1.0:
            raise ValueError("POI pf_min must be in (0, 1]")
        if len(self.transformers) == 0:
            raise ValueError("Topology must include at least one transformer bank")
        for t in self.transformers:
            if t.count <= 0:
                raise ValueError(f"Transformer {t.name} count must be >= 1")
            if t.mva_rating <= 0:
                raise ValueError(f"Transformer {t.name} mva_rating must be > 0")

    def transformer_total_effective_mva(self) -> float:
        return sum(t.effective_mva_capacity() for t in self.transformers)

