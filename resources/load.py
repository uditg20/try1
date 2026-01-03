from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Load:
    """
    Load definition with critical/non-critical split and optional curtailment.
    Reactive power is modeled via a fixed power factor.
    """

    name: str
    p_critical_mw: List[float]
    p_noncritical_mw: List[float]
    p_curtailable_max_mw: List[float]
    pf_load: float  # magnitude, e.g. 0.98

    def validate(self) -> None:
        n = len(self.p_critical_mw)
        if n == 0:
            raise ValueError("Load time series must be non-empty")
        if not (len(self.p_noncritical_mw) == len(self.p_curtailable_max_mw) == n):
            raise ValueError("Load series lengths must match")
        if self.pf_load <= 0 or self.pf_load > 1.0:
            raise ValueError("Load pf_load must be in (0, 1]")

