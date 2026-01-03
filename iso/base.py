from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


ReserveDirection = Literal["up", "down"]


@dataclass(frozen=True)
class ReserveProduct:
    """
    ISO-agnostic reserve product definition.

    - duration_minutes: energy-backing duration requirement
    - direction: up reserve consumes discharge headroom/energy, down consumes charge headroom/energy
    """

    name: str
    direction: ReserveDirection
    duration_minutes: int


class ISOAdapter:
    """
    Base class for ISO adapters.

    Adapters define:
    - dispatch interval
    - reserve products
    - headroom rules and settlement assumptions
    """

    iso_name: str = "BASE"

    def dispatch_interval_minutes(self) -> int:
        raise NotImplementedError

    def reserve_products(self) -> List[ReserveProduct]:
        return []

    def energy_price_series_key(self) -> str:
        """
        Key in case JSON for energy price series ($/MWh).
        """

        return "energy_price_$per_mwh"

    def reserve_price_series_keys(self) -> Dict[str, str]:
        """
        Map reserve product -> case JSON price series key ($/MW-interval-hour).
        For simplicity, we settle reserves as $/MW-h with dt scaling in runner.
        """

        return {}

    def default_big_m(self) -> float:
        """
        Big-M used for certain logical constraints.
        """

        return 1e4

    def validate_case(self, case: dict) -> None:
        """
        Adapter-level validation hook. Keep it explainable and explicit.
        """

        # Minimal checks; deeper validation can be layered via pydantic in runner.
        if "time" not in case:
            raise ValueError("case missing required key: time")

