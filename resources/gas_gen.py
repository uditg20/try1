from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GasGenFleet:
    name: str
    n_units: int
    unit_p_max_mw: float
    unit_p_min_mw: float
    ramp_mw_per_min: float
    start_time_minutes: int
    heat_rate_mmbtu_per_mwh: float
    fuel_cost_per_mmbtu: float
    vom_cost_per_mwh: float
    start_cost: float
    pf_min: float  # minimum lagging/leading PF magnitude at generator terminals (screening)

    def validate(self) -> None:
        if self.n_units <= 0:
            raise ValueError("GasGenFleet n_units must be >= 1")
        if self.unit_p_max_mw <= 0:
            raise ValueError("GasGenFleet unit_p_max_mw must be > 0")
        if self.unit_p_min_mw < 0 or self.unit_p_min_mw > self.unit_p_max_mw:
            raise ValueError("GasGenFleet unit_p_min_mw must lie within [0, unit_p_max_mw]")
        if self.start_time_minutes < 0:
            raise ValueError("GasGenFleet start_time_minutes must be >= 0")
        if self.heat_rate_mmbtu_per_mwh <= 0:
            raise ValueError("GasGenFleet heat_rate_mmbtu_per_mwh must be > 0")
        if self.pf_min <= 0 or self.pf_min > 1.0:
            raise ValueError("GasGenFleet pf_min must be in (0, 1]")

    @property
    def fleet_p_max_mw(self) -> float:
        return self.n_units * self.unit_p_max_mw

    @property
    def fleet_p_min_mw(self) -> float:
        return self.n_units * self.unit_p_min_mw

    def marginal_fuel_cost_per_mwh(self) -> float:
        return self.heat_rate_mmbtu_per_mwh * self.fuel_cost_per_mmbtu

