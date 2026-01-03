from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BESS:
    name: str
    p_charge_max_mw: float
    p_discharge_max_mw: float
    e_max_mwh: float
    pcs_mva: float
    eta_charge: float
    eta_discharge: float
    ramp_mw_per_min: float
    soc_init_mwh: float
    soc_min_mwh: float
    soc_max_mwh: float
    soc_market_reserve_mwh: float
    degradation_cost_per_mwh_throughput: float

    def validate(self) -> None:
        if self.e_max_mwh <= 0:
            raise ValueError("BESS e_max_mwh must be > 0")
        if not (0 < self.eta_charge <= 1.0) or not (0 < self.eta_discharge <= 1.0):
            raise ValueError("BESS efficiencies must be in (0, 1]")
        if self.pcs_mva <= 0:
            raise ValueError("BESS pcs_mva must be > 0")
        if self.soc_min_mwh < 0 or self.soc_max_mwh > self.e_max_mwh:
            raise ValueError("BESS SOC bounds must lie within [0, e_max_mwh]")
        if not (self.soc_min_mwh <= self.soc_init_mwh <= self.soc_max_mwh):
            raise ValueError("BESS soc_init_mwh must lie within SOC bounds")
        if self.soc_market_reserve_mwh < 0:
            raise ValueError("BESS soc_market_reserve_mwh must be >= 0")

