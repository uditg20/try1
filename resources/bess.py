from pydantic import BaseModel, Field
from typing import Optional

class BESS(BaseModel):
    name: str
    capacity_mw: float = Field(..., description="Maximum discharge power (MW)")
    capacity_mwh: float = Field(..., description="Total energy capacity (MWh)")
    pcs_mva: float = Field(..., description="Power Conversion System rating (MVA)")
    efficiency_round_trip: float = Field(0.9, description="Round trip efficiency")
    soc_min_reserve: float = Field(0.05, description="Minimum SOC reserve (0-1)")
    soc_initial: float = Field(0.5, description="Initial SOC (0-1)")
    ramp_rate_mw_per_min: Optional[float] = Field(None, description="Ramp rate limit (MW/min)")

    @property
    def charge_efficiency(self):
        return self.efficiency_round_trip ** 0.5
    
    @property
    def discharge_efficiency(self):
        return self.efficiency_round_trip ** 0.5
