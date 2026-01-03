from pydantic import BaseModel, Field
from typing import Optional

class GasGen(BaseModel):
    name: str
    num_units: int = Field(..., description="Number of identical units")
    unit_capacity_mw: float = Field(..., description="Capacity per unit (MW)")
    min_stable_level_mw: float = Field(..., description="Minimum stable generation per unit (MW)")
    ramp_rate_mw_per_min: float = Field(..., description="Ramp rate (MW/min)")
    start_time_minutes: int = Field(..., description="Start-up time (minutes)")
    heat_rate_mmbtu_per_mwh: float = Field(..., description="Heat rate (MMBtu/MWh)")
    fuel_cost_per_mmbtu: float = Field(..., description="Fuel cost ($/MMBtu)")
    start_cost_dollars: float = Field(0.0, description="Start-up cost ($)")
    om_cost_per_mwh: float = Field(0.0, description="Variable O&M cost ($/MWh)")

    @property
    def total_capacity_mw(self):
        return self.num_units * self.unit_capacity_mw
