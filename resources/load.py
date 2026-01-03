from pydantic import BaseModel, Field
from typing import List, Optional

class LoadProfile(BaseModel):
    name: str
    is_critical: bool = Field(True, description="If True, unserved energy has very high penalty")
    # In a real app, this might be a CSV path or a list of values. 
    # For now we'll assume it's passed as a list of float values for the horizon.
    forecast_mw: List[float] = Field(..., description="Load forecast (MW) for the horizon")
    value_of_lost_load: float = Field(10000.0, description="Penalty for unserved energy ($/MWh)")
