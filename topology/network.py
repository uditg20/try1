from pydantic import BaseModel, Field

class Transformer(BaseModel):
    name: str
    capacity_mva: float
    high_side_kv: float
    low_side_kv: float

class POI(BaseModel):
    name: str
    import_limit_mw: float
    export_limit_mw: float
    import_limit_mva: float
    export_limit_mva: float
    min_power_factor: float = Field(0.95, description="Lagging/Leading PF requirement")

class Topology(BaseModel):
    poi: POI
    transformers: list[Transformer]
    
    # We can add validation logic here later if needed
