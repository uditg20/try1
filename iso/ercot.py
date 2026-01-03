from .base import IsoAdapter, ReserveProduct
from typing import List, Dict, Any

class ErcotAdapter(IsoAdapter):
    
    @property
    def name(self) -> str:
        return "ERCOT"

    @property
    def dispatch_interval_minutes(self) -> int:
        return 15
    
    @property
    def reserve_products(self) -> List[ReserveProduct]:
        return [
            ReserveProduct(name="RegUp", type="regulation", direction="up", duration_minutes=15),
            ReserveProduct(name="RegDown", type="regulation", direction="down", duration_minutes=15),
            ReserveProduct(name="RRS", type="spinning", direction="up", duration_minutes=60),
            ReserveProduct(name="NonSpin", type="non_spinning", direction="up", duration_minutes=240),
        ]

    def get_market_data(self, start_time: str, end_time: str) -> Dict[str, Any]:
        # This would interface with an API in production. 
        # For now it's just a placeholder.
        return {}
