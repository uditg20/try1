"""
Generic ISO Adapter
===================

Default adapter for non-specific or simplified market modeling.
Uses common market structures applicable across most ISOs.
"""

from typing import Dict, List, Optional

from .base import (
    ISOAdapter,
    ASProduct,
    ASProductSpec,
    MarketInterval,
)


class GenericAdapter(ISOAdapter):
    """
    Generic market adapter for ISO-agnostic modeling.
    
    Provides reasonable defaults that work for most markets:
    - 15-minute dispatch intervals
    - Standard regulation and reserve products
    - Configurable parameters
    """
    
    def __init__(
        self,
        dispatch_interval_min: int = 15,
        energy_backing_min: int = 60
    ):
        """
        Initialize generic adapter with configurable parameters.
        
        Args:
            dispatch_interval_min: Dispatch interval duration (default 15 min)
            energy_backing_min: Required energy backing for reserves (default 60 min)
        """
        self._dispatch_interval = dispatch_interval_min
        self._energy_backing = energy_backing_min
    
    @property
    def name(self) -> str:
        return "GENERIC"
    
    @property
    def market_interval(self) -> MarketInterval:
        return MarketInterval(
            rtm_interval_min=self._dispatch_interval,
            dam_interval_min=60,
            settlement_interval_min=self._dispatch_interval,
            gate_closure_min=30
        )
    
    @property
    def as_products(self) -> List[ASProductSpec]:
        """
        Generic AS products representing common market structures.
        """
        return [
            ASProductSpec(
                name="Regulation Up",
                product_type=ASProduct.REG_UP,
                dispatch_interval_min=self._dispatch_interval,
                energy_backing_duration_min=self._energy_backing,
                response_time_sec=4.0,
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=12.0
            ),
            ASProductSpec(
                name="Regulation Down",
                product_type=ASProduct.REG_DOWN,
                dispatch_interval_min=self._dispatch_interval,
                energy_backing_duration_min=self._energy_backing,
                response_time_sec=4.0,
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=6.0
            ),
            ASProductSpec(
                name="Spinning Reserve",
                product_type=ASProduct.SPIN,
                dispatch_interval_min=self._dispatch_interval,
                energy_backing_duration_min=self._energy_backing,
                response_time_sec=600.0,  # 10 minutes
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=8.0
            ),
            ASProductSpec(
                name="Non-Spinning Reserve",
                product_type=ASProduct.NON_SPIN,
                dispatch_interval_min=self._dispatch_interval,
                energy_backing_duration_min=30,
                response_time_sec=1800.0,  # 30 minutes
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=4.0
            ),
        ]
    
    def get_reserve_requirement(
        self,
        product: ASProduct,
        capacity_mw: float,
        soc_mwh: Optional[float] = None,
        duration_hours: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate reserve requirements with energy backing."""
        spec = self._get_product_spec(product)
        if spec is None:
            return {
                "max_provision_mw": 0.0,
                "energy_backing_mwh": 0.0,
                "headroom_mw": 0.0
            }
        
        backing_hours = spec.energy_backing_duration_min / 60.0
        
        if soc_mwh is not None:
            max_from_energy = soc_mwh / backing_hours
            max_provision = min(capacity_mw, max_from_energy)
        else:
            max_provision = capacity_mw
        
        return {
            "max_provision_mw": max_provision,
            "energy_backing_mwh": max_provision * backing_hours,
            "headroom_mw": spec.requires_headroom_mw
        }
    
    def get_dispatch_intervals(self, horizon_hours: int) -> List[int]:
        """Get dispatch intervals for planning horizon."""
        intervals_per_hour = 60 // self.market_interval.rtm_interval_min
        total_intervals = horizon_hours * intervals_per_hour
        return list(range(total_intervals))
    
    def get_interval_duration_hours(self) -> float:
        """Duration of each dispatch interval in hours."""
        return self.market_interval.rtm_interval_min / 60.0


# Factory function for creating ISO adapters
def get_iso_adapter(iso_name: str) -> ISOAdapter:
    """
    Factory function to get the appropriate ISO adapter.
    
    Args:
        iso_name: ISO identifier ('ERCOT', 'CAISO', 'PJM', 'GENERIC', etc.)
        
    Returns:
        Appropriate ISOAdapter instance
    """
    from .ercot import ERCOTAdapter
    
    adapters = {
        "ERCOT": ERCOTAdapter,
        "GENERIC": GenericAdapter,
        # Future: Add more ISOs
        # "CAISO": CAISOAdapter,
        # "PJM": PJMAdapter,
        # "MISO": MISOAdapter,
        # "NYISO": NYISOAdapter,
        # "ISONE": ISONEAdapter,
        # "SPP": SPPAdapter,
    }
    
    adapter_class = adapters.get(iso_name.upper(), GenericAdapter)
    return adapter_class()
