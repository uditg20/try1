"""
ERCOT ISO Adapter
=================

Electric Reliability Council of Texas market rules.

Key characteristics:
- 5-minute real-time dispatch
- 15-minute settlement
- Unique AS products: RRS, ECRS, Reg-Up, Reg-Down, Non-Spin
- Energy-only market (no capacity payments)
"""

from typing import Dict, List, Optional

from .base import (
    ISOAdapter,
    ASProduct,
    ASProductSpec,
    MarketInterval,
)


class ERCOTAdapter(ISOAdapter):
    """
    ERCOT market adapter implementing Texas-specific rules.
    
    ERCOT operates an energy-only market with several AS products.
    BESS participation requires specific energy-backing durations.
    """
    
    @property
    def name(self) -> str:
        return "ERCOT"
    
    @property
    def market_interval(self) -> MarketInterval:
        return MarketInterval(
            rtm_interval_min=5,       # 5-minute real-time dispatch
            dam_interval_min=60,      # Hourly day-ahead market
            settlement_interval_min=15,  # 15-minute settlement
            gate_closure_min=60       # 60 minutes before interval
        )
    
    @property
    def as_products(self) -> List[ASProductSpec]:
        """
        ERCOT Ancillary Services as of 2024:
        - Reg-Up / Reg-Down: Frequency regulation
        - RRS: Responsive Reserve Service (10-min response)
        - ECRS: ERCOT Contingency Reserve Service
        - Non-Spin: Non-Spinning Reserve (30-min response)
        """
        return [
            ASProductSpec(
                name="Regulation Up",
                product_type=ASProduct.REG_UP,
                dispatch_interval_min=5,
                energy_backing_duration_min=60,  # 1 hour continuous
                response_time_sec=4.0,  # AGC response
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,  # Uses deployed MW
                typical_price_per_mw=15.0  # $/MW-h
            ),
            ASProductSpec(
                name="Regulation Down",
                product_type=ASProduct.REG_DOWN,
                dispatch_interval_min=5,
                energy_backing_duration_min=60,
                response_time_sec=4.0,
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=8.0
            ),
            ASProductSpec(
                name="Responsive Reserve Service",
                product_type=ASProduct.RRS,
                dispatch_interval_min=5,
                energy_backing_duration_min=60,  # Must sustain for 1 hour
                response_time_sec=10.0 * 60,  # 10 minutes
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=12.0
            ),
            ASProductSpec(
                name="ERCOT Contingency Reserve",
                product_type=ASProduct.ECRS,
                dispatch_interval_min=5,
                energy_backing_duration_min=60,
                response_time_sec=10.0 * 60,
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=10.0
            ),
            ASProductSpec(
                name="Non-Spinning Reserve",
                product_type=ASProduct.NON_SPIN,
                dispatch_interval_min=5,
                energy_backing_duration_min=30,  # 30-minute backing
                response_time_sec=30.0 * 60,  # 30 minutes
                min_duration_hours=1.0,
                symmetric=False,
                requires_headroom_mw=0.0,
                typical_price_per_mw=5.0
            ),
        ]
    
    def get_reserve_requirement(
        self,
        product: ASProduct,
        capacity_mw: float,
        soc_mwh: Optional[float] = None,
        duration_hours: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate ERCOT-specific reserve requirements.
        
        ERCOT requires energy backing for the full AS duration.
        BESS must have sufficient SOC to deliver for the required period.
        """
        spec = self._get_product_spec(product)
        if spec is None:
            return {
                "max_provision_mw": 0.0,
                "energy_backing_mwh": 0.0,
                "headroom_mw": 0.0
            }
        
        backing_hours = spec.energy_backing_duration_min / 60.0
        
        # If SOC provided, calculate max provision based on energy
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
        """
        Get 5-minute dispatch intervals for ERCOT.
        
        Args:
            horizon_hours: Planning horizon in hours
            
        Returns:
            List of interval indices (12 per hour)
        """
        intervals_per_hour = 60 // self.market_interval.rtm_interval_min
        total_intervals = horizon_hours * intervals_per_hour
        return list(range(total_intervals))
    
    def get_interval_duration_hours(self) -> float:
        """5-minute intervals = 1/12 hour"""
        return self.market_interval.rtm_interval_min / 60.0
    
    def get_energy_price_node(self, node_name: str = "HB_HOUSTON") -> str:
        """
        Get settlement point name for energy pricing.
        
        Common ERCOT hubs:
        - HB_HOUSTON: Houston Hub
        - HB_NORTH: North Hub
        - HB_SOUTH: South Hub
        - HB_WEST: West Hub
        """
        return node_name
    
    def calculate_4cp_exposure(
        self,
        peak_demand_mw: float,
        coincident_fraction: float = 0.8
    ) -> Dict[str, float]:
        """
        Calculate 4CP (Four Coincident Peak) transmission cost exposure.
        
        ERCOT transmission costs are allocated based on load during
        the 4 highest system peaks (one per summer month June-Sept).
        
        Args:
            peak_demand_mw: Facility peak demand
            coincident_fraction: Expected coincidence with system peak
            
        Returns:
            Dict with annual transmission cost estimate
        """
        # Approximate 4CP rate ($/kW-year)
        four_cp_rate = 85.0  # $/kW-year, varies by zone
        
        estimated_4cp_mw = peak_demand_mw * coincident_fraction
        annual_cost = estimated_4cp_mw * 1000 * four_cp_rate / 1000  # Convert to $/year
        
        return {
            "estimated_4cp_mw": estimated_4cp_mw,
            "annual_transmission_cost": annual_cost,
            "rate_per_kw_year": four_cp_rate
        }
