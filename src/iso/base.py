"""
Base ISO Adapter
================

Abstract base class defining the interface for all ISO adapters.
Each ISO has different market rules, dispatch intervals, and AS products.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ASProduct(Enum):
    """Ancillary Service Product Types"""
    # Frequency Regulation
    REG_UP = "regulation_up"
    REG_DOWN = "regulation_down"
    
    # Spinning/Non-Spinning Reserves
    SPIN = "spinning_reserve"
    NON_SPIN = "non_spinning_reserve"
    
    # ERCOT-specific
    ECRS = "ercot_contingency_reserve"  # ERCOT Contingency Reserve Service
    RRS = "responsive_reserve"  # Responsive Reserve Service
    
    # Fast-response products
    FFR = "fast_frequency_response"
    
    # Generic reserves
    PRIMARY = "primary_reserve"
    SECONDARY = "secondary_reserve"
    TERTIARY = "tertiary_reserve"


@dataclass
class ASProductSpec:
    """Specification for an ancillary service product"""
    name: str
    product_type: ASProduct
    dispatch_interval_min: int  # Minutes
    energy_backing_duration_min: int  # How long must energy be sustained
    response_time_sec: float  # Required response time
    min_duration_hours: float  # Minimum continuous provision duration
    symmetric: bool  # True if up/down must be equal
    requires_headroom_mw: float  # MW headroom required
    typical_price_per_mw: float  # $/MW-h typical clearing price


@dataclass
class MarketInterval:
    """Market timing parameters"""
    rtm_interval_min: int  # Real-time market interval
    dam_interval_min: int  # Day-ahead market interval
    settlement_interval_min: int  # Settlement interval
    gate_closure_min: int  # Minutes before interval for bid submission


class ISOAdapter(ABC):
    """
    Abstract base class for ISO market adapters.
    
    Each ISO adapter must implement:
    - Market timing parameters
    - Available AS products
    - Reserve requirement calculations
    - Price signal processing
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """ISO identifier (e.g., 'ERCOT', 'CAISO')"""
        pass
    
    @property
    @abstractmethod
    def market_interval(self) -> MarketInterval:
        """Market timing parameters"""
        pass
    
    @property
    @abstractmethod
    def as_products(self) -> List[ASProductSpec]:
        """Available ancillary service products"""
        pass
    
    @abstractmethod
    def get_reserve_requirement(
        self,
        product: ASProduct,
        capacity_mw: float,
        soc_mwh: Optional[float] = None,
        duration_hours: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate reserve provision capability and requirements.
        
        Returns:
            Dict with keys:
            - 'max_provision_mw': Maximum MW that can be offered
            - 'energy_backing_mwh': Required energy backing
            - 'headroom_mw': Required MW headroom
        """
        pass
    
    @abstractmethod
    def get_dispatch_intervals(self, horizon_hours: int) -> List[int]:
        """
        Get list of dispatch interval indices for optimization horizon.
        
        Args:
            horizon_hours: Planning horizon in hours
            
        Returns:
            List of interval indices (0-indexed)
        """
        pass
    
    @abstractmethod
    def get_interval_duration_hours(self) -> float:
        """Duration of each dispatch interval in hours"""
        pass
    
    def validate_bess_for_as(
        self,
        product: ASProduct,
        power_mw: float,
        energy_mwh: float,
        soc_fraction: float
    ) -> Dict[str, any]:
        """
        Validate if BESS can provide a specific AS product.
        
        Returns:
            Dict with 'eligible', 'reason', and 'max_mw' keys
        """
        spec = self._get_product_spec(product)
        if spec is None:
            return {
                "eligible": False,
                "reason": f"Product {product.value} not available in {self.name}",
                "max_mw": 0.0
            }
        
        # Check energy backing
        required_energy = power_mw * (spec.energy_backing_duration_min / 60.0)
        available_energy = energy_mwh * soc_fraction
        
        if available_energy < required_energy:
            max_mw = available_energy / (spec.energy_backing_duration_min / 60.0)
            return {
                "eligible": True,
                "reason": f"Limited by energy backing ({spec.energy_backing_duration_min} min)",
                "max_mw": max_mw
            }
        
        return {
            "eligible": True,
            "reason": "Meets all requirements",
            "max_mw": power_mw
        }
    
    def _get_product_spec(self, product: ASProduct) -> Optional[ASProductSpec]:
        """Get specification for a product type"""
        for spec in self.as_products:
            if spec.product_type == product:
                return spec
        return None
