"""
Bus Model
=========

Electrical buses connecting resources within the microgrid.
Tracks power balance and voltage regulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class BusType(Enum):
    """Types of electrical buses."""
    MAIN = "main"              # Main distribution bus
    CRITICAL = "critical"      # Critical load bus (UPS-backed)
    GENERATOR = "generator"    # Generator bus
    BESS = "bess"              # Battery bus
    AUXILIARY = "auxiliary"    # Auxiliary/balance of plant


@dataclass
class Bus:
    """
    Electrical bus within the microgrid.
    
    Represents a node where multiple resources connect.
    Enforces power balance at each timestep.
    
    Attributes:
        name: Bus identifier
        bus_type: Type of bus (main, critical, etc.)
        voltage_kv: Nominal bus voltage
        connected_resources: List of resource IDs connected to this bus
    """
    name: str
    bus_type: BusType
    voltage_kv: float
    connected_resources: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate bus parameters."""
        if self.voltage_kv <= 0:
            raise ValueError("voltage_kv must be positive")
    
    def add_resource(self, resource_id: str) -> None:
        """Add a resource to this bus."""
        if resource_id not in self.connected_resources:
            self.connected_resources.append(resource_id)
    
    def remove_resource(self, resource_id: str) -> None:
        """Remove a resource from this bus."""
        if resource_id in self.connected_resources:
            self.connected_resources.remove(resource_id)
    
    def check_power_balance(
        self,
        injections: Dict[str, float],
        tolerance_mw: float = 0.01
    ) -> dict:
        """
        Check power balance at the bus.
        
        Args:
            injections: Dict of {resource_id: power_mw}
                       Positive = injection, Negative = consumption
            tolerance_mw: Allowable imbalance
            
        Returns:
            Dict with balance status and any mismatch
        """
        total_injection = sum(injections.values())
        balanced = abs(total_injection) <= tolerance_mw
        
        return {
            "balanced": balanced,
            "net_injection_mw": total_injection,
            "tolerance_mw": tolerance_mw,
            "injections": injections
        }


@dataclass  
class PCS:
    """
    Power Conversion System for BESS.
    
    Models the inverter/converter connecting battery to AC bus.
    Enforces MVA rating and P-Q capability.
    
    Attributes:
        name: PCS identifier
        mva_rating: Apparent power rating
        max_p_mw: Maximum real power (may be less than MVA)
        efficiency: Round-trip efficiency factor
        ramp_rate_mw_per_min: Maximum ramp rate
    """
    name: str
    mva_rating: float
    max_p_mw: Optional[float] = None
    efficiency: float = 0.96
    ramp_rate_mw_per_min: float = 100.0  # Very fast for batteries
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.max_p_mw is None:
            self.max_p_mw = self.mva_rating
        if self.mva_rating <= 0:
            raise ValueError("mva_rating must be positive")
        if not (0 < self.efficiency <= 1.0):
            raise ValueError("efficiency must be between 0 and 1")
    
    def get_p_q_capability(self, priority: str = "p") -> dict:
        """
        Get P-Q capability curve for the PCS.
        
        The PCS operates within a circle defined by MVA rating.
        Can prioritize P or Q depending on application.
        
        Args:
            priority: 'p' for real power priority, 'q' for reactive
            
        Returns:
            Dict describing the capability curve
        """
        import math
        
        # Full P, then calculate available Q
        if priority == "p":
            p_max = min(self.max_p_mw, self.mva_rating)
            q_at_max_p = math.sqrt(max(0, self.mva_rating**2 - p_max**2))
            return {
                "p_max_mw": p_max,
                "p_min_mw": -p_max,  # Symmetric for BESS
                "q_max_mvar": self.mva_rating,  # At P=0
                "q_at_max_p_mvar": q_at_max_p,
                "mva_rating": self.mva_rating
            }
        else:
            # Full Q priority
            q_max = self.mva_rating
            p_at_max_q = 0.0
            return {
                "p_max_mw": self.max_p_mw,
                "q_max_mvar": q_max,
                "p_at_max_q_mw": p_at_max_q,
                "mva_rating": self.mva_rating
            }
    
    def check_operating_point(self, p_mw: float, q_mvar: float) -> dict:
        """
        Check if an operating point is within PCS capability.
        
        Args:
            p_mw: Real power (positive = discharge)
            q_mvar: Reactive power
            
        Returns:
            Dict with feasibility and any violations
        """
        import math
        
        s_mva = math.sqrt(p_mw**2 + q_mvar**2)
        violations = []
        
        # Check MVA rating
        if s_mva > self.mva_rating:
            violations.append({
                "constraint": "mva_rating",
                "limit": self.mva_rating,
                "actual": s_mva,
                "margin_mva": s_mva - self.mva_rating
            })
        
        # Check P limit
        if abs(p_mw) > self.max_p_mw:
            violations.append({
                "constraint": "max_p",
                "limit": self.max_p_mw,
                "actual": abs(p_mw),
                "margin_mw": abs(p_mw) - self.max_p_mw
            })
        
        return {
            "feasible": len(violations) == 0,
            "violations": violations,
            "s_mva": s_mva,
            "loading_pct": (s_mva / self.mva_rating) * 100
        }
    
    def get_ramp_limit(self, interval_minutes: float) -> float:
        """
        Get maximum power change for a given interval.
        
        Args:
            interval_minutes: Dispatch interval duration
            
        Returns:
            Maximum MW change in one interval
        """
        return self.ramp_rate_mw_per_min * interval_minutes
