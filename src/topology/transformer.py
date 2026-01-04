"""
Transformer Model
=================

Models step-down transformers between the POI and internal buses.
Enforces MVA ratings and tracks loading.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class Transformer:
    """
    Power transformer connecting grid to microgrid buses.
    
    Attributes:
        name: Transformer identifier
        mva_rating: Nameplate MVA rating
        primary_kv: Primary (high) side voltage
        secondary_kv: Secondary (low) side voltage
        impedance_pu: Per-unit impedance (on transformer base)
        x_r_ratio: X/R ratio for losses calculation
        tap_range: Tap adjustment range (e.g., +/- 10%)
        redundancy_config: 'N', 'N+1', '2N' configuration
    """
    name: str
    mva_rating: float
    primary_kv: float
    secondary_kv: float
    impedance_pu: float = 0.06
    x_r_ratio: float = 15.0
    tap_range: Tuple[float, float] = (-0.1, 0.1)
    redundancy_config: str = "N+1"
    
    def __post_init__(self):
        """Validate transformer parameters."""
        if self.mva_rating <= 0:
            raise ValueError("mva_rating must be positive")
        if self.primary_kv <= 0 or self.secondary_kv <= 0:
            raise ValueError("Voltage ratings must be positive")
    
    @property
    def turns_ratio(self) -> float:
        """Nominal turns ratio (primary/secondary)."""
        return self.primary_kv / self.secondary_kv
    
    @property
    def resistance_pu(self) -> float:
        """Per-unit resistance."""
        return self.impedance_pu / math.sqrt(1 + self.x_r_ratio**2)
    
    @property
    def reactance_pu(self) -> float:
        """Per-unit reactance."""
        return self.impedance_pu * self.x_r_ratio / math.sqrt(1 + self.x_r_ratio**2)
    
    def get_loading(self, p_mw: float, q_mvar: float) -> dict:
        """
        Calculate transformer loading for given power flow.
        
        Args:
            p_mw: Real power through transformer
            q_mvar: Reactive power through transformer
            
        Returns:
            Dict with loading percentage and status
        """
        s_mva = math.sqrt(p_mw**2 + q_mvar**2)
        loading_pct = (s_mva / self.mva_rating) * 100
        
        # Determine status based on loading
        if loading_pct <= 80:
            status = "normal"
        elif loading_pct <= 100:
            status = "elevated"
        elif loading_pct <= 120:
            status = "emergency"
        else:
            status = "overload"
        
        return {
            "s_mva": s_mva,
            "loading_pct": loading_pct,
            "status": status,
            "margin_mva": self.mva_rating - s_mva,
            "feasible": s_mva <= self.mva_rating
        }
    
    def get_losses(self, p_mw: float, q_mvar: float) -> dict:
        """
        Estimate transformer losses.
        
        Uses simplified loss model based on loading.
        
        Args:
            p_mw: Real power flow
            q_mvar: Reactive power flow
            
        Returns:
            Dict with real and reactive losses
        """
        s_mva = math.sqrt(p_mw**2 + q_mvar**2)
        loading_pu = s_mva / self.mva_rating
        
        # No-load losses (approx 0.2% of rating)
        p_no_load = 0.002 * self.mva_rating
        
        # Load losses (I^2 * R, proportional to loading squared)
        # Copper losses at full load approx 0.5% of rating
        p_load_loss = 0.005 * self.mva_rating * (loading_pu ** 2)
        
        # Reactive losses
        q_loss = 0.01 * self.mva_rating * (loading_pu ** 2)
        
        return {
            "p_loss_mw": p_no_load + p_load_loss,
            "q_loss_mvar": q_loss,
            "no_load_mw": p_no_load,
            "load_loss_mw": p_load_loss
        }
    
    def get_effective_rating(self) -> float:
        """
        Get effective MVA rating considering redundancy.
        
        For N+1 config, only N/(N+1) capacity is firm.
        For 2N config, only 50% is firm.
        """
        if self.redundancy_config == "N":
            return self.mva_rating
        elif self.redundancy_config == "N+1":
            # Assume 2 units in N+1, so 50% is firm
            # For larger configs, would need to know N
            return self.mva_rating * 0.5
        elif self.redundancy_config == "2N":
            return self.mva_rating * 0.5
        else:
            return self.mva_rating


@dataclass
class TransformerBank:
    """
    Bank of parallel transformers with redundancy.
    
    Models multiple transformers operating in parallel,
    sharing load with N+1 or 2N redundancy.
    """
    name: str
    transformers: List[Transformer]
    redundancy_config: str = "N+1"
    
    @property
    def total_mva(self) -> float:
        """Total MVA of all transformers."""
        return sum(t.mva_rating for t in self.transformers)
    
    @property
    def firm_mva(self) -> float:
        """Firm (available after contingency) MVA."""
        if not self.transformers:
            return 0.0
        
        if self.redundancy_config == "N":
            return self.total_mva
        elif self.redundancy_config == "N+1":
            # Lose largest unit
            largest = max(t.mva_rating for t in self.transformers)
            return self.total_mva - largest
        elif self.redundancy_config == "2N":
            return self.total_mva / 2
        else:
            return self.total_mva
    
    def get_loading(self, p_mw: float, q_mvar: float) -> dict:
        """
        Calculate bank loading assuming equal sharing.
        
        Args:
            p_mw: Total real power through bank
            q_mvar: Total reactive power through bank
            
        Returns:
            Dict with loading status per transformer
        """
        if not self.transformers:
            return {"feasible": False, "reason": "No transformers in bank"}
        
        n = len(self.transformers)
        p_per_xfmr = p_mw / n
        q_per_xfmr = q_mvar / n
        
        results = []
        all_feasible = True
        
        for xfmr in self.transformers:
            loading = xfmr.get_loading(p_per_xfmr, q_per_xfmr)
            results.append({
                "name": xfmr.name,
                **loading
            })
            if not loading["feasible"]:
                all_feasible = False
        
        return {
            "feasible": all_feasible,
            "transformers": results,
            "total_s_mva": math.sqrt(p_mw**2 + q_mvar**2),
            "bank_loading_pct": (math.sqrt(p_mw**2 + q_mvar**2) / self.total_mva) * 100
        }
