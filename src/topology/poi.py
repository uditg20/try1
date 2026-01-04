"""
Point of Interconnection (POI)
==============================

The POI represents the electrical boundary between the microgrid
and the utility grid. It defines import/export limits and power
factor requirements.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import math


@dataclass
class PointOfInterconnection:
    """
    Point of Interconnection with the utility grid.
    
    The POI defines the electrical connection limits between
    the data center microgrid and the external grid.
    
    Attributes:
        name: Identifier for this POI
        max_import_mw: Maximum power import from grid (MW)
        max_export_mw: Maximum power export to grid (MW)
        mva_rating: Apparent power rating (MVA)
        min_power_factor: Minimum power factor requirement (leading/lagging)
        voltage_kv: Nominal voltage at POI (kV)
        x_r_ratio: System X/R ratio at POI for fault calculations
    """
    name: str
    max_import_mw: float
    max_export_mw: float
    mva_rating: float
    min_power_factor: float = 0.95
    voltage_kv: float = 138.0
    x_r_ratio: float = 10.0
    
    def __post_init__(self):
        """Validate POI parameters."""
        if self.max_import_mw < 0:
            raise ValueError("max_import_mw must be non-negative")
        if self.max_export_mw < 0:
            raise ValueError("max_export_mw must be non-negative")
        if self.mva_rating <= 0:
            raise ValueError("mva_rating must be positive")
        if not (0 < self.min_power_factor <= 1.0):
            raise ValueError("min_power_factor must be between 0 and 1")
    
    def get_q_limits(self, p_mw: float) -> Tuple[float, float]:
        """
        Calculate reactive power limits for a given real power.
        
        The POI must operate within its MVA rating and power factor limits.
        
        Args:
            p_mw: Real power flow (positive = import, negative = export)
            
        Returns:
            Tuple of (q_min_mvar, q_max_mvar)
        """
        p_abs = abs(p_mw)
        
        # MVA circle constraint: P^2 + Q^2 <= S^2
        if p_abs > self.mva_rating:
            # Operating beyond MVA rating - no valid Q
            return (0.0, 0.0)
        
        q_max_from_mva = math.sqrt(self.mva_rating**2 - p_abs**2)
        
        # Power factor constraint: |P| / S >= PF_min
        # => S <= |P| / PF_min
        # => Q <= sqrt(S^2 - P^2) = |P| * sqrt(1/PF^2 - 1)
        if self.min_power_factor < 1.0 and p_abs > 0:
            pf_factor = math.sqrt(1.0 / self.min_power_factor**2 - 1.0)
            q_max_from_pf = p_abs * pf_factor
            q_max = min(q_max_from_mva, q_max_from_pf)
        else:
            q_max = q_max_from_mva
        
        return (-q_max, q_max)
    
    def check_operating_point(
        self,
        p_mw: float,
        q_mvar: float
    ) -> dict:
        """
        Check if an operating point is feasible at the POI.
        
        Args:
            p_mw: Real power (positive = import, negative = export)
            q_mvar: Reactive power (positive = inductive/lagging)
            
        Returns:
            Dict with feasibility status and any violations
        """
        violations = []
        
        # Check MW limits
        if p_mw > self.max_import_mw:
            violations.append({
                "constraint": "max_import",
                "limit": self.max_import_mw,
                "actual": p_mw,
                "margin_mw": p_mw - self.max_import_mw
            })
        
        if p_mw < -self.max_export_mw:
            violations.append({
                "constraint": "max_export",
                "limit": self.max_export_mw,
                "actual": -p_mw,
                "margin_mw": -p_mw - self.max_export_mw
            })
        
        # Check MVA rating
        s_mva = math.sqrt(p_mw**2 + q_mvar**2)
        if s_mva > self.mva_rating:
            violations.append({
                "constraint": "mva_rating",
                "limit": self.mva_rating,
                "actual": s_mva,
                "margin_mva": s_mva - self.mva_rating
            })
        
        # Check power factor
        if s_mva > 0.01:  # Avoid division by zero
            pf = abs(p_mw) / s_mva
            if pf < self.min_power_factor:
                violations.append({
                    "constraint": "power_factor",
                    "limit": self.min_power_factor,
                    "actual": pf,
                    "margin_pf": self.min_power_factor - pf
                })
        
        return {
            "feasible": len(violations) == 0,
            "violations": violations,
            "operating_point": {
                "p_mw": p_mw,
                "q_mvar": q_mvar,
                "s_mva": s_mva,
                "pf": abs(p_mw) / s_mva if s_mva > 0.01 else 1.0
            }
        }
    
    def get_operating_envelope(self, num_points: int = 100) -> dict:
        """
        Generate the feasible operating envelope (P-Q capability curve).
        
        Returns:
            Dict with P and Q arrays defining the feasible region
        """
        import numpy as np
        
        # P ranges from -export to +import
        p_values = np.linspace(-self.max_export_mw, self.max_import_mw, num_points)
        
        q_min_values = []
        q_max_values = []
        
        for p in p_values:
            q_min, q_max = self.get_q_limits(p)
            q_min_values.append(q_min)
            q_max_values.append(q_max)
        
        return {
            "p_mw": p_values.tolist(),
            "q_min_mvar": q_min_values,
            "q_max_mvar": q_max_values,
            "mva_rating": self.mva_rating,
            "min_pf": self.min_power_factor
        }
