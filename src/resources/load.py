"""
Load Model
==========

Models data center loads with different criticality levels
and flexibility characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class LoadType(Enum):
    """Load criticality types."""
    CRITICAL = "critical"          # Must always be served (Tier IV loads)
    NON_CRITICAL = "non_critical"  # Can be shed in emergency
    CURTAILABLE = "curtailable"    # Can be curtailed for economics
    SHIFTABLE = "shiftable"        # Can be shifted in time


@dataclass
class Load:
    """
    Data center load model.
    
    Represents a category of load with specific characteristics.
    
    Attributes:
        name: Load identifier
        load_type: Criticality type
        base_mw: Base load (MW)
        peak_mw: Peak load (MW)
        power_factor: Load power factor
        curtailable_fraction: Fraction that can be curtailed (0-1)
        shift_window_hours: Time window for shifting load
        curtailment_cost_per_mwh: Cost of curtailing (VoLL proxy)
    """
    name: str
    load_type: LoadType
    base_mw: float
    peak_mw: Optional[float] = None
    power_factor: float = 0.95
    curtailable_fraction: float = 0.0
    shift_window_hours: float = 0.0
    curtailment_cost_per_mwh: float = 10000.0  # Default VoLL
    
    def __post_init__(self):
        """Set defaults and validate."""
        if self.peak_mw is None:
            self.peak_mw = self.base_mw
        
        if self.base_mw < 0:
            raise ValueError("base_mw must be non-negative")
        if self.peak_mw < self.base_mw:
            raise ValueError("peak_mw must be >= base_mw")
        if not (0 < self.power_factor <= 1):
            raise ValueError("power_factor must be between 0 and 1")
        
        # Set curtailment cost based on load type
        if self.load_type == LoadType.CRITICAL:
            # Critical load has very high VoLL
            self.curtailment_cost_per_mwh = 50000.0
            self.curtailable_fraction = 0.0  # Cannot curtail
        elif self.load_type == LoadType.CURTAILABLE:
            self.curtailment_cost_per_mwh = 500.0  # Moderate cost
    
    @property
    def reactive_mvar(self) -> float:
        """Reactive power at base load."""
        import math
        tan_phi = math.sqrt(1 / self.power_factor**2 - 1)
        return self.base_mw * tan_phi
    
    @property
    def apparent_mva(self) -> float:
        """Apparent power at base load."""
        return self.base_mw / self.power_factor
    
    def get_curtailable_mw(self) -> float:
        """Get curtailable portion of load."""
        return self.base_mw * self.curtailable_fraction
    
    def get_non_curtailable_mw(self) -> float:
        """Get non-curtailable (firm) portion of load."""
        return self.base_mw * (1 - self.curtailable_fraction)
    
    def get_curtailment_cost(self, curtailed_mw: float, hours: float) -> float:
        """
        Calculate cost of curtailing load.
        
        Args:
            curtailed_mw: Amount curtailed
            hours: Duration of curtailment
            
        Returns:
            Curtailment cost ($)
        """
        return curtailed_mw * hours * self.curtailment_cost_per_mwh


@dataclass
class LoadProfile:
    """
    Time-varying load profile.
    
    Represents load over a time horizon with possible variations.
    
    Attributes:
        name: Profile identifier
        loads: List of Load objects
        profile_data: Dict of {load_name: [mw_values]} over time
        n_intervals: Number of time intervals
    """
    name: str
    loads: List[Load] = field(default_factory=list)
    profile_data: Dict[str, List[float]] = field(default_factory=dict)
    n_intervals: int = 0
    
    def add_load(self, load: Load, profile: Optional[List[float]] = None) -> None:
        """
        Add a load with optional time-varying profile.
        
        Args:
            load: Load object
            profile: List of MW values per interval (or None for constant)
        """
        self.loads.append(load)
        
        if profile is not None:
            self.profile_data[load.name] = profile
            self.n_intervals = max(self.n_intervals, len(profile))
        else:
            # Constant at base_mw
            if self.n_intervals > 0:
                self.profile_data[load.name] = [load.base_mw] * self.n_intervals
    
    def get_total_load(self, interval: int) -> dict:
        """
        Get total load at a specific interval.
        
        Args:
            interval: Time interval index
            
        Returns:
            Dict with total and breakdown by type
        """
        result = {
            "total_mw": 0.0,
            "critical_mw": 0.0,
            "non_critical_mw": 0.0,
            "curtailable_mw": 0.0,
            "shiftable_mw": 0.0,
            "by_load": {}
        }
        
        for load in self.loads:
            if load.name in self.profile_data and interval < len(self.profile_data[load.name]):
                mw = self.profile_data[load.name][interval]
            else:
                mw = load.base_mw
            
            result["total_mw"] += mw
            result["by_load"][load.name] = mw
            
            if load.load_type == LoadType.CRITICAL:
                result["critical_mw"] += mw
            elif load.load_type == LoadType.NON_CRITICAL:
                result["non_critical_mw"] += mw
            elif load.load_type == LoadType.CURTAILABLE:
                result["curtailable_mw"] += mw
            elif load.load_type == LoadType.SHIFTABLE:
                result["shiftable_mw"] += mw
        
        return result
    
    def get_load_timeseries(self) -> dict:
        """
        Get complete load timeseries.
        
        Returns:
            Dict with arrays for each load type
        """
        total = np.zeros(self.n_intervals)
        critical = np.zeros(self.n_intervals)
        non_critical = np.zeros(self.n_intervals)
        curtailable = np.zeros(self.n_intervals)
        
        for t in range(self.n_intervals):
            load_data = self.get_total_load(t)
            total[t] = load_data["total_mw"]
            critical[t] = load_data["critical_mw"]
            non_critical[t] = load_data["non_critical_mw"]
            curtailable[t] = load_data["curtailable_mw"]
        
        return {
            "total_mw": total.tolist(),
            "critical_mw": critical.tolist(),
            "non_critical_mw": non_critical.tolist(),
            "curtailable_mw": curtailable.tolist(),
            "n_intervals": self.n_intervals
        }
    
    def get_summary(self) -> dict:
        """Get load profile summary."""
        if self.n_intervals == 0:
            return {
                "n_loads": len(self.loads),
                "peak_mw": sum(l.peak_mw for l in self.loads),
                "base_mw": sum(l.base_mw for l in self.loads)
            }
        
        ts = self.get_load_timeseries()
        total_arr = np.array(ts["total_mw"])
        
        return {
            "n_loads": len(self.loads),
            "n_intervals": self.n_intervals,
            "peak_mw": float(total_arr.max()),
            "min_mw": float(total_arr.min()),
            "avg_mw": float(total_arr.mean()),
            "critical_peak_mw": float(np.array(ts["critical_mw"]).max()),
            "loads": [
                {
                    "name": l.name,
                    "type": l.load_type.value,
                    "base_mw": l.base_mw,
                    "peak_mw": l.peak_mw
                }
                for l in self.loads
            ]
        }


def create_typical_dc_profile(
    it_load_mw: float,
    cooling_fraction: float = 0.3,
    lighting_fraction: float = 0.02,
    n_intervals: int = 96,  # 15-min intervals for 24 hours
    load_variation: float = 0.1
) -> LoadProfile:
    """
    Create a typical data center load profile.
    
    Args:
        it_load_mw: IT load (servers, networking)
        cooling_fraction: Cooling as fraction of IT load
        lighting_fraction: Lighting/misc as fraction of IT load
        n_intervals: Number of time intervals
        load_variation: Fractional variation in load (0-1)
        
    Returns:
        Configured LoadProfile
    """
    profile = LoadProfile(name="DataCenter")
    
    # Critical IT load - slight diurnal variation
    t = np.linspace(0, 24, n_intervals)
    it_variation = 1 + load_variation * np.sin(2 * np.pi * t / 24 - np.pi/2)
    it_profile = (it_load_mw * it_variation).tolist()
    
    it_load = Load(
        name="IT_Load",
        load_type=LoadType.CRITICAL,
        base_mw=it_load_mw,
        peak_mw=it_load_mw * (1 + load_variation)
    )
    profile.add_load(it_load, it_profile)
    
    # Cooling load - follows IT with lag
    cooling_mw = it_load_mw * cooling_fraction
    cooling_variation = 1 + load_variation * np.sin(2 * np.pi * t / 24 - np.pi/4)
    cooling_profile = (cooling_mw * cooling_variation).tolist()
    
    cooling_load = Load(
        name="Cooling",
        load_type=LoadType.CURTAILABLE,
        base_mw=cooling_mw,
        peak_mw=cooling_mw * (1 + load_variation),
        curtailable_fraction=0.2  # Can shed 20% briefly
    )
    profile.add_load(cooling_load, cooling_profile)
    
    # Lighting and misc - constant
    lighting_mw = it_load_mw * lighting_fraction
    lighting_load = Load(
        name="Lighting_Misc",
        load_type=LoadType.NON_CRITICAL,
        base_mw=lighting_mw
    )
    profile.add_load(lighting_load, [lighting_mw] * n_intervals)
    
    return profile
