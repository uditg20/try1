"""
Gas Generator Model
===================

Models on-site gas-fired generation (turbines, reciprocating engines)
with start-up logic, heat rate curves, and N+1 redundancy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class GenStatus(Enum):
    """Generator operational status."""
    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    STOPPING = "stopping"


@dataclass
class GasGeneratorUnit:
    """
    Single gas generator unit.
    
    Attributes:
        name: Unit identifier
        capacity_mw: Maximum output (MW)
        min_output_mw: Minimum stable output (MW)
        ramp_rate_mw_per_min: Ramp rate
        start_time_min: Cold start time (minutes)
        hot_start_time_min: Hot start time (minutes)
        min_run_time_hours: Minimum run time once started
        min_down_time_hours: Minimum down time before restart
        heat_rate_mmbtu_per_mwh: Full load heat rate
        fuel_cost_per_mmbtu: Fuel cost
        vom_per_mwh: Variable O&M cost
        start_cost: Cost per start event
    """
    name: str
    capacity_mw: float
    min_output_mw: float = 0.0
    ramp_rate_mw_per_min: float = 5.0
    start_time_min: float = 10.0
    hot_start_time_min: float = 5.0
    min_run_time_hours: float = 1.0
    min_down_time_hours: float = 0.5
    heat_rate_mmbtu_per_mwh: float = 8.5
    fuel_cost_per_mmbtu: float = 3.0
    vom_per_mwh: float = 5.0
    start_cost: float = 500.0
    
    def __post_init__(self):
        """Validate parameters."""
        if self.capacity_mw <= 0:
            raise ValueError("capacity_mw must be positive")
        if self.min_output_mw < 0:
            raise ValueError("min_output_mw must be non-negative")
        if self.min_output_mw > self.capacity_mw:
            raise ValueError("min_output_mw cannot exceed capacity_mw")
    
    @property
    def variable_cost_per_mwh(self) -> float:
        """Total variable cost including fuel and VOM."""
        fuel_cost = self.heat_rate_mmbtu_per_mwh * self.fuel_cost_per_mmbtu
        return fuel_cost + self.vom_per_mwh
    
    def get_fuel_cost(self, output_mw: float, hours: float) -> float:
        """
        Calculate fuel cost for given output and duration.
        
        Uses simplified constant heat rate model.
        
        Args:
            output_mw: Generator output (MW)
            hours: Operating duration
            
        Returns:
            Fuel cost ($)
        """
        energy_mwh = output_mw * hours
        fuel_mmbtu = energy_mwh * self.heat_rate_mmbtu_per_mwh
        return fuel_mmbtu * self.fuel_cost_per_mmbtu
    
    def get_ramp_limit(self, interval_min: float) -> float:
        """Get maximum power change for interval."""
        return self.ramp_rate_mw_per_min * interval_min
    
    def get_start_intervals(self, interval_min: float, hot: bool = False) -> int:
        """Get number of intervals required to start."""
        start_time = self.hot_start_time_min if hot else self.start_time_min
        return max(1, int(math.ceil(start_time / interval_min)))


@dataclass
class GasGenerator:
    """
    Gas generator fleet with multiple units and N+1 logic.
    
    Models a collection of generator units that can be dispatched
    together with start-up sequencing and redundancy.
    
    Attributes:
        name: Fleet identifier
        units: List of generator units
        redundancy_config: 'N' or 'N+1'
    """
    name: str
    units: List[GasGeneratorUnit] = field(default_factory=list)
    redundancy_config: str = "N+1"
    
    def add_unit(self, unit: GasGeneratorUnit) -> None:
        """Add a generator unit to the fleet."""
        self.units.append(unit)
    
    @property
    def total_capacity_mw(self) -> float:
        """Total installed capacity."""
        return sum(u.capacity_mw for u in self.units)
    
    @property
    def firm_capacity_mw(self) -> float:
        """Firm capacity after N-1 contingency."""
        if not self.units:
            return 0.0
        
        if self.redundancy_config == "N":
            return self.total_capacity_mw
        elif self.redundancy_config == "N+1":
            # Lose largest unit
            largest = max(u.capacity_mw for u in self.units)
            return self.total_capacity_mw - largest
        else:
            return self.total_capacity_mw
    
    @property
    def fastest_start_min(self) -> float:
        """Fastest unit start time."""
        if not self.units:
            return float('inf')
        return min(u.start_time_min for u in self.units)
    
    @property
    def average_heat_rate(self) -> float:
        """Capacity-weighted average heat rate."""
        if not self.units:
            return 0.0
        
        total_cap = sum(u.capacity_mw for u in self.units)
        weighted = sum(u.capacity_mw * u.heat_rate_mmbtu_per_mwh for u in self.units)
        return weighted / total_cap if total_cap > 0 else 0.0
    
    def get_dispatch_cost(
        self,
        total_output_mw: float,
        hours: float,
        n_units_online: int = None
    ) -> dict:
        """
        Calculate dispatch cost for given output.
        
        Uses merit order to minimize cost.
        
        Args:
            total_output_mw: Total required output
            hours: Operating duration
            n_units_online: Number of units to use (or auto-select)
            
        Returns:
            Dict with cost breakdown and unit dispatch
        """
        if not self.units:
            return {
                "feasible": False,
                "reason": "No generator units available"
            }
        
        # Sort units by variable cost (merit order)
        sorted_units = sorted(self.units, key=lambda u: u.variable_cost_per_mwh)
        
        # Determine number of units needed
        if n_units_online is None:
            # Find minimum units to meet output
            cumulative_cap = 0
            n_units_online = 0
            for unit in sorted_units:
                cumulative_cap += unit.capacity_mw
                n_units_online += 1
                if cumulative_cap >= total_output_mw:
                    break
        
        # Check feasibility
        selected_units = sorted_units[:n_units_online]
        available_cap = sum(u.capacity_mw for u in selected_units)
        
        if available_cap < total_output_mw:
            return {
                "feasible": False,
                "reason": f"Insufficient capacity: {available_cap:.1f} MW < {total_output_mw:.1f} MW"
            }
        
        # Dispatch proportionally
        unit_dispatch = []
        remaining = total_output_mw
        total_fuel_cost = 0.0
        total_vom = 0.0
        
        for unit in selected_units:
            dispatch = min(unit.capacity_mw, remaining)
            remaining -= dispatch
            
            fuel = unit.get_fuel_cost(dispatch, hours)
            vom = dispatch * hours * unit.vom_per_mwh
            
            total_fuel_cost += fuel
            total_vom += vom
            
            unit_dispatch.append({
                "name": unit.name,
                "output_mw": dispatch,
                "fuel_cost": fuel,
                "vom": vom
            })
        
        return {
            "feasible": True,
            "total_output_mw": total_output_mw,
            "n_units_online": n_units_online,
            "fuel_cost": total_fuel_cost,
            "vom_cost": total_vom,
            "total_cost": total_fuel_cost + total_vom,
            "unit_dispatch": unit_dispatch
        }
    
    def get_n_plus_1_requirement(self, critical_load_mw: float) -> dict:
        """
        Determine generator configuration for N+1 reliability.
        
        Args:
            critical_load_mw: Critical load to serve
            
        Returns:
            Dict with required units and status
        """
        if not self.units:
            return {
                "feasible": False,
                "reason": "No generator units available"
            }
        
        # Sort by capacity
        sorted_units = sorted(self.units, key=lambda u: u.capacity_mw, reverse=True)
        
        # Find minimum units for N+1
        # After losing largest online unit, remaining must serve load
        for n in range(1, len(sorted_units) + 1):
            selected = sorted_units[:n]
            total_cap = sum(u.capacity_mw for u in selected)
            largest = max(u.capacity_mw for u in selected)
            firm_cap = total_cap - largest
            
            if firm_cap >= critical_load_mw:
                return {
                    "feasible": True,
                    "n_units_required": n,
                    "total_capacity_mw": total_cap,
                    "firm_capacity_mw": firm_cap,
                    "margin_mw": firm_cap - critical_load_mw,
                    "units": [u.name for u in selected]
                }
        
        return {
            "feasible": False,
            "reason": f"Cannot achieve N+1 for {critical_load_mw:.1f} MW critical load",
            "max_firm_capacity_mw": self.firm_capacity_mw
        }
    
    def get_parameters(self) -> dict:
        """Get generator fleet parameters."""
        return {
            "name": self.name,
            "n_units": len(self.units),
            "total_capacity_mw": self.total_capacity_mw,
            "firm_capacity_mw": self.firm_capacity_mw,
            "redundancy_config": self.redundancy_config,
            "fastest_start_min": self.fastest_start_min,
            "average_heat_rate": self.average_heat_rate,
            "units": [
                {
                    "name": u.name,
                    "capacity_mw": u.capacity_mw,
                    "min_output_mw": u.min_output_mw,
                    "start_time_min": u.start_time_min,
                    "variable_cost_per_mwh": u.variable_cost_per_mwh
                }
                for u in self.units
            ]
        }


# Import math for GasGeneratorUnit
import math
