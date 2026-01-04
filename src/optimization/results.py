"""
Optimization Results
====================

Structured container for MILP optimization results with
explainability and constraint analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class ConstraintStatus:
    """Status of a constraint in the solution."""
    name: str
    binding: bool
    slack: float
    dual_value: Optional[float] = None
    description: str = ""


@dataclass
class IntervalResult:
    """Results for a single dispatch interval."""
    interval: int
    timestamp_hours: float
    
    # Power flows
    grid_import_mw: float
    grid_export_mw: float
    grid_net_mw: float  # Positive = import
    
    # BESS
    bess_charge_mw: float
    bess_discharge_mw: float
    bess_soc: float
    bess_soc_mwh: float
    
    # Generators
    gen_output_mw: float
    gen_units_online: int
    gen_starting: bool
    
    # Loads
    total_load_mw: float
    critical_load_mw: float
    curtailed_mw: float
    unserved_mw: float
    
    # Reserves
    reg_up_mw: float = 0.0
    reg_down_mw: float = 0.0
    spin_mw: float = 0.0
    
    # Economics (for this interval)
    energy_cost: float = 0.0
    energy_revenue: float = 0.0
    as_revenue: float = 0.0
    fuel_cost: float = 0.0
    degradation_cost: float = 0.0
    curtailment_cost: float = 0.0
    
    # Binding constraints
    binding_constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "interval": self.interval,
            "timestamp_hours": self.timestamp_hours,
            "grid_import_mw": self.grid_import_mw,
            "grid_export_mw": self.grid_export_mw,
            "grid_net_mw": self.grid_net_mw,
            "bess_charge_mw": self.bess_charge_mw,
            "bess_discharge_mw": self.bess_discharge_mw,
            "bess_soc": self.bess_soc,
            "bess_soc_mwh": self.bess_soc_mwh,
            "gen_output_mw": self.gen_output_mw,
            "gen_units_online": self.gen_units_online,
            "gen_starting": self.gen_starting,
            "total_load_mw": self.total_load_mw,
            "critical_load_mw": self.critical_load_mw,
            "curtailed_mw": self.curtailed_mw,
            "unserved_mw": self.unserved_mw,
            "reg_up_mw": self.reg_up_mw,
            "reg_down_mw": self.reg_down_mw,
            "spin_mw": self.spin_mw,
            "energy_cost": self.energy_cost,
            "energy_revenue": self.energy_revenue,
            "as_revenue": self.as_revenue,
            "fuel_cost": self.fuel_cost,
            "degradation_cost": self.degradation_cost,
            "curtailment_cost": self.curtailment_cost,
            "binding_constraints": self.binding_constraints
        }


@dataclass
class OptimizationResults:
    """
    Complete optimization results with explainability.
    
    Contains:
    - Dispatch schedule for all intervals
    - Economic summary
    - Constraint analysis
    - Decision explanations
    """
    # Solution status
    solved: bool
    solver_status: str
    solve_time_sec: float
    objective_value: float
    
    # Time parameters
    n_intervals: int
    interval_hours: float
    horizon_hours: float
    
    # Interval results
    intervals: List[IntervalResult] = field(default_factory=list)
    
    # Aggregated economics
    total_energy_cost: float = 0.0
    total_energy_revenue: float = 0.0
    total_as_revenue: float = 0.0
    total_fuel_cost: float = 0.0
    total_degradation_cost: float = 0.0
    total_curtailment_cost: float = 0.0
    net_value: float = 0.0
    
    # Reliability metrics
    total_unserved_mwh: float = 0.0
    max_unserved_mw: float = 0.0
    ride_through_achieved: bool = True
    
    # Constraint analysis
    constraint_status: List[ConstraintStatus] = field(default_factory=list)
    binding_constraint_summary: Dict[str, int] = field(default_factory=dict)
    
    # Decision explanations
    explanations: Dict[int, List[str]] = field(default_factory=dict)
    
    def calculate_aggregates(self) -> None:
        """Calculate aggregate values from interval results."""
        self.total_energy_cost = sum(i.energy_cost for i in self.intervals)
        self.total_energy_revenue = sum(i.energy_revenue for i in self.intervals)
        self.total_as_revenue = sum(i.as_revenue for i in self.intervals)
        self.total_fuel_cost = sum(i.fuel_cost for i in self.intervals)
        self.total_degradation_cost = sum(i.degradation_cost for i in self.intervals)
        self.total_curtailment_cost = sum(i.curtailment_cost for i in self.intervals)
        
        self.net_value = (
            self.total_energy_revenue 
            + self.total_as_revenue
            - self.total_energy_cost
            - self.total_fuel_cost
            - self.total_degradation_cost
            - self.total_curtailment_cost
        )
        
        # Reliability
        self.total_unserved_mwh = sum(
            i.unserved_mw * self.interval_hours for i in self.intervals
        )
        self.max_unserved_mw = max(i.unserved_mw for i in self.intervals) if self.intervals else 0.0
        self.ride_through_achieved = self.total_unserved_mwh < 0.001
        
        # Count binding constraints
        self.binding_constraint_summary = {}
        for interval in self.intervals:
            for constraint in interval.binding_constraints:
                self.binding_constraint_summary[constraint] = (
                    self.binding_constraint_summary.get(constraint, 0) + 1
                )
    
    def get_timeseries(self) -> dict:
        """Get dispatch timeseries for plotting."""
        return {
            "intervals": [i.interval for i in self.intervals],
            "hours": [i.timestamp_hours for i in self.intervals],
            "grid_import_mw": [i.grid_import_mw for i in self.intervals],
            "grid_export_mw": [i.grid_export_mw for i in self.intervals],
            "grid_net_mw": [i.grid_net_mw for i in self.intervals],
            "bess_charge_mw": [i.bess_charge_mw for i in self.intervals],
            "bess_discharge_mw": [i.bess_discharge_mw for i in self.intervals],
            "bess_net_mw": [i.bess_discharge_mw - i.bess_charge_mw for i in self.intervals],
            "bess_soc": [i.bess_soc for i in self.intervals],
            "bess_soc_mwh": [i.bess_soc_mwh for i in self.intervals],
            "gen_output_mw": [i.gen_output_mw for i in self.intervals],
            "gen_units_online": [i.gen_units_online for i in self.intervals],
            "total_load_mw": [i.total_load_mw for i in self.intervals],
            "critical_load_mw": [i.critical_load_mw for i in self.intervals],
            "curtailed_mw": [i.curtailed_mw for i in self.intervals],
            "unserved_mw": [i.unserved_mw for i in self.intervals],
            "reg_up_mw": [i.reg_up_mw for i in self.intervals],
            "reg_down_mw": [i.reg_down_mw for i in self.intervals],
            "spin_mw": [i.spin_mw for i in self.intervals]
        }
    
    def get_economics_summary(self) -> dict:
        """Get economics summary for display."""
        return {
            "revenue": {
                "energy_export": self.total_energy_revenue,
                "ancillary_services": self.total_as_revenue,
                "total_revenue": self.total_energy_revenue + self.total_as_revenue
            },
            "costs": {
                "energy_import": self.total_energy_cost,
                "fuel": self.total_fuel_cost,
                "degradation": self.total_degradation_cost,
                "curtailment": self.total_curtailment_cost,
                "total_cost": (
                    self.total_energy_cost 
                    + self.total_fuel_cost 
                    + self.total_degradation_cost
                    + self.total_curtailment_cost
                )
            },
            "net_value": self.net_value,
            "horizon_hours": self.horizon_hours
        }
    
    def get_reliability_summary(self) -> dict:
        """Get reliability metrics summary."""
        return {
            "unserved_mwh": self.total_unserved_mwh,
            "max_unserved_mw": self.max_unserved_mw,
            "ride_through_achieved": self.ride_through_achieved,
            "intervals_with_curtailment": sum(
                1 for i in self.intervals if i.curtailed_mw > 0.001
            ),
            "intervals_with_unserved": sum(
                1 for i in self.intervals if i.unserved_mw > 0.001
            )
        }
    
    def explain_interval(self, interval_idx: int) -> List[str]:
        """
        Generate explanations for decisions at an interval.
        
        Returns human-readable explanations based on binding constraints.
        """
        if interval_idx >= len(self.intervals):
            return ["Invalid interval index"]
        
        result = self.intervals[interval_idx]
        explanations = []
        
        # BESS decisions
        if result.bess_discharge_mw > 0.001:
            if "energy_price_high" in result.binding_constraints:
                explanations.append(
                    f"Battery discharging {result.bess_discharge_mw:.1f} MW: "
                    "Energy price exceeds marginal value of stored energy"
                )
            elif "load_balance" in result.binding_constraints:
                explanations.append(
                    f"Battery discharging {result.bess_discharge_mw:.1f} MW: "
                    "Required to meet load balance"
                )
        
        if result.bess_charge_mw > 0.001:
            if "energy_price_low" in result.binding_constraints:
                explanations.append(
                    f"Battery charging {result.bess_charge_mw:.1f} MW: "
                    "Energy price below value of future discharge"
                )
        
        if abs(result.bess_discharge_mw) < 0.001 and abs(result.bess_charge_mw) < 0.001:
            if "soc_reserve" in result.binding_constraints:
                explanations.append(
                    "Battery idle: SOC at reserve floor, cannot discharge for market"
                )
            elif "soc_max" in result.binding_constraints:
                explanations.append(
                    "Battery idle: SOC at maximum, cannot charge"
                )
            else:
                explanations.append(
                    "Battery idle: Holding energy for higher-value periods"
                )
        
        # Generator decisions
        if result.gen_output_mw > 0.001:
            if "grid_limit" in result.binding_constraints:
                explanations.append(
                    f"Generator running {result.gen_output_mw:.1f} MW: "
                    "Grid import limit binding"
                )
            elif "price_arbitrage" in result.binding_constraints:
                explanations.append(
                    f"Generator running {result.gen_output_mw:.1f} MW: "
                    "Grid price exceeds generation cost"
                )
        
        if result.gen_starting:
            explanations.append(
                "Generator starting: Reliability requirement or price signal"
            )
        
        # Grid decisions
        if result.grid_export_mw > 0.001:
            explanations.append(
                f"Exporting {result.grid_export_mw:.1f} MW to grid: "
                "Excess generation at favorable price"
            )
        
        # Curtailment
        if result.curtailed_mw > 0.001:
            explanations.append(
                f"WARNING: Curtailing {result.curtailed_mw:.1f} MW load: "
                "Insufficient supply or binding constraints"
            )
        
        if result.unserved_mw > 0.001:
            explanations.append(
                f"CRITICAL: {result.unserved_mw:.1f} MW unserved load: "
                "System capacity or constraints insufficient"
            )
        
        # Reserve provision
        if result.reg_up_mw > 0.001:
            explanations.append(
                f"Providing {result.reg_up_mw:.1f} MW Reg-Up from BESS headroom"
            )
        
        if not explanations:
            explanations.append("Normal operation - no binding constraints")
        
        return explanations
    
    def to_dict(self) -> dict:
        """Convert complete results to dictionary."""
        return {
            "solved": self.solved,
            "solver_status": self.solver_status,
            "solve_time_sec": self.solve_time_sec,
            "objective_value": self.objective_value,
            "n_intervals": self.n_intervals,
            "interval_hours": self.interval_hours,
            "horizon_hours": self.horizon_hours,
            "timeseries": self.get_timeseries(),
            "economics": self.get_economics_summary(),
            "reliability": self.get_reliability_summary(),
            "binding_constraints": self.binding_constraint_summary,
            "intervals": [i.to_dict() for i in self.intervals]
        }
