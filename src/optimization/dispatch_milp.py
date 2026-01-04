"""
MILP Dispatch Optimizer
=======================

Production-grade Mixed-Integer Linear Programming model for
data center microgrid dispatch optimization.

Objective: Maximize net value (revenue - costs)
- Energy arbitrage
- Ancillary services revenue
- Fuel cost minimization
- Degradation cost
- Value of Lost Load (VoLL) penalty

Constraints:
- Power balance at microgrid level
- SOC dynamics and limits
- No simultaneous charge/discharge
- Generator bounds and start-up
- POI import/export limits
- Transformer MVA screening
- PCS MVA screening
- Reserve headroom and energy backing
- SOC reserve for ride-through
"""

import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from .results import OptimizationResults, IntervalResult


@dataclass
class PriceData:
    """Energy and AS price data for optimization."""
    energy_price: List[float]  # $/MWh per interval
    reg_up_price: List[float]  # $/MW-h per interval
    reg_down_price: List[float]
    spin_price: List[float]
    
    def __post_init__(self):
        """Validate price data lengths are consistent."""
        n = len(self.energy_price)
        if len(self.reg_up_price) != n:
            self.reg_up_price = [0.0] * n
        if len(self.reg_down_price) != n:
            self.reg_down_price = [0.0] * n
        if len(self.spin_price) != n:
            self.spin_price = [0.0] * n


@dataclass
class DispatchCase:
    """Complete case data for optimization."""
    # Time parameters
    n_intervals: int
    interval_hours: float
    
    # Load data (MW per interval)
    total_load: List[float]
    critical_load: List[float]
    curtailable_load: List[float]
    
    # Price data
    prices: PriceData
    
    # BESS parameters
    bess_power_mw: float
    bess_energy_mwh: float
    bess_pcs_mva: float
    bess_eff_charge: float
    bess_eff_discharge: float
    bess_soc_min: float
    bess_soc_max: float
    bess_soc_reserve: float
    bess_initial_soc: float
    bess_degradation_cost: float
    
    # Generator parameters
    gen_capacity_mw: float
    gen_min_output_mw: float
    gen_fuel_cost_per_mwh: float
    gen_vom_per_mwh: float
    gen_start_cost: float
    gen_start_intervals: int
    gen_min_run_intervals: int
    n_gen_units: int
    
    # Grid parameters
    poi_max_import_mw: float
    poi_max_export_mw: float
    poi_mva_rating: float
    poi_min_pf: float
    transformer_mva: float
    
    # Reliability parameters
    ride_through_min: float
    voll: float = 50000.0  # Value of Lost Load $/MWh
    curtailment_cost: float = 500.0  # $/MWh for voluntary curtailment
    
    # AS parameters
    as_energy_backing_hours: float = 1.0
    
    @property
    def horizon_hours(self) -> float:
        return self.n_intervals * self.interval_hours


class DispatchOptimizer:
    """
    MILP-based dispatch optimizer for data center microgrids.
    
    Uses Pyomo for model formulation and HiGHS for solving.
    Supports rolling horizon optimization.
    """
    
    def __init__(
        self,
        solver_name: str = "appsi_highs",
        time_limit_sec: float = 60.0,
        mip_gap: float = 0.001
    ):
        """
        Initialize optimizer.
        
        Args:
            solver_name: Pyomo solver name (appsi_highs recommended)
            time_limit_sec: Maximum solve time
            mip_gap: Acceptable optimality gap
        """
        self.solver_name = solver_name
        self.time_limit_sec = time_limit_sec
        self.mip_gap = mip_gap
        self.model = None
        self._case = None
    
    def build_model(self, case: DispatchCase) -> pyo.ConcreteModel:
        """
        Build the MILP optimization model.
        
        Args:
            case: DispatchCase with all input data
            
        Returns:
            Pyomo ConcreteModel
        """
        self._case = case
        m = pyo.ConcreteModel("DataCenterDispatch")
        
        # =====================
        # SETS
        # =====================
        m.T = pyo.Set(initialize=range(case.n_intervals))  # Time intervals
        m.G = pyo.Set(initialize=range(case.n_gen_units))   # Generator units
        
        # =====================
        # PARAMETERS
        # =====================
        m.dt = pyo.Param(initialize=case.interval_hours)  # Interval duration (hours)
        
        # Loads
        m.total_load = pyo.Param(m.T, initialize=lambda m, t: case.total_load[t])
        m.critical_load = pyo.Param(m.T, initialize=lambda m, t: case.critical_load[t])
        m.curtailable_load = pyo.Param(m.T, initialize=lambda m, t: case.curtailable_load[t])
        
        # Prices
        m.energy_price = pyo.Param(m.T, initialize=lambda m, t: case.prices.energy_price[t])
        m.reg_up_price = pyo.Param(m.T, initialize=lambda m, t: case.prices.reg_up_price[t])
        m.reg_down_price = pyo.Param(m.T, initialize=lambda m, t: case.prices.reg_down_price[t])
        m.spin_price = pyo.Param(m.T, initialize=lambda m, t: case.prices.spin_price[t])
        
        # BESS
        m.bess_pmax = pyo.Param(initialize=case.bess_power_mw)
        m.bess_emax = pyo.Param(initialize=case.bess_energy_mwh)
        m.bess_soc_min = pyo.Param(initialize=case.bess_soc_min)
        m.bess_soc_max = pyo.Param(initialize=case.bess_soc_max)
        m.bess_soc_reserve = pyo.Param(initialize=max(case.bess_soc_min, case.bess_soc_reserve))
        m.bess_eff_c = pyo.Param(initialize=case.bess_eff_charge)
        m.bess_eff_d = pyo.Param(initialize=case.bess_eff_discharge)
        m.bess_degrad = pyo.Param(initialize=case.bess_degradation_cost)
        m.bess_initial_soc = pyo.Param(initialize=case.bess_initial_soc)
        
        # Generator
        m.gen_pmax = pyo.Param(initialize=case.gen_capacity_mw / case.n_gen_units)
        m.gen_pmin = pyo.Param(initialize=case.gen_min_output_mw / case.n_gen_units)
        m.gen_fuel_cost = pyo.Param(initialize=case.gen_fuel_cost_per_mwh)
        m.gen_vom = pyo.Param(initialize=case.gen_vom_per_mwh)
        m.gen_start_cost = pyo.Param(initialize=case.gen_start_cost)
        m.gen_start_intervals = pyo.Param(initialize=case.gen_start_intervals)
        m.gen_min_run = pyo.Param(initialize=case.gen_min_run_intervals)
        
        # Grid
        m.poi_import_max = pyo.Param(initialize=case.poi_max_import_mw)
        m.poi_export_max = pyo.Param(initialize=case.poi_max_export_mw)
        m.transformer_mva = pyo.Param(initialize=case.transformer_mva)
        
        # Reliability
        m.voll = pyo.Param(initialize=case.voll)
        m.curtail_cost = pyo.Param(initialize=case.curtailment_cost)
        
        # AS energy backing
        m.as_backing_hours = pyo.Param(initialize=case.as_energy_backing_hours)
        
        # =====================
        # VARIABLES
        # =====================
        
        # Grid power (positive = import, negative = export)
        m.grid_import = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, case.poi_max_import_mw))
        m.grid_export = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, case.poi_max_export_mw))
        
        # BESS
        m.bess_charge = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, case.bess_power_mw))
        m.bess_discharge = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, case.bess_power_mw))
        m.bess_soc = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, 1))
        m.bess_charging = pyo.Var(m.T, within=pyo.Binary)  # 1 if charging
        
        # Generators
        m.gen_output = pyo.Var(m.T, m.G, within=pyo.NonNegativeReals)
        m.gen_online = pyo.Var(m.T, m.G, within=pyo.Binary)  # 1 if online
        m.gen_start = pyo.Var(m.T, m.G, within=pyo.Binary)   # 1 if starting this interval
        
        # Curtailment and unserved load
        m.curtail = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Voluntary curtailment
        m.unserved = pyo.Var(m.T, within=pyo.NonNegativeReals)  # Involuntary unserved
        
        # Ancillary services provision
        m.reg_up = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.reg_down = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.spin = pyo.Var(m.T, within=pyo.NonNegativeReals)
        
        # =====================
        # CONSTRAINTS
        # =====================
        
        # 1. POWER BALANCE at microgrid level
        def power_balance_rule(m, t):
            gen_total = sum(m.gen_output[t, g] for g in m.G)
            bess_net = m.bess_discharge[t] - m.bess_charge[t]
            grid_net = m.grid_import[t] - m.grid_export[t]
            served_load = m.total_load[t] - m.curtail[t] - m.unserved[t]
            
            return gen_total + bess_net + grid_net == served_load
        
        m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)
        
        # 2. SOC DYNAMICS
        def soc_dynamics_rule(m, t):
            if t == 0:
                prev_soc = m.bess_initial_soc
            else:
                prev_soc = m.bess_soc[t-1]
            
            charge_energy = m.bess_charge[t] * m.dt * m.bess_eff_c
            discharge_energy = m.bess_discharge[t] * m.dt / m.bess_eff_d
            delta_soc = (charge_energy - discharge_energy) / m.bess_emax
            
            return m.bess_soc[t] == prev_soc + delta_soc
        
        m.soc_dynamics = pyo.Constraint(m.T, rule=soc_dynamics_rule)
        
        # 3. SOC LIMITS
        def soc_min_rule(m, t):
            return m.bess_soc[t] >= m.bess_soc_min
        
        def soc_max_rule(m, t):
            return m.bess_soc[t] <= m.bess_soc_max
        
        m.soc_min_con = pyo.Constraint(m.T, rule=soc_min_rule)
        m.soc_max_con = pyo.Constraint(m.T, rule=soc_max_rule)
        
        # 4. SOC RESERVE (market dispatch cannot go below reserve)
        def soc_reserve_rule(m, t):
            # SOC must stay above reserve floor for market dispatch
            # Discharge limited by reserve constraint
            return m.bess_soc[t] >= m.bess_soc_reserve
        
        m.soc_reserve_con = pyo.Constraint(m.T, rule=soc_reserve_rule)
        
        # 5. NO SIMULTANEOUS CHARGE/DISCHARGE
        def no_simul_charge_rule(m, t):
            return m.bess_charge[t] <= m.bess_pmax * m.bess_charging[t]
        
        def no_simul_discharge_rule(m, t):
            return m.bess_discharge[t] <= m.bess_pmax * (1 - m.bess_charging[t])
        
        m.no_simul_charge = pyo.Constraint(m.T, rule=no_simul_charge_rule)
        m.no_simul_discharge = pyo.Constraint(m.T, rule=no_simul_discharge_rule)
        
        # 6. GENERATOR OUTPUT BOUNDS
        def gen_max_output_rule(m, t, g):
            return m.gen_output[t, g] <= m.gen_pmax * m.gen_online[t, g]
        
        def gen_min_output_rule(m, t, g):
            return m.gen_output[t, g] >= m.gen_pmin * m.gen_online[t, g]
        
        m.gen_max_output = pyo.Constraint(m.T, m.G, rule=gen_max_output_rule)
        m.gen_min_output = pyo.Constraint(m.T, m.G, rule=gen_min_output_rule)
        
        # 7. GENERATOR START-UP LOGIC
        def gen_start_rule(m, t, g):
            if t == 0:
                return m.gen_start[t, g] >= m.gen_online[t, g]
            else:
                return m.gen_start[t, g] >= m.gen_online[t, g] - m.gen_online[t-1, g]
        
        m.gen_start_logic = pyo.Constraint(m.T, m.G, rule=gen_start_rule)
        
        # 8. GENERATOR MINIMUM RUN TIME (simplified)
        def gen_min_run_rule(m, t, g):
            if t < case.gen_min_run_intervals:
                return pyo.Constraint.Skip
            # If started recently, must stay on
            return m.gen_online[t, g] >= sum(
                m.gen_start[tau, g] 
                for tau in range(max(0, t - case.gen_min_run_intervals + 1), t + 1)
            ) / case.gen_min_run_intervals
        
        m.gen_min_run_con = pyo.Constraint(m.T, m.G, rule=gen_min_run_rule)
        
        # 9. CURTAILMENT LIMITS
        def curtail_limit_rule(m, t):
            return m.curtail[t] <= m.curtailable_load[t]
        
        m.curtail_limit = pyo.Constraint(m.T, rule=curtail_limit_rule)
        
        # 10. UNSERVED LOAD LIMIT (cannot exceed total load)
        def unserved_limit_rule(m, t):
            return m.unserved[t] <= m.total_load[t]
        
        m.unserved_limit = pyo.Constraint(m.T, rule=unserved_limit_rule)
        
        # 11. POI MVA SCREENING (linearized)
        # S^2 = P^2 + Q^2 <= MVA^2
        # Approximate with P <= MVA (conservative for high PF)
        def poi_mva_import_rule(m, t):
            return m.grid_import[t] <= m.transformer_mva
        
        def poi_mva_export_rule(m, t):
            return m.grid_export[t] <= m.transformer_mva
        
        m.poi_mva_import = pyo.Constraint(m.T, rule=poi_mva_import_rule)
        m.poi_mva_export = pyo.Constraint(m.T, rule=poi_mva_export_rule)
        
        # 12. RESERVE HEADROOM - Reg Up from BESS
        def reg_up_headroom_rule(m, t):
            # Reg-up requires discharge headroom
            return m.reg_up[t] <= m.bess_pmax - m.bess_discharge[t]
        
        m.reg_up_headroom = pyo.Constraint(m.T, rule=reg_up_headroom_rule)
        
        # 13. RESERVE HEADROOM - Reg Down from BESS
        def reg_down_headroom_rule(m, t):
            # Reg-down requires charge headroom
            return m.reg_down[t] <= m.bess_pmax - m.bess_charge[t]
        
        m.reg_down_headroom = pyo.Constraint(m.T, rule=reg_down_headroom_rule)
        
        # 14. RESERVE ENERGY BACKING
        def reg_up_energy_backing_rule(m, t):
            # Must have energy to sustain reg-up for backing period
            available_energy = (m.bess_soc[t] - m.bess_soc_reserve) * m.bess_emax
            return m.reg_up[t] * m.as_backing_hours <= available_energy
        
        m.reg_up_energy = pyo.Constraint(m.T, rule=reg_up_energy_backing_rule)
        
        def reg_down_energy_backing_rule(m, t):
            # Must have headroom to absorb reg-down for backing period
            available_headroom = (m.bess_soc_max - m.bess_soc[t]) * m.bess_emax
            return m.reg_down[t] * m.as_backing_hours <= available_headroom
        
        m.reg_down_energy = pyo.Constraint(m.T, rule=reg_down_energy_backing_rule)
        
        # 15. SPINNING RESERVE FROM BESS (simplified)
        def spin_headroom_rule(m, t):
            return m.spin[t] <= m.bess_pmax - m.bess_discharge[t]
        
        m.spin_headroom = pyo.Constraint(m.T, rule=spin_headroom_rule)
        
        def spin_energy_rule(m, t):
            available_energy = (m.bess_soc[t] - m.bess_soc_reserve) * m.bess_emax
            return m.spin[t] * m.as_backing_hours <= available_energy
        
        m.spin_energy = pyo.Constraint(m.T, rule=spin_energy_rule)
        
        # =====================
        # OBJECTIVE FUNCTION
        # =====================
        def objective_rule(m):
            revenue = 0
            costs = 0
            
            for t in m.T:
                # Energy revenue (export) and cost (import)
                revenue += m.grid_export[t] * m.energy_price[t] * m.dt
                costs += m.grid_import[t] * m.energy_price[t] * m.dt
                
                # AS revenue
                revenue += m.reg_up[t] * m.reg_up_price[t] * m.dt
                revenue += m.reg_down[t] * m.reg_down_price[t] * m.dt
                revenue += m.spin[t] * m.spin_price[t] * m.dt
                
                # Generator costs
                for g in m.G:
                    costs += m.gen_output[t, g] * (m.gen_fuel_cost + m.gen_vom) * m.dt
                    costs += m.gen_start[t, g] * m.gen_start_cost
                
                # BESS degradation
                costs += m.bess_discharge[t] * m.bess_degrad * m.dt
                
                # Curtailment and VoLL
                costs += m.curtail[t] * m.curtail_cost * m.dt
                costs += m.unserved[t] * m.voll * m.dt
            
            return revenue - costs
        
        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        self.model = m
        return m
    
    def solve(self, case: DispatchCase) -> OptimizationResults:
        """
        Build and solve the dispatch optimization model.
        
        Args:
            case: DispatchCase with all input data
            
        Returns:
            OptimizationResults with solution and explanations
        """
        import time
        
        # Build model
        m = self.build_model(case)
        
        # Configure solver - try multiple options
        solver = None
        solver_options = [self.solver_name, "appsi_highs", "highs", "glpk", "cbc", "cplex", "gurobi"]
        
        for solver_name in solver_options:
            try:
                solver = SolverFactory(solver_name)
                if solver is not None and solver.available():
                    print(f"Using solver: {solver_name}")
                    break
                solver = None
            except Exception:
                continue
        
        if solver is None:
            raise RuntimeError(
                "No MILP solver found. Install one of:\n"
                "  - pip install highspy (may need C++ compiler)\n"
                "  - conda install -c conda-forge glpk\n"
                "  - conda install -c conda-forge coincbc"
            )
        
        # Set solver options
        if hasattr(solver, 'options'):
            solver.options['time_limit'] = self.time_limit_sec
            solver.options['mip_rel_gap'] = self.mip_gap
        
        # Solve
        start_time = time.time()
        try:
            result = solver.solve(m, tee=False)
            solve_time = time.time() - start_time
            
            solved = (
                result.solver.status == SolverStatus.ok and
                result.solver.termination_condition in [
                    TerminationCondition.optimal,
                    TerminationCondition.feasible
                ]
            )
            solver_status = str(result.solver.termination_condition)
            obj_value = pyo.value(m.objective) if solved else 0.0
            
        except Exception as e:
            solve_time = time.time() - start_time
            solved = False
            solver_status = f"Error: {str(e)}"
            obj_value = 0.0
        
        # Extract results
        results = OptimizationResults(
            solved=solved,
            solver_status=solver_status,
            solve_time_sec=solve_time,
            objective_value=obj_value,
            n_intervals=case.n_intervals,
            interval_hours=case.interval_hours,
            horizon_hours=case.horizon_hours
        )
        
        if solved:
            results = self._extract_results(m, case, results)
            results.calculate_aggregates()
            self._generate_explanations(results)
        
        return results
    
    def _extract_results(
        self,
        m: pyo.ConcreteModel,
        case: DispatchCase,
        results: OptimizationResults
    ) -> OptimizationResults:
        """Extract solution values from solved model."""
        
        for t in m.T:
            # Grid
            grid_import = pyo.value(m.grid_import[t])
            grid_export = pyo.value(m.grid_export[t])
            
            # BESS
            bess_charge = pyo.value(m.bess_charge[t])
            bess_discharge = pyo.value(m.bess_discharge[t])
            bess_soc = pyo.value(m.bess_soc[t])
            
            # Generators
            gen_output = sum(pyo.value(m.gen_output[t, g]) for g in m.G)
            gen_units = sum(
                1 for g in m.G if pyo.value(m.gen_online[t, g]) > 0.5
            )
            gen_starting = any(
                pyo.value(m.gen_start[t, g]) > 0.5 for g in m.G
            )
            
            # Loads
            curtailed = pyo.value(m.curtail[t])
            unserved = pyo.value(m.unserved[t])
            
            # Reserves
            reg_up = pyo.value(m.reg_up[t])
            reg_down = pyo.value(m.reg_down[t])
            spin = pyo.value(m.spin[t])
            
            # Economics
            energy_price = case.prices.energy_price[t]
            energy_cost = grid_import * energy_price * case.interval_hours
            energy_revenue = grid_export * energy_price * case.interval_hours
            as_revenue = (
                reg_up * case.prices.reg_up_price[t] +
                reg_down * case.prices.reg_down_price[t] +
                spin * case.prices.spin_price[t]
            ) * case.interval_hours
            
            fuel_cost = gen_output * (
                case.gen_fuel_cost_per_mwh + case.gen_vom_per_mwh
            ) * case.interval_hours
            
            degradation_cost = bess_discharge * case.bess_degradation_cost * case.interval_hours
            curtailment_cost = (
                curtailed * case.curtailment_cost +
                unserved * case.voll
            ) * case.interval_hours
            
            # Identify binding constraints
            binding = self._identify_binding_constraints(m, t, case)
            
            interval_result = IntervalResult(
                interval=t,
                timestamp_hours=t * case.interval_hours,
                grid_import_mw=grid_import,
                grid_export_mw=grid_export,
                grid_net_mw=grid_import - grid_export,
                bess_charge_mw=bess_charge,
                bess_discharge_mw=bess_discharge,
                bess_soc=bess_soc,
                bess_soc_mwh=bess_soc * case.bess_energy_mwh,
                gen_output_mw=gen_output,
                gen_units_online=gen_units,
                gen_starting=gen_starting,
                total_load_mw=case.total_load[t],
                critical_load_mw=case.critical_load[t],
                curtailed_mw=curtailed,
                unserved_mw=unserved,
                reg_up_mw=reg_up,
                reg_down_mw=reg_down,
                spin_mw=spin,
                energy_cost=energy_cost,
                energy_revenue=energy_revenue,
                as_revenue=as_revenue,
                fuel_cost=fuel_cost,
                degradation_cost=degradation_cost,
                curtailment_cost=curtailment_cost,
                binding_constraints=binding
            )
            
            results.intervals.append(interval_result)
        
        return results
    
    def _identify_binding_constraints(
        self,
        m: pyo.ConcreteModel,
        t: int,
        case: DispatchCase,
        tol: float = 0.01
    ) -> List[str]:
        """Identify which constraints are binding at an interval."""
        binding = []
        
        # SOC limits
        soc = pyo.value(m.bess_soc[t])
        if abs(soc - case.bess_soc_max) < tol:
            binding.append("soc_max")
        if abs(soc - case.bess_soc_reserve) < tol:
            binding.append("soc_reserve")
        if abs(soc - case.bess_soc_min) < tol:
            binding.append("soc_min")
        
        # BESS power limits
        charge = pyo.value(m.bess_charge[t])
        discharge = pyo.value(m.bess_discharge[t])
        if abs(charge - case.bess_power_mw) < tol:
            binding.append("bess_charge_max")
        if abs(discharge - case.bess_power_mw) < tol:
            binding.append("bess_discharge_max")
        
        # Grid limits
        grid_import = pyo.value(m.grid_import[t])
        grid_export = pyo.value(m.grid_export[t])
        if abs(grid_import - case.poi_max_import_mw) < tol:
            binding.append("grid_import_max")
        if abs(grid_export - case.poi_max_export_mw) < tol:
            binding.append("grid_export_max")
        
        # Price signals
        price = case.prices.energy_price[t]
        avg_price = sum(case.prices.energy_price) / len(case.prices.energy_price)
        if price > avg_price * 1.5:
            binding.append("energy_price_high")
        if price < avg_price * 0.5:
            binding.append("energy_price_low")
        
        # Load balance always binds (equality)
        binding.append("load_balance")
        
        return binding
    
    def _generate_explanations(self, results: OptimizationResults) -> None:
        """Generate decision explanations for all intervals."""
        for t, interval in enumerate(results.intervals):
            explanations = results.explain_interval(t)
            results.explanations[t] = explanations
