import pyomo.environ as pyo
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from resources.bess import BESS
from resources.gas_gen import GasGen
from resources.load import LoadProfile
from topology.network import Topology
from iso.base import IsoAdapter

class DispatchOptimizer:
    def __init__(
        self, 
        iso: IsoAdapter, 
        topology: Topology, 
        bess: BESS, 
        gen: GasGen, 
        load: LoadProfile,
        market_prices: pd.DataFrame, 
        horizon_hours: int = 24
    ):
        self.iso = iso
        self.topology = topology
        self.bess = bess
        self.gen = gen
        self.load = load
        self.market_prices = market_prices
        self.horizon_hours = horizon_hours
        self.dt = iso.dispatch_interval_minutes / 60.0 # hours
        self.steps = int(horizon_hours * 60 / iso.dispatch_interval_minutes)
        self.model = None

    def build_model(self):
        m = pyo.ConcreteModel()
        
        # Sets
        m.T = pyo.RangeSet(0, self.steps - 1)
        
        # --- Variables ---
        
        # Grid interaction
        m.grid_import = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.grid_export = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.grid_is_exporting = pyo.Var(m.T, domain=pyo.Binary) 
        
        # Generator
        m.gen_units_on = pyo.Var(m.T, domain=pyo.Integers, bounds=(0, self.gen.num_units))
        m.gen_mw = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.gen_start = pyo.Var(m.T, domain=pyo.Integers, bounds=(0, self.gen.num_units)) 
        
        # BESS
        m.bess_charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.bess_discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.bess_soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, self.bess.capacity_mwh))
        m.bess_is_charging = pyo.Var(m.T, domain=pyo.Binary)
        
        # Load
        m.load_served = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.load_unserved = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        
        # Ancillary Services
        self.as_products = self.iso.reserve_products
        self.as_vars = {}
        for prod in self.as_products:
            m.add_component(f"as_{prod.name}_bess", pyo.Var(m.T, domain=pyo.NonNegativeReals))
            m.add_component(f"as_{prod.name}_gen", pyo.Var(m.T, domain=pyo.NonNegativeReals))
            self.as_vars[prod.name] = {
                'bess': getattr(m, f"as_{prod.name}_bess"),
                'gen': getattr(m, f"as_{prod.name}_gen")
            }

        # --- Constraints ---
        
        # 1. Power Balance
        def power_balance_rule(m, t):
            return (m.grid_import[t] + m.gen_mw[t] + m.bess_discharge[t] == 
                    m.load_served[t] + m.bess_charge[t] + m.grid_export[t])
        m.PowerBalance = pyo.Constraint(m.T, rule=power_balance_rule)
        
        # 2. Load
        def load_rule(m, t):
            forecast = self.load.forecast_mw[t] if t < len(self.load.forecast_mw) else self.load.forecast_mw[-1]
            return m.load_served[t] + m.load_unserved[t] == forecast
        m.LoadSatisfied = pyo.Constraint(m.T, rule=load_rule)

        # 3. POI Limits
        poi_export_limit = min(self.topology.poi.export_limit_mw, self.topology.poi.export_limit_mva)
        poi_import_limit = min(self.topology.poi.import_limit_mw, self.topology.poi.import_limit_mva)
        
        def poi_import_rule(m, t):
            return m.grid_import[t] <= poi_import_limit * (1 - m.grid_is_exporting[t])
        m.POIImportLimit = pyo.Constraint(m.T, rule=poi_import_rule)
        
        def poi_export_rule(m, t):
            return m.grid_export[t] <= poi_export_limit * m.grid_is_exporting[t]
        m.POIExportLimit = pyo.Constraint(m.T, rule=poi_export_rule)
        
        # 4. Generator Limits
        def gen_capacity_rule(m, t):
            return m.gen_mw[t] <= m.gen_units_on[t] * self.gen.unit_capacity_mw
        m.GenMaxCap = pyo.Constraint(m.T, rule=gen_capacity_rule)
        
        def gen_min_stable_rule(m, t):
            return m.gen_mw[t] >= m.gen_units_on[t] * self.gen.min_stable_level_mw
        m.GenMinStable = pyo.Constraint(m.T, rule=gen_min_stable_rule)
        
        def gen_start_rule(m, t):
            if t == 0:
                # Assume initially OFF.
                return m.gen_start[t] >= m.gen_units_on[t]
            return m.gen_start[t] >= m.gen_units_on[t] - m.gen_units_on[t-1]
        m.GenStartLogic = pyo.Constraint(m.T, rule=gen_start_rule)
        
        # 5. BESS Dynamics
        def soc_rule(m, t):
            if t == 0:
                soc_prev = self.bess.soc_initial * self.bess.capacity_mwh
            else:
                soc_prev = m.bess_soc[t-1]
            
            return m.bess_soc[t] == soc_prev + (m.bess_charge[t] * self.bess.charge_efficiency - m.bess_discharge[t] / self.bess.discharge_efficiency) * self.dt
        m.SOCDynamics = pyo.Constraint(m.T, rule=soc_rule)
        
        def bess_charge_limit_rule(m, t):
            return m.bess_charge[t] <= self.bess.pcs_mva * m.bess_is_charging[t]
        m.BESSChargeLimit = pyo.Constraint(m.T, rule=bess_charge_limit_rule)
        
        def bess_discharge_limit_rule(m, t):
            return m.bess_discharge[t] <= self.bess.pcs_mva * (1 - m.bess_is_charging[t])
        m.BESSDischargeLimit = pyo.Constraint(m.T, rule=bess_discharge_limit_rule)
        
        def soc_min_rule(m, t):
            return m.bess_soc[t] >= self.bess.soc_min_reserve * self.bess.capacity_mwh
        m.SOCMin = pyo.Constraint(m.T, rule=soc_min_rule)

        # 6. Ancillary Services
        def gen_headroom_rule(m, t):
            total_gen_as_up = sum(self.as_vars[p.name]['gen'][t] for p in self.as_products if p.direction == 'up')
            return m.gen_mw[t] + total_gen_as_up <= m.gen_units_on[t] * self.gen.unit_capacity_mw
        m.GenHeadroom = pyo.Constraint(m.T, rule=gen_headroom_rule)
        
        def gen_footroom_rule(m, t):
            total_gen_as_down = sum(self.as_vars[p.name]['gen'][t] for p in self.as_products if p.direction == 'down')
            return m.gen_mw[t] - total_gen_as_down >= m.gen_units_on[t] * self.gen.min_stable_level_mw
        m.GenFootroom = pyo.Constraint(m.T, rule=gen_footroom_rule)
        
        def bess_headroom_discharge_rule(m, t):
            total_bess_as_up = sum(self.as_vars[p.name]['bess'][t] for p in self.as_products if p.direction == 'up')
            # Conservative: Headroom on discharge side.
            return m.bess_discharge[t] + total_bess_as_up <= self.bess.pcs_mva
        m.BESSHeadroomDischarge = pyo.Constraint(m.T, rule=bess_headroom_discharge_rule)
        
        def bess_headroom_charge_rule(m, t):
             total_bess_as_down = sum(self.as_vars[p.name]['bess'][t] for p in self.as_products if p.direction == 'down')
             # Reg Down = Ability to Charge More. 
             # Charge + RegDown <= PCS
             return m.bess_charge[t] + total_bess_as_down <= self.bess.pcs_mva
        m.BESSHeadroomCharge = pyo.Constraint(m.T, rule=bess_headroom_charge_rule)
        
        def bess_as_energy_backing(m, t):
            required_energy = sum(
                self.as_vars[p.name]['bess'][t] * (p.duration_minutes / 60.0) 
                for p in self.as_products if p.direction == 'up'
            )
            available_soc = m.bess_soc[t] - (self.bess.soc_min_reserve * self.bess.capacity_mwh)
            return available_soc >= required_energy
        m.BESSASEnergyBacking = pyo.Constraint(m.T, rule=bess_as_energy_backing)

        # --- Objective ---
        def obj_rule(m):
            revenue = 0
            cost = 0
            
            for t in m.T:
                import_price = self.market_prices.iloc[t].get('energy_import', 0)
                export_price = self.market_prices.iloc[t].get('energy_export', 0)
                fuel_price = self.gen.fuel_cost_per_mmbtu
                
                revenue += m.grid_export[t] * export_price * self.dt
                cost += m.grid_import[t] * import_price * self.dt
                
                for p in self.as_products:
                    price = self.market_prices.iloc[t].get(f"as_{p.name}", 0)
                    qty = self.as_vars[p.name]['bess'][t] + self.as_vars[p.name]['gen'][t]
                    revenue += qty * price * self.dt
                
                fuel_consumption = m.gen_mw[t] * self.gen.heat_rate_mmbtu_per_mwh * self.dt
                cost += fuel_consumption * fuel_price
                cost += m.gen_mw[t] * self.gen.om_cost_per_mwh * self.dt
                cost += m.gen_start[t] * self.gen.start_cost_dollars
                
                deg_cost = 10.0 
                cost += (m.bess_charge[t] + m.bess_discharge[t]) * deg_cost * self.dt
                
                cost += m.load_unserved[t] * self.load.value_of_lost_load * self.dt

            return revenue - cost
        
        m.Objective = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
        self.model = m

    def solve(self):
        # Use glpk
        solver = pyo.SolverFactory('glpk')
        try:
            result = solver.solve(self.model, tee=False)
            return result
        except Exception as e:
            print(f"Solver Error: {e}")
            class DummyResult:
                class Solver:
                    termination_condition = 'error'
                solver = Solver()
            return DummyResult()

    def get_results(self):
        m = self.model
        data = []
        for t in m.T:
            row = {
                't': t,
                'grid_import': pyo.value(m.grid_import[t]),
                'grid_export': pyo.value(m.grid_export[t]),
                'gen_mw': pyo.value(m.gen_mw[t]),
                'gen_units_on': pyo.value(m.gen_units_on[t]),
                'bess_charge': pyo.value(m.bess_charge[t]),
                'bess_discharge': pyo.value(m.bess_discharge[t]),
                'bess_soc': pyo.value(m.bess_soc[t]),
                'load_served': pyo.value(m.load_served[t]),
                'load_unserved': pyo.value(m.load_unserved[t]),
            }
            for p in self.as_products:
                row[f"as_{p.name}_bess"] = pyo.value(self.as_vars[p.name]['bess'][t])
                row[f"as_{p.name}_gen"] = pyo.value(self.as_vars[p.name]['gen'][t])
            
            data.append(row)
        return pd.DataFrame(data)
