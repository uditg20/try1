import json
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from iso.ercot import ErcotAdapter
from resources.bess import BESS
from resources.gas_gen import GasGen
from resources.load import LoadProfile
from topology.network import Topology, POI, Transformer
from optimization.dispatch_milp import DispatchOptimizer

class Runner:
    def __init__(self, case_file: str):
        with open(case_file, 'r') as f:
            self.data = json.load(f)
        
        self.iso_name = self.data.get("iso", "ERCOT")
        if self.iso_name == "ERCOT":
            self.iso = ErcotAdapter()
        else:
            raise ValueError(f"Unknown ISO: {self.iso_name}")
            
        self.bess = BESS(**self.data['bess'])
        self.gen = GasGen(**self.data['gen'])
        self.topology = Topology(**self.data['topology'])
        
        # Load Construction (Simple synthetic profile)
        horizon = self.data['horizon_hours']
        steps = int(horizon * 60 / self.iso.dispatch_interval_minutes)
        
        # Create synthetic load profile
        base = self.data['load']['base_load_mw']
        peak = self.data['load']['peak_load_mw']
        volatility = self.data['load']['volatility']
        
        # Sine wave + noise
        t = np.linspace(0, 24, steps)
        load_curve = base + (peak - base) * np.sin(np.pi * t / 12)**2
        noise = np.random.normal(0, volatility * base, steps)
        load_curve = np.maximum(0, load_curve + noise).tolist()
        
        self.load = LoadProfile(
            name=self.data['load']['name'],
            is_critical=self.data['load']['is_critical'],
            forecast_mw=load_curve
        )
        
        # Market Data Construction
        # Create synthetic prices
        base_price = self.data['market_data']['base_energy_price']
        peak_price = self.data['market_data']['peak_energy_price']
        
        prices = []
        for i in range(steps):
            hour = (i * self.iso.dispatch_interval_minutes) / 60.0
            price = base_price
            if 14 <= hour <= 19: # Peak hours
                price = peak_price
            
            # Add some volatility
            price += np.random.normal(0, 5)
            
            row = {
                'energy_import': price,
                'energy_export': price * 0.9, # Sell for slightly less
            }
            # AS Prices
            row['as_RegUp'] = self.data['market_data'].get('as_reg_up_price', 5.0)
            row['as_RegDown'] = 2.0
            row['as_RRS'] = self.data['market_data'].get('as_rrs_price', 4.0)
            row['as_NonSpin'] = 1.0
            
            prices.append(row)
            
        self.market_prices = pd.DataFrame(prices)
        
    def run(self):
        optimizer = DispatchOptimizer(
            iso=self.iso,
            topology=self.topology,
            bess=self.bess,
            gen=self.gen,
            load=self.load,
            market_prices=self.market_prices,
            horizon_hours=self.data['horizon_hours']
        )
        
        optimizer.build_model()
        status = optimizer.solve()
        
        print(f"Solver Status: {status.solver.termination_condition}")
        
        if str(status.solver.termination_condition) != 'optimal':
             # Return early with empty results
             return {
                "status": str(status.solver.termination_condition),
                "results": pd.DataFrame(),
                "inputs": {
                    "bess": self.bess,
                    "gen": self.gen,
                    "topology": self.topology,
                    "load": self.load,
                    "iso": self.iso.name
                }
             }

        results = optimizer.get_results()
        
        return {
            "status": str(status.solver.termination_condition),
            "results": results,
            "inputs": {
                "bess": self.bess,
                "gen": self.gen,
                "topology": self.topology,
                "load": self.load,
                "iso": self.iso.name
            }
        }

if __name__ == "__main__":
    runner = Runner("example_case.json")
    out = runner.run()
    print("Optimization Complete")
    if not out['results'].empty:
        print(out['results'].head())
