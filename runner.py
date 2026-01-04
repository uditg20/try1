#!/usr/bin/env python3
"""
Data Center Energy Platform - Runner
=====================================

Main entry point for running dispatch optimization cases.
Accepts a JSON case file, builds the topology and dispatch model,
solves, and returns structured results for UI visualization.

Usage:
    python runner.py examples/case_ercot_50mw.json
    python runner.py examples/case_ercot_50mw.json --output results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iso.generic import get_iso_adapter
from iso.base import ISOAdapter
from topology.poi import PointOfInterconnection
from topology.transformer import Transformer
from topology.bus import Bus, BusType, PCS
from topology.network import MicrogridTopology
from resources.bess import BESS
from resources.gas_gen import GasGenerator, GasGeneratorUnit
from resources.load import Load, LoadType, LoadProfile
from optimization.dispatch_milp import DispatchOptimizer, DispatchCase, PriceData


def load_case_file(filepath: str) -> dict:
    """Load and validate a JSON case file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {filepath}")
    
    with open(path, 'r') as f:
        case_data = json.load(f)
    
    # Validate required fields
    required = ["name", "iso", "time", "topology", "resources"]
    for field in required:
        if field not in case_data:
            raise ValueError(f"Missing required field: {field}")
    
    return case_data


def build_iso_adapter(case_data: dict) -> ISOAdapter:
    """Create the appropriate ISO adapter."""
    iso_name = case_data.get("iso", "GENERIC")
    return get_iso_adapter(iso_name)


def build_topology(case_data: dict) -> MicrogridTopology:
    """Build the electrical topology from case data."""
    topo_data = case_data["topology"]
    
    # POI
    poi_data = topo_data["poi"]
    poi = PointOfInterconnection(
        name=poi_data.get("name", "POI"),
        max_import_mw=poi_data["max_import_mw"],
        max_export_mw=poi_data["max_export_mw"],
        mva_rating=poi_data["mva_rating"],
        min_power_factor=poi_data.get("min_power_factor", 0.95),
        voltage_kv=poi_data.get("voltage_kv", 138.0)
    )
    
    # Transformers
    transformers = []
    for xfmr_data in topo_data.get("transformers", []):
        xfmr = Transformer(
            name=xfmr_data["name"],
            mva_rating=xfmr_data["mva_rating"],
            primary_kv=xfmr_data.get("primary_kv", 138.0),
            secondary_kv=xfmr_data.get("secondary_kv", 13.8),
            redundancy_config=xfmr_data.get("redundancy", "N+1")
        )
        transformers.append(xfmr)
    
    # Standard buses
    buses = [
        Bus(name="MAIN", bus_type=BusType.MAIN, voltage_kv=13.8),
        Bus(name="CRITICAL", bus_type=BusType.CRITICAL, voltage_kv=13.8),
        Bus(name="GENERATOR", bus_type=BusType.GENERATOR, voltage_kv=13.8),
        Bus(name="BESS", bus_type=BusType.BESS, voltage_kv=13.8),
    ]
    
    topology = MicrogridTopology(
        name=case_data["name"],
        poi=poi,
        transformers=transformers,
        buses=buses
    )
    
    # Add PCS for BESS
    bess_data = case_data["resources"].get("bess", {})
    if bess_data:
        pcs = PCS(
            name=f"{bess_data.get('name', 'BESS')}_PCS",
            mva_rating=bess_data.get("pcs_mva", bess_data.get("power_mw", 10)),
            max_p_mw=bess_data.get("power_mw", 10),
            efficiency=0.96
        )
        topology.add_pcs(bess_data.get("name", "BESS"), pcs)
    
    return topology


def build_resources(case_data: dict) -> tuple:
    """Build resource models from case data."""
    resources = case_data["resources"]
    
    # BESS
    bess_data = resources.get("bess", {})
    bess = None
    if bess_data:
        bess = BESS(
            name=bess_data.get("name", "BESS"),
            power_mw=bess_data["power_mw"],
            energy_mwh=bess_data["energy_mwh"],
            pcs_mva=bess_data.get("pcs_mva", bess_data["power_mw"]),
            efficiency_charge=bess_data.get("efficiency_charge", 0.94),
            efficiency_discharge=bess_data.get("efficiency_discharge", 0.94),
            soc_min=bess_data.get("soc_min", 0.10),
            soc_max=bess_data.get("soc_max", 0.90),
            soc_reserve=bess_data.get("soc_reserve", 0.20),
            initial_soc=bess_data.get("initial_soc", 0.50),
            degradation_cost_per_mwh=bess_data.get("degradation_cost_per_mwh", 5.0)
        )
    
    # Generators
    gen_data = resources.get("generators", {})
    generator = None
    if gen_data:
        generator = GasGenerator(
            name=gen_data.get("name", "GenFleet"),
            redundancy_config=gen_data.get("redundancy", "N+1")
        )
        for unit_data in gen_data.get("units", []):
            unit = GasGeneratorUnit(
                name=unit_data["name"],
                capacity_mw=unit_data["capacity_mw"],
                min_output_mw=unit_data.get("min_output_mw", 0),
                start_time_min=unit_data.get("start_time_min", 10),
                heat_rate_mmbtu_per_mwh=unit_data.get("heat_rate_mmbtu_per_mwh", 8.5),
                fuel_cost_per_mmbtu=unit_data.get("fuel_cost_per_mmbtu", 3.0),
                vom_per_mwh=unit_data.get("vom_per_mwh", 5.0),
                start_cost=unit_data.get("start_cost", 500)
            )
            generator.add_unit(unit)
    
    # Loads
    loads = []
    for load_data in resources.get("loads", []):
        load_type_str = load_data.get("type", "non_critical").upper()
        load_type = getattr(LoadType, load_type_str, LoadType.NON_CRITICAL)
        
        load = Load(
            name=load_data["name"],
            load_type=load_type,
            base_mw=load_data["base_mw"],
            peak_mw=load_data.get("peak_mw", load_data["base_mw"]),
            power_factor=load_data.get("power_factor", 0.95),
            curtailable_fraction=load_data.get("curtailable_fraction", 0.0)
        )
        loads.append(load)
    
    return bess, generator, loads


def generate_price_data(
    case_data: dict,
    n_intervals: int,
    interval_minutes: int
) -> PriceData:
    """Generate price timeseries from case specification."""
    price_config = case_data.get("prices", {})
    
    # Defaults
    base_price = price_config.get("base_price", 35)
    peak_adder = price_config.get("peak_adder", 50)
    peak_hours = price_config.get("peak_hours", [7, 8, 9, 17, 18, 19, 20])
    super_peak_adder = price_config.get("super_peak_adder", 150)
    super_peak_hours = price_config.get("super_peak_hours", [18, 19])
    night_discount = price_config.get("night_discount", 10)
    night_hours = price_config.get("night_hours", [0, 1, 2, 3, 4, 5, 22, 23])
    
    as_config = price_config.get("as_prices", {})
    reg_up_base = as_config.get("reg_up_base", 12)
    reg_down_base = as_config.get("reg_down_base", 6)
    spin_base = as_config.get("spin_base", 8)
    as_peak_mult = as_config.get("peak_multiplier", 1.5)
    
    energy_prices = []
    reg_up_prices = []
    reg_down_prices = []
    spin_prices = []
    
    intervals_per_hour = 60 // interval_minutes
    
    for t in range(n_intervals):
        hour = (t // intervals_per_hour) % 24
        
        # Energy price
        price = base_price
        if hour in night_hours:
            price -= night_discount
        if hour in peak_hours:
            price += peak_adder
        if hour in super_peak_hours:
            price += super_peak_adder
        
        # Add some randomness
        noise = np.random.normal(0, 5)
        price = max(0, price + noise)
        energy_prices.append(price)
        
        # AS prices
        if hour in peak_hours:
            reg_up_prices.append(reg_up_base * as_peak_mult)
            reg_down_prices.append(reg_down_base * as_peak_mult)
            spin_prices.append(spin_base * as_peak_mult)
        else:
            reg_up_prices.append(reg_up_base)
            reg_down_prices.append(reg_down_base)
            spin_prices.append(spin_base)
    
    return PriceData(
        energy_price=energy_prices,
        reg_up_price=reg_up_prices,
        reg_down_price=reg_down_prices,
        spin_price=spin_prices
    )


def generate_load_profile(
    loads: List[Load],
    n_intervals: int,
    interval_minutes: int
) -> tuple:
    """Generate load timeseries."""
    intervals_per_hour = 60 // interval_minutes
    
    total_load = []
    critical_load = []
    curtailable_load = []
    
    for t in range(n_intervals):
        hour = (t // intervals_per_hour) % 24
        
        # Diurnal variation factor
        variation = 1.0 + 0.1 * math.sin(2 * math.pi * hour / 24 - math.pi / 2)
        
        total = 0
        critical = 0
        curtailable = 0
        
        for load in loads:
            load_mw = load.base_mw * variation
            total += load_mw
            
            if load.load_type == LoadType.CRITICAL:
                critical += load_mw
            elif load.load_type == LoadType.CURTAILABLE:
                curtailable += load_mw * load.curtailable_fraction
        
        total_load.append(total)
        critical_load.append(critical)
        curtailable_load.append(curtailable)
    
    return total_load, critical_load, curtailable_load


def build_dispatch_case(
    case_data: dict,
    topology: MicrogridTopology,
    bess: Optional[BESS],
    generator: Optional[GasGenerator],
    loads: List[Load],
    iso: ISOAdapter
) -> DispatchCase:
    """Build the dispatch case for optimization."""
    time_config = case_data["time"]
    horizon_hours = time_config["horizon_hours"]
    interval_minutes = time_config.get("interval_minutes", 15)
    interval_hours = interval_minutes / 60.0
    n_intervals = int(horizon_hours / interval_hours)
    
    # Generate price data
    prices = generate_price_data(case_data, n_intervals, interval_minutes)
    
    # Generate load profiles
    total_load, critical_load, curtailable_load = generate_load_profile(
        loads, n_intervals, interval_minutes
    )
    
    # Generator parameters
    if generator and generator.units:
        gen_capacity = generator.total_capacity_mw
        gen_min = sum(u.min_output_mw for u in generator.units)
        first_unit = generator.units[0]
        gen_fuel_cost = first_unit.heat_rate_mmbtu_per_mwh * first_unit.fuel_cost_per_mmbtu
        gen_vom = first_unit.vom_per_mwh
        gen_start_cost = first_unit.start_cost
        gen_start_intervals = max(1, int(first_unit.start_time_min / interval_minutes))
        n_gen_units = len(generator.units)
    else:
        gen_capacity = 0
        gen_min = 0
        gen_fuel_cost = 30.0
        gen_vom = 5.0
        gen_start_cost = 500
        gen_start_intervals = 1
        n_gen_units = 0
    
    # BESS parameters
    if bess:
        bess_power = bess.power_mw
        bess_energy = bess.energy_mwh
        bess_pcs = bess.pcs_mva
        bess_eff_c = bess.efficiency_charge
        bess_eff_d = bess.efficiency_discharge
        bess_soc_min = bess.soc_min
        bess_soc_max = bess.soc_max
        bess_soc_reserve = bess.soc_reserve
        bess_initial = bess.initial_soc
        bess_degrad = bess.degradation_cost_per_mwh
    else:
        bess_power = 0
        bess_energy = 0
        bess_pcs = 0
        bess_eff_c = 0.94
        bess_eff_d = 0.94
        bess_soc_min = 0.1
        bess_soc_max = 0.9
        bess_soc_reserve = 0.2
        bess_initial = 0.5
        bess_degrad = 5.0
    
    # Reliability parameters
    reliability = case_data.get("reliability", {})
    voll = reliability.get("voll_per_mwh", 50000)
    curtail_cost = reliability.get("curtailment_cost_per_mwh", 500)
    ride_through = reliability.get("ride_through_minutes", 15)
    
    # Optimization parameters
    opt_config = case_data.get("optimization", {})
    as_backing = opt_config.get("as_energy_backing_hours", 1.0)
    
    return DispatchCase(
        n_intervals=n_intervals,
        interval_hours=interval_hours,
        total_load=total_load,
        critical_load=critical_load,
        curtailable_load=curtailable_load,
        prices=prices,
        bess_power_mw=bess_power,
        bess_energy_mwh=bess_energy,
        bess_pcs_mva=bess_pcs,
        bess_eff_charge=bess_eff_c,
        bess_eff_discharge=bess_eff_d,
        bess_soc_min=bess_soc_min,
        bess_soc_max=bess_soc_max,
        bess_soc_reserve=bess_soc_reserve,
        bess_initial_soc=bess_initial,
        bess_degradation_cost=bess_degrad,
        gen_capacity_mw=gen_capacity,
        gen_min_output_mw=gen_min,
        gen_fuel_cost_per_mwh=gen_fuel_cost,
        gen_vom_per_mwh=gen_vom,
        gen_start_cost=gen_start_cost,
        gen_start_intervals=gen_start_intervals,
        gen_min_run_intervals=4,  # 1 hour at 15-min intervals
        n_gen_units=n_gen_units,
        poi_max_import_mw=topology.poi.max_import_mw,
        poi_max_export_mw=topology.poi.max_export_mw,
        poi_mva_rating=topology.poi.mva_rating,
        poi_min_pf=topology.poi.min_power_factor,
        transformer_mva=topology.total_transformer_mva,
        ride_through_min=ride_through,
        voll=voll,
        curtailment_cost=curtail_cost,
        as_energy_backing_hours=as_backing
    )


def run_case(case_file: str, output_file: Optional[str] = None) -> dict:
    """
    Run a complete dispatch optimization case.
    
    Args:
        case_file: Path to JSON case file
        output_file: Optional path for JSON output
        
    Returns:
        Dict with complete results
    """
    print(f"Loading case: {case_file}")
    case_data = load_case_file(case_file)
    
    print(f"Case: {case_data['name']}")
    print(f"ISO: {case_data['iso']}")
    
    # Build components
    print("Building ISO adapter...")
    iso = build_iso_adapter(case_data)
    
    print("Building topology...")
    topology = build_topology(case_data)
    
    print("Building resources...")
    bess, generator, loads = build_resources(case_data)
    
    print("Building dispatch case...")
    dispatch_case = build_dispatch_case(
        case_data, topology, bess, generator, loads, iso
    )
    
    print(f"Optimization horizon: {dispatch_case.horizon_hours} hours")
    print(f"Intervals: {dispatch_case.n_intervals}")
    print(f"BESS: {dispatch_case.bess_power_mw} MW / {dispatch_case.bess_energy_mwh} MWh")
    print(f"Generators: {dispatch_case.n_gen_units} units, {dispatch_case.gen_capacity_mw} MW total")
    
    # Create optimizer
    opt_config = case_data.get("optimization", {})
    optimizer = DispatchOptimizer(
        solver_name=opt_config.get("solver", "appsi_highs"),
        time_limit_sec=opt_config.get("time_limit_sec", 60),
        mip_gap=opt_config.get("mip_gap", 0.001)
    )
    
    # Solve
    print("\nSolving optimization...")
    results = optimizer.solve(dispatch_case)
    
    print(f"\nSolver status: {results.solver_status}")
    print(f"Solve time: {results.solve_time_sec:.2f} seconds")
    
    if results.solved:
        print(f"\nObjective (Net Value): ${results.objective_value:,.2f}")
        print("\nEconomics Summary:")
        econ = results.get_economics_summary()
        print(f"  Energy Revenue: ${econ['revenue']['energy_export']:,.2f}")
        print(f"  AS Revenue: ${econ['revenue']['ancillary_services']:,.2f}")
        print(f"  Energy Cost: ${econ['costs']['energy_import']:,.2f}")
        print(f"  Fuel Cost: ${econ['costs']['fuel']:,.2f}")
        print(f"  Degradation: ${econ['costs']['degradation']:,.2f}")
        print(f"  Net Value: ${econ['net_value']:,.2f}")
        
        reliability = results.get_reliability_summary()
        print("\nReliability Summary:")
        print(f"  Unserved Energy: {reliability['unserved_mwh']:.3f} MWh")
        print(f"  Ride-through Achieved: {reliability['ride_through_achieved']}")
    else:
        print("Optimization failed!")
    
    # Build complete output
    output = {
        "case": {
            "name": case_data["name"],
            "iso": case_data["iso"],
            "file": case_file
        },
        "topology": topology.get_topology_summary(),
        "resources": {
            "bess": bess.get_parameters() if bess else None,
            "generators": generator.get_parameters() if generator else None,
            "loads": [
                {"name": l.name, "type": l.load_type.value, "base_mw": l.base_mw}
                for l in loads
            ]
        },
        "dispatch_case": {
            "n_intervals": dispatch_case.n_intervals,
            "interval_hours": dispatch_case.interval_hours,
            "horizon_hours": dispatch_case.horizon_hours,
            "prices": {
                "energy": dispatch_case.prices.energy_price,
                "reg_up": dispatch_case.prices.reg_up_price,
                "reg_down": dispatch_case.prices.reg_down_price,
                "spin": dispatch_case.prices.spin_price
            },
            "loads": {
                "total": dispatch_case.total_load,
                "critical": dispatch_case.critical_load,
                "curtailable": dispatch_case.curtailable_load
            }
        },
        "results": results.to_dict()
    }
    
    # Save output if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Center Energy Platform - Dispatch Optimizer"
    )
    parser.add_argument(
        "case_file",
        help="Path to JSON case file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path for JSON output file"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit UI after solving"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    results = run_case(args.case_file, args.output)
    
    if args.ui:
        print("\nLaunching UI...")
        import subprocess
        subprocess.run(["streamlit", "run", "src/ui/app.py", "--", args.case_file])
    
    return results


if __name__ == "__main__":
    main()
