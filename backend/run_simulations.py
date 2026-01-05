"""
Time-Series Power Flow Simulation
=================================

Runs pandapower power flow for each time interval in a scenario.

This module:
1. Takes scenario data (load/BESS profiles)
2. Creates pandapower network
3. Runs AC power flow for each interval
4. Collects results (voltages, loadings, grid exchange)
5. Checks for constraint violations

IMPORTANT LIMITATIONS:
- This is PLANNING-LEVEL, POSITIVE-SEQUENCE analysis ONLY
- No transient/EMT simulation
- No inverter control dynamics
- No protection coordination
- Results are for PLANNING purposes, not operational use
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time as pytime

from network import (
    create_datacenter_network,
    set_load_dispatch,
    set_bess_dispatch,
    run_powerflow,
    check_constraints,
    NetworkConfig
)
from scenarios import ScenarioConfig, generate_scenario, BESSConfig


@dataclass
class SimulationConfig:
    """
    Configuration for time-series simulation.
    
    Attributes:
        voltage_min_pu: Minimum voltage limit
        voltage_max_pu: Maximum voltage limit
        transformer_max_loading_pct: Maximum transformer loading
        verbose: Print progress
    """
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    transformer_max_loading_pct: float = 100.0
    verbose: bool = True


def run_timeseries_powerflow(
    scenario: Dict,
    network_config: Optional[NetworkConfig] = None,
    sim_config: Optional[SimulationConfig] = None
) -> Dict[str, Any]:
    """
    Run time-series power flow for a scenario.
    
    Executes AC power flow for each time interval and collects:
    - Bus voltages (p.u.)
    - Transformer loading (%)
    - Grid P/Q exchange (MW/MVAr)
    - BESS dispatch (MW)
    - Constraint violations
    
    Args:
        scenario: Scenario data dict from scenarios.py
        network_config: Network configuration
        sim_config: Simulation configuration
        
    Returns:
        Dictionary with time-series results
    """
    if sim_config is None:
        sim_config = SimulationConfig()
    
    # Create network
    net = create_datacenter_network(network_config)
    
    n_intervals = scenario["n_intervals"]
    
    # Initialize result arrays
    results = {
        "scenario_name": scenario["name"],
        "load_type": scenario["load_type"],
        "architecture": scenario["architecture"],
        "power_factor": scenario["power_factor"],
        "n_intervals": n_intervals,
        "interval_minutes": scenario["interval_minutes"],
        
        # Time
        "time_hours": scenario["time_hours"],
        
        # Voltage
        "voltage_pu": [],
        
        # Transformer
        "transformer_loading_pct": [],
        
        # Grid exchange
        "grid_mw": [],
        "grid_mvar": [],
        
        # Load
        "load_p_mw": scenario["total_load_p_mw"],
        "load_q_mvar": scenario["total_load_q_mvar"],
        
        # BESS
        "bess_mw": scenario["bess_p_mw"],
        "bess_soc": scenario["bess_soc"],
        
        # Violations
        "violations": [],
        "convergence_failures": [],
        
        # Summary stats
        "summary": {}
    }
    
    # Run power flow for each interval
    if sim_config.verbose:
        print(f"Running time-series power flow for {scenario['name']}...")
        print(f"  Intervals: {n_intervals}")
    
    start_time = pytime.time()
    
    violation_count = 0
    convergence_failures = 0
    
    for t in range(n_intervals):
        # Set load dispatch
        p_mw = scenario["total_load_p_mw"][t]
        q_mvar = scenario["total_load_q_mvar"][t]
        set_load_dispatch(net, p_mw, q_mvar)
        
        # Set BESS dispatch
        bess_p = scenario["bess_p_mw"][t]
        set_bess_dispatch(net, bess_p, 0.0)  # Assume unity PF for BESS
        
        # Run power flow
        pf_results = run_powerflow(net)
        
        if not pf_results["converged"]:
            # Handle convergence failure
            convergence_failures += 1
            results["convergence_failures"].append(t)
            
            # Use placeholder values
            results["voltage_pu"].append(1.0)
            results["transformer_loading_pct"].append(0.0)
            results["grid_mw"].append(p_mw - bess_p)
            results["grid_mvar"].append(q_mvar)
            continue
        
        # Extract results
        results["voltage_pu"].append(pf_results["dc_bus_voltage_pu"])
        results["transformer_loading_pct"].append(pf_results["transformer_loading"])
        results["grid_mw"].append(pf_results["grid_mw"])
        results["grid_mvar"].append(pf_results["grid_mvar"])
        
        # Check constraints
        constraint_check = check_constraints(
            pf_results,
            voltage_min_pu=sim_config.voltage_min_pu,
            voltage_max_pu=sim_config.voltage_max_pu,
            transformer_max_loading_pct=sim_config.transformer_max_loading_pct
        )
        
        if constraint_check["has_violations"]:
            violation_count += 1
            for v in constraint_check["violations"]:
                v["interval"] = t
                v["time_hours"] = scenario["time_hours"][t]
                results["violations"].append(v)
    
    elapsed_time = pytime.time() - start_time
    
    # Calculate summary statistics
    results["summary"] = {
        "simulation_time_sec": elapsed_time,
        "convergence_rate": (n_intervals - convergence_failures) / n_intervals,
        "violation_count": violation_count,
        "violation_rate": violation_count / n_intervals,
        
        # Voltage stats
        "voltage_min_pu": min(results["voltage_pu"]),
        "voltage_max_pu": max(results["voltage_pu"]),
        "voltage_avg_pu": np.mean(results["voltage_pu"]),
        
        # Transformer stats
        "transformer_loading_max_pct": max(results["transformer_loading_pct"]),
        "transformer_loading_avg_pct": np.mean(results["transformer_loading_pct"]),
        
        # Grid exchange stats
        "grid_mw_max": max(results["grid_mw"]),
        "grid_mw_min": min(results["grid_mw"]),
        "grid_mw_avg": np.mean(results["grid_mw"]),
        "grid_mvar_max": max(results["grid_mvar"]),
        "grid_energy_mwh": np.sum(results["grid_mw"]) * (scenario["interval_minutes"] / 60),
        
        # Load stats
        "load_mw_max": max(results["load_p_mw"]),
        "load_mw_min": min(results["load_p_mw"]),
        "load_mw_avg": np.mean(results["load_p_mw"]),
        "load_energy_mwh": np.sum(results["load_p_mw"]) * (scenario["interval_minutes"] / 60),
        
        # BESS stats
        "bess_mw_max_discharge": max(results["bess_mw"]),
        "bess_mw_max_charge": min(results["bess_mw"]),
        "bess_energy_discharged_mwh": sum(p for p in results["bess_mw"] if p > 0) * (scenario["interval_minutes"] / 60),
        "bess_energy_charged_mwh": -sum(p for p in results["bess_mw"] if p < 0) * (scenario["interval_minutes"] / 60),
    }
    
    # Categorize violations
    violation_summary = {
        "undervoltage": 0,
        "overvoltage": 0,
        "overload": 0,
        "convergence": convergence_failures
    }
    for v in results["violations"]:
        v_type = v.get("type", "unknown")
        if v_type in violation_summary:
            violation_summary[v_type] += 1
    
    results["summary"]["violation_breakdown"] = violation_summary
    
    if sim_config.verbose:
        print(f"  Completed in {elapsed_time:.2f}s")
        print(f"  Voltage range: {results['summary']['voltage_min_pu']:.4f} - {results['summary']['voltage_max_pu']:.4f} p.u.")
        print(f"  Max transformer loading: {results['summary']['transformer_loading_max_pct']:.1f}%")
        print(f"  Violations: {violation_count}")
    
    return results


def run_all_scenarios(
    scenarios: List[Dict],
    network_config: Optional[NetworkConfig] = None,
    sim_config: Optional[SimulationConfig] = None
) -> List[Dict]:
    """
    Run time-series power flow for all scenarios.
    
    Args:
        scenarios: List of scenario dictionaries
        network_config: Network configuration
        sim_config: Simulation configuration
        
    Returns:
        List of results dictionaries
    """
    all_results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Running scenario: {scenario['name']}")
        results = run_timeseries_powerflow(scenario, network_config, sim_config)
        all_results.append(results)
    
    return all_results


def create_comparison_summary(results_list: List[Dict]) -> Dict:
    """
    Create a summary comparing all scenarios.
    
    Args:
        results_list: List of simulation results
        
    Returns:
        Comparison summary dictionary
    """
    summary = {
        "n_scenarios": len(results_list),
        "scenarios": []
    }
    
    for r in results_list:
        scenario_summary = {
            "name": r["scenario_name"],
            "load_type": r["load_type"],
            "architecture": r["architecture"],
            "power_factor": r["power_factor"],
            "has_bess": max(abs(p) for p in r["bess_mw"]) > 0.1,
            "voltage_min_pu": r["summary"]["voltage_min_pu"],
            "voltage_max_pu": r["summary"]["voltage_max_pu"],
            "transformer_loading_max_pct": r["summary"]["transformer_loading_max_pct"],
            "grid_mw_max": r["summary"]["grid_mw_max"],
            "violation_count": r["summary"]["violation_count"],
        }
        summary["scenarios"].append(scenario_summary)
    
    # Find best/worst performers
    voltage_mins = [s["voltage_min_pu"] for s in summary["scenarios"]]
    trafo_maxes = [s["transformer_loading_max_pct"] for s in summary["scenarios"]]
    
    summary["best_voltage_scenario"] = summary["scenarios"][np.argmax(voltage_mins)]["name"]
    summary["worst_voltage_scenario"] = summary["scenarios"][np.argmin(voltage_mins)]["name"]
    summary["lowest_loading_scenario"] = summary["scenarios"][np.argmin(trafo_maxes)]["name"]
    summary["highest_loading_scenario"] = summary["scenarios"][np.argmax(trafo_maxes)]["name"]
    
    return summary


if __name__ == "__main__":
    from scenarios import create_standard_scenarios
    
    print("=" * 60)
    print("DATA CENTER POWER FLOW SIMULATION")
    print("Planning-Level, Positive-Sequence Analysis")
    print("=" * 60)
    
    # Create a subset of scenarios for testing
    print("\nCreating test scenarios...")
    
    from scenarios import LoadType, DCArchitecture, ScenarioConfig, generate_scenario, BESSConfig
    
    test_scenarios = []
    
    # Training with AC, no BESS
    test_scenarios.append(generate_scenario(ScenarioConfig(
        name="training_ac_no_bess",
        load_type=LoadType.TRAINING,
        architecture=DCArchitecture.AC_TRADITIONAL,
        base_load_mw=50.0,
        bess_config=None
    )))
    
    # Training with AC, 25 MW BESS
    test_scenarios.append(generate_scenario(ScenarioConfig(
        name="training_ac_bess_25mw",
        load_type=LoadType.TRAINING,
        architecture=DCArchitecture.AC_TRADITIONAL,
        base_load_mw=50.0,
        bess_config=BESSConfig(power_mw=25, energy_mwh=100)
    )))
    
    # Inference with 800V DC, 25 MW BESS
    test_scenarios.append(generate_scenario(ScenarioConfig(
        name="inference_dc800v_bess_25mw",
        load_type=LoadType.INFERENCE,
        architecture=DCArchitecture.DC_800V,
        base_load_mw=50.0,
        bess_config=BESSConfig(power_mw=25, energy_mwh=100)
    )))
    
    # Run simulations
    print(f"\nRunning {len(test_scenarios)} scenarios...")
    
    results = run_all_scenarios(
        test_scenarios,
        network_config=NetworkConfig(transformer_mva=80.0),
        sim_config=SimulationConfig(verbose=True)
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison = create_comparison_summary(results)
    
    print(f"\nBest voltage profile: {comparison['best_voltage_scenario']}")
    print(f"Lowest transformer loading: {comparison['lowest_loading_scenario']}")
    
    print("\nScenario Details:")
    print("-" * 60)
    for s in comparison["scenarios"]:
        print(f"\n  {s['name']}:")
        print(f"    Load Type: {s['load_type']}, Architecture: {s['architecture']}")
        print(f"    Voltage: {s['voltage_min_pu']:.4f} - {s['voltage_max_pu']:.4f} p.u.")
        print(f"    Max Transformer Loading: {s['transformer_loading_max_pct']:.1f}%")
        print(f"    Max Grid Import: {s['grid_mw_max']:.1f} MW")
        print(f"    Violations: {s['violation_count']}")
