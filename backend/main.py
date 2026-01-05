#!/usr/bin/env python3
"""
Data Center Power Flow Simulation - Main Entry Point
=====================================================

Runs the complete planning-level power flow simulation:
1. Creates scenario definitions
2. Runs time-series AC power flow
3. Exports results to JSON for static UI

Usage:
    python main.py                    # Run default scenarios
    python main.py --base-load 75     # Custom base load
    python main.py --output results/  # Custom output directory

IMPORTANT DISCLAIMERS:
- This is PLANNING-LEVEL analysis ONLY
- Positive-sequence, balanced three-phase, steady-state
- NOT suitable for operational decisions
- NOT suitable for interconnection approval
- Results require engineering review before use
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from network import NetworkConfig, create_datacenter_network, get_network_summary
from scenarios import (
    ScenarioConfig, generate_scenario, create_standard_scenarios,
    LoadType, DCArchitecture, BESSConfig
)
from run_simulations import (
    run_timeseries_powerflow, run_all_scenarios,
    create_comparison_summary, SimulationConfig
)
from export_results import export_results_to_json, export_individual_scenarios


def create_planning_scenarios(base_load_mw: float = 50.0) -> list:
    """
    Create scenarios for planning study.
    
    Creates a focused set of scenarios covering:
    - Training vs Inference workloads
    - AC vs 48V DC vs 800V DC architectures
    - With and without BESS
    
    Args:
        base_load_mw: Base IT load capacity
        
    Returns:
        List of scenario dictionaries
    """
    scenarios = []
    
    # Define BESS options
    bess_none = None
    bess_10mw = BESSConfig(power_mw=10, energy_mwh=40, initial_soc=0.50)
    bess_25mw = BESSConfig(power_mw=25, energy_mwh=100, initial_soc=0.50)
    
    # Key scenario combinations
    scenario_configs = [
        # Training workloads
        ("training_ac_no_bess", LoadType.TRAINING, DCArchitecture.AC_TRADITIONAL, bess_none),
        ("training_ac_bess_10mw", LoadType.TRAINING, DCArchitecture.AC_TRADITIONAL, bess_10mw),
        ("training_ac_bess_25mw", LoadType.TRAINING, DCArchitecture.AC_TRADITIONAL, bess_25mw),
        ("training_dc48v_no_bess", LoadType.TRAINING, DCArchitecture.DC_48V, bess_none),
        ("training_dc48v_bess_25mw", LoadType.TRAINING, DCArchitecture.DC_48V, bess_25mw),
        ("training_dc800v_no_bess", LoadType.TRAINING, DCArchitecture.DC_800V, bess_none),
        ("training_dc800v_bess_25mw", LoadType.TRAINING, DCArchitecture.DC_800V, bess_25mw),
        
        # Inference workloads
        ("inference_ac_no_bess", LoadType.INFERENCE, DCArchitecture.AC_TRADITIONAL, bess_none),
        ("inference_ac_bess_25mw", LoadType.INFERENCE, DCArchitecture.AC_TRADITIONAL, bess_25mw),
        ("inference_dc48v_no_bess", LoadType.INFERENCE, DCArchitecture.DC_48V, bess_none),
        ("inference_dc48v_bess_25mw", LoadType.INFERENCE, DCArchitecture.DC_48V, bess_25mw),
        ("inference_dc800v_no_bess", LoadType.INFERENCE, DCArchitecture.DC_800V, bess_none),
        ("inference_dc800v_bess_25mw", LoadType.INFERENCE, DCArchitecture.DC_800V, bess_25mw),
        
        # Mixed workloads
        ("mixed_ac_bess_25mw", LoadType.MIXED, DCArchitecture.AC_TRADITIONAL, bess_25mw),
        ("mixed_dc800v_bess_25mw", LoadType.MIXED, DCArchitecture.DC_800V, bess_25mw),
    ]
    
    for name, load_type, arch, bess in scenario_configs:
        config = ScenarioConfig(
            name=name,
            load_type=load_type,
            architecture=arch,
            base_load_mw=base_load_mw,
            bess_config=bess,
            time_hours=24.0,
            interval_minutes=15
        )
        scenarios.append(generate_scenario(config))
    
    return scenarios


def run_planning_study(
    base_load_mw: float = 50.0,
    transformer_mva: float = 80.0,
    output_dir: str = "../frontend/public/data",
    verbose: bool = True
) -> dict:
    """
    Run complete planning study.
    
    Args:
        base_load_mw: Base IT load capacity
        transformer_mva: Main transformer rating
        output_dir: Output directory for JSON files
        verbose: Print progress
        
    Returns:
        Summary of study results
    """
    print("=" * 70)
    print("DATA CENTER POWER FLOW PLANNING STUDY")
    print("=" * 70)
    print("\n⚠️  DISCLAIMER:")
    print("    This is a PLANNING-LEVEL study using positive-sequence,")
    print("    balanced three-phase, steady-state power flow analysis.")
    print("    Results are NOT suitable for operational decisions.")
    print("=" * 70)
    
    # Network configuration
    network_config = NetworkConfig(
        name="DataCenter_50MW",
        grid_voltage_kv=138.0,
        dc_bus_voltage_kv=13.8,
        transformer_mva=transformer_mva,
        transformer_vk_percent=8.0,
        grid_s_sc_mva=2000.0
    )
    
    # Simulation configuration
    sim_config = SimulationConfig(
        voltage_min_pu=0.95,
        voltage_max_pu=1.05,
        transformer_max_loading_pct=100.0,
        verbose=verbose
    )
    
    # Print network summary
    net = create_datacenter_network(network_config)
    if verbose:
        print("\nNetwork Configuration:")
        summary = get_network_summary(net)
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Create scenarios
    print("\nCreating planning scenarios...")
    scenarios = create_planning_scenarios(base_load_mw)
    print(f"  Created {len(scenarios)} scenarios")
    
    # Run simulations
    print("\nRunning time-series power flow simulations...")
    results = run_all_scenarios(scenarios, network_config, sim_config)
    
    # Create comparison summary
    comparison = create_comparison_summary(results)
    
    # Export results
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export combined results
    combined_path = os.path.join(output_dir, "simulation_results.json")
    export_results_to_json(
        results,
        combined_path,
        name=f"Data Center {base_load_mw}MW Planning Study",
        description=f"Planning-level power flow study for {base_load_mw}MW data center with BESS",
        comparison=comparison
    )
    
    # Export individual scenario files
    scenarios_dir = os.path.join(output_dir, "scenarios")
    export_individual_scenarios(results, scenarios_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("STUDY SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal scenarios analyzed: {len(results)}")
    print(f"\nKey findings:")
    print(f"  Best voltage profile: {comparison['best_voltage_scenario']}")
    print(f"  Worst voltage profile: {comparison['worst_voltage_scenario']}")
    print(f"  Lowest transformer loading: {comparison['lowest_loading_scenario']}")
    print(f"  Highest transformer loading: {comparison['highest_loading_scenario']}")
    
    # Identify scenarios with violations
    violations_scenarios = [
        s["name"] for s in comparison["scenarios"] 
        if s["violation_count"] > 0
    ]
    
    if violations_scenarios:
        print(f"\n⚠️  Scenarios with constraint violations:")
        for name in violations_scenarios:
            print(f"    - {name}")
    else:
        print(f"\n✓ All scenarios within operating limits")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"\n  Combined results: {combined_path}")
    print(f"  Individual scenarios: {scenarios_dir}/")
    
    return {
        "n_scenarios": len(results),
        "comparison": comparison,
        "output_path": combined_path
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data Center Power Flow Planning Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                         # Run with defaults (50MW load)
    python main.py --base-load 75          # 75MW data center
    python main.py --transformer-mva 100   # 100 MVA transformer
    python main.py --output ./results      # Custom output directory
    
DISCLAIMER:
    This tool performs PLANNING-LEVEL power flow analysis only.
    Results are NOT suitable for operational decisions or 
    interconnection approval without engineering review.
        """
    )
    
    parser.add_argument(
        "--base-load", 
        type=float, 
        default=50.0,
        help="Base IT load capacity in MW (default: 50)"
    )
    
    parser.add_argument(
        "--transformer-mva",
        type=float,
        default=80.0,
        help="Main transformer rating in MVA (default: 80)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="../frontend/public/data",
        help="Output directory for JSON files"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Run study
    summary = run_planning_study(
        base_load_mw=args.base_load,
        transformer_mva=args.transformer_mva,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    print("\n✓ Study complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
