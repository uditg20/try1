"""
Result Export for Static UI
===========================

Exports power flow simulation results to JSON format for the static React UI.

OUTPUT SCHEMA:
{
    "metadata": {
        "version": "1.0",
        "generated": "ISO timestamp",
        "disclaimer": "Planning-level results only",
        ...
    },
    "scenarios": [
        {
            "name": "...",
            "load_type": "training|inference|mixed",
            "architecture": "ac_traditional|dc_48v|dc_800v",
            "bess_config": {...} | null,
            "time": [...],
            "voltage_pu": [...],
            "transformer_loading_pct": [...],
            "grid_mw": [...],
            "grid_mvar": [...],
            "bess_mw": [...],
            "bess_soc": [...],
            "violations": {...},
            "summary": {...}
        },
        ...
    ],
    "comparison": {...}
}
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


def create_metadata(
    name: str = "Data Center Power Flow Study",
    description: str = "Planning-level electrical simulation results"
) -> Dict[str, Any]:
    """
    Create metadata block for export.
    
    Args:
        name: Study name
        description: Study description
        
    Returns:
        Metadata dictionary
    """
    return {
        "version": "1.0.0",
        "name": name,
        "description": description,
        "generated": datetime.utcnow().isoformat() + "Z",
        
        "disclaimer": (
            "PLANNING-LEVEL RESULTS ONLY. "
            "This analysis uses positive-sequence, balanced three-phase, "
            "steady-state power flow simulation. Results are suitable for "
            "planning and sizing studies ONLY. Do NOT use for operational "
            "decisions, protection coordination, or interconnection studies "
            "without proper engineering review."
        ),
        
        "scope": {
            "abstraction_level": "planning",
            "power_flow_type": "positive_sequence_ac",
            "system_assumption": "balanced_three_phase",
            "time_domain": "steady_state"
        },
        
        "limitations": [
            "No EMT/transient simulation",
            "No inverter control dynamics",
            "No protection coordination",
            "No harmonic analysis",
            "No unbalanced conditions",
            "Aggregated load model only (no explicit GPU/rack/PSU modeling)"
        ],
        
        "not_suitable_for": [
            "Operational dispatch decisions",
            "Protection relay settings",
            "Interconnection approval",
            "Firmware/control design",
            "Real-time optimization"
        ]
    }


def format_scenario_for_export(result: Dict) -> Dict[str, Any]:
    """
    Format a scenario result for JSON export.
    
    Args:
        result: Raw simulation result dictionary
        
    Returns:
        Formatted scenario for export
    """
    return {
        "name": result["scenario_name"],
        "load_type": result["load_type"],
        "architecture": result["architecture"],
        "power_factor": result["power_factor"],
        "n_intervals": result["n_intervals"],
        "interval_minutes": result["interval_minutes"],
        
        # Time series data
        "time": result["time_hours"],
        "voltage_pu": [round(v, 4) for v in result["voltage_pu"]],
        "transformer_loading_pct": [round(t, 2) for t in result["transformer_loading_pct"]],
        "grid_mw": [round(p, 2) for p in result["grid_mw"]],
        "grid_mvar": [round(q, 2) for q in result["grid_mvar"]],
        "load_p_mw": [round(p, 2) for p in result["load_p_mw"]],
        "load_q_mvar": [round(q, 2) for q in result["load_q_mvar"]],
        "bess_mw": [round(p, 2) for p in result["bess_mw"]],
        "bess_soc": [round(s, 3) for s in result["bess_soc"]],
        
        # Violations summary
        "violations": {
            "count": result["summary"]["violation_count"],
            "rate": round(result["summary"]["violation_rate"], 4),
            "breakdown": result["summary"]["violation_breakdown"],
            "details": result["violations"][:10]  # First 10 violations
        },
        
        # Summary statistics
        "summary": {
            "voltage": {
                "min_pu": round(result["summary"]["voltage_min_pu"], 4),
                "max_pu": round(result["summary"]["voltage_max_pu"], 4),
                "avg_pu": round(result["summary"]["voltage_avg_pu"], 4)
            },
            "transformer": {
                "max_loading_pct": round(result["summary"]["transformer_loading_max_pct"], 2),
                "avg_loading_pct": round(result["summary"]["transformer_loading_avg_pct"], 2)
            },
            "grid": {
                "max_import_mw": round(result["summary"]["grid_mw_max"], 2),
                "min_import_mw": round(result["summary"]["grid_mw_min"], 2),
                "avg_import_mw": round(result["summary"]["grid_mw_avg"], 2),
                "total_energy_mwh": round(result["summary"]["grid_energy_mwh"], 2)
            },
            "load": {
                "peak_mw": round(result["summary"]["load_mw_max"], 2),
                "min_mw": round(result["summary"]["load_mw_min"], 2),
                "avg_mw": round(result["summary"]["load_mw_avg"], 2),
                "total_energy_mwh": round(result["summary"]["load_energy_mwh"], 2)
            },
            "bess": {
                "max_discharge_mw": round(result["summary"]["bess_mw_max_discharge"], 2),
                "max_charge_mw": round(abs(result["summary"]["bess_mw_max_charge"]), 2),
                "energy_discharged_mwh": round(result["summary"]["bess_energy_discharged_mwh"], 2),
                "energy_charged_mwh": round(result["summary"]["bess_energy_charged_mwh"], 2)
            }
        }
    }


def export_results_to_json(
    results: List[Dict],
    output_path: str,
    name: str = "Data Center Power Flow Study",
    description: str = "Planning-level electrical simulation results",
    comparison: Optional[Dict] = None
) -> str:
    """
    Export all simulation results to a JSON file.
    
    Args:
        results: List of simulation result dictionaries
        output_path: Path for output JSON file
        name: Study name
        description: Study description
        comparison: Optional comparison summary
        
    Returns:
        Path to exported file
    """
    # Create output structure
    output = {
        "metadata": create_metadata(name, description),
        "scenarios": [format_scenario_for_export(r) for r in results]
    }
    
    # Add comparison if provided
    if comparison:
        output["comparison"] = comparison
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results exported to: {output_path}")
    return output_path


def export_individual_scenarios(
    results: List[Dict],
    output_dir: str
) -> List[str]:
    """
    Export each scenario to a separate JSON file.
    
    Useful for lazy loading in the UI.
    
    Args:
        results: List of simulation result dictionaries
        output_dir: Directory for output files
        
    Returns:
        List of exported file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = []
    manifest = {
        "version": "1.0.0",
        "generated": datetime.utcnow().isoformat() + "Z",
        "scenarios": []
    }
    
    for result in results:
        scenario_data = format_scenario_for_export(result)
        filename = f"{result['scenario_name']}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        
        exported_files.append(filepath)
        manifest["scenarios"].append({
            "name": result["scenario_name"],
            "file": filename,
            "load_type": result["load_type"],
            "architecture": result["architecture"],
            "has_bess": max(abs(p) for p in result["bess_mw"]) > 0.1
        })
    
    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Exported {len(exported_files)} scenario files to: {output_dir}")
    return exported_files


def generate_sample_data() -> Dict:
    """
    Generate sample data structure for UI development.
    
    Returns:
        Sample data dictionary matching export schema
    """
    import numpy as np
    
    n_intervals = 96  # 15-min intervals for 24 hours
    time = np.linspace(0, 24, n_intervals).tolist()
    
    # Synthetic data
    base_load = 50
    t_arr = np.array(time)
    
    # Training-like load profile
    load_profile = base_load * (0.9 + 0.1 * np.sin(2 * np.pi * t_arr / 24))
    
    # Voltage response
    voltage = 1.0 - 0.01 * (load_profile - base_load) / 10
    
    # Transformer loading
    trafo_loading = load_profile / 80 * 100
    
    # Grid power
    grid_mw = load_profile.copy()
    
    # BESS peak shaving
    peak_threshold = 52
    bess_mw = np.where(load_profile > peak_threshold, load_profile - peak_threshold, 0)
    bess_mw = np.clip(bess_mw, -25, 25)
    
    grid_mw = grid_mw - bess_mw
    
    # SOC tracking
    soc = [0.5]
    for p in bess_mw[:-1]:
        delta = -p * 0.25 / 100  # Simplified
        soc.append(np.clip(soc[-1] + delta, 0.1, 0.9))
    
    sample = {
        "metadata": create_metadata("Sample Study", "Development sample data"),
        "scenarios": [
            {
                "name": "sample_training_ac_bess",
                "load_type": "training",
                "architecture": "ac_traditional",
                "power_factor": 0.92,
                "n_intervals": n_intervals,
                "interval_minutes": 15,
                "time": time,
                "voltage_pu": voltage.tolist(),
                "transformer_loading_pct": trafo_loading.tolist(),
                "grid_mw": grid_mw.tolist(),
                "grid_mvar": (grid_mw * 0.4).tolist(),
                "load_p_mw": load_profile.tolist(),
                "load_q_mvar": (load_profile * 0.4).tolist(),
                "bess_mw": bess_mw.tolist(),
                "bess_soc": soc,
                "violations": {
                    "count": 0,
                    "rate": 0,
                    "breakdown": {},
                    "details": []
                },
                "summary": {
                    "voltage": {"min_pu": 0.98, "max_pu": 1.01, "avg_pu": 0.995},
                    "transformer": {"max_loading_pct": 65, "avg_loading_pct": 60},
                    "grid": {"max_import_mw": 50, "min_import_mw": 42, "avg_import_mw": 46},
                    "load": {"peak_mw": 55, "min_mw": 45, "avg_mw": 50},
                    "bess": {"max_discharge_mw": 5, "max_charge_mw": 3}
                }
            }
        ]
    }
    
    return sample


if __name__ == "__main__":
    # Generate sample data for testing
    print("Generating sample data...")
    sample = generate_sample_data()
    
    # Export sample
    output_path = "../frontend/public/data/sample_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(sample, f, indent=2)
    
    print(f"Sample data exported to: {output_path}")
