"""
Data Center Network Model (pandapower)
======================================

Defines the planning-level electrical network for a data center with BESS.

TOPOLOGY:
    Grid (External Grid - Slack)
    │
    └──► HV/MV Transformer (e.g., 138kV/13.8kV)
         │
         └──► Data Center Main Bus (MV)
              │
              ├──► Aggregated Data Center Load (P + Q)
              │    (Represents all IT, cooling, lighting via P/Q envelope)
              │
              └──► BESS Inverter (PQ-controlled)
                   (Charge/discharge with reactive support)

ABSTRACTION LEVEL:
- This is a PLANNING-LEVEL, positive-sequence, balanced 3-phase model
- GPUs, racks, PSUs, UPS, rectifiers are NOT modeled explicitly
- All internal DC loads are aggregated into P/Q envelopes
- Power factor variations represent architecture differences

NOT MODELED:
- EMT transients
- Inverter controls/firmware
- Protection coordination
- Unbalanced conditions
- Harmonics

Author: Power Systems Planning Team
"""

import pandapower as pp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class NetworkConfig:
    """
    Configuration for data center network model.
    
    Attributes:
        name: Network identifier
        grid_voltage_kv: HV grid voltage (kV)
        dc_bus_voltage_kv: Data center MV bus voltage (kV)
        transformer_mva: Main transformer rating (MVA)
        transformer_vk_percent: Transformer short-circuit voltage (%)
        transformer_vkr_percent: Transformer resistive SC voltage (%)
        grid_s_sc_mva: Grid short-circuit capacity (MVA)
        grid_rx_ratio: Grid R/X ratio
    """
    name: str = "DataCenter"
    grid_voltage_kv: float = 138.0
    dc_bus_voltage_kv: float = 13.8
    transformer_mva: float = 80.0  # Total MVA (e.g., 2x40 MVA N+1)
    transformer_vk_percent: float = 8.0
    transformer_vkr_percent: float = 0.5
    grid_s_sc_mva: float = 2000.0  # Strong grid
    grid_rx_ratio: float = 0.1


def create_datacenter_network(config: Optional[NetworkConfig] = None) -> pp.pandapowerNet:
    """
    Create a pandapower network representing a data center with BESS.
    
    The network topology is:
        ext_grid (slack) @ HV bus
            │
            └── transformer
                    │
                    └── DC main bus @ MV
                        │
                        ├── load (aggregated DC load)
                        │
                        └── sgen (BESS, PQ-controlled)
    
    Args:
        config: Network configuration parameters
        
    Returns:
        pandapower network object
    
    Note:
        This is a PLANNING-LEVEL model. All loads are aggregated P/Q.
        BESS is modeled as a static generator with controllable P/Q.
    """
    if config is None:
        config = NetworkConfig()
    
    # Create empty network
    net = pp.create_empty_network(name=config.name)
    
    # =========================================================================
    # BUSES
    # =========================================================================
    
    # HV bus (grid connection point)
    hv_bus = pp.create_bus(
        net,
        vn_kv=config.grid_voltage_kv,
        name="HV_Grid_Bus",
        type="b",  # busbar
        zone="grid"
    )
    
    # MV bus (data center main distribution)
    mv_bus = pp.create_bus(
        net,
        vn_kv=config.dc_bus_voltage_kv,
        name="DC_Main_Bus",
        type="b",
        zone="datacenter"
    )
    
    # Store bus indices for reference
    net.bus_indices = {
        "hv_bus": hv_bus,
        "mv_bus": mv_bus
    }
    
    # =========================================================================
    # EXTERNAL GRID (Slack)
    # =========================================================================
    
    # External grid connection - this is the slack bus
    # Models the utility interconnection
    pp.create_ext_grid(
        net,
        bus=hv_bus,
        vm_pu=1.0,  # Voltage setpoint (planning assumption)
        va_degree=0.0,  # Reference angle
        name="Grid_POI",
        s_sc_max_mva=config.grid_s_sc_mva,
        rx_max=config.grid_rx_ratio,
        in_service=True
    )
    
    # =========================================================================
    # TRANSFORMER (HV/MV)
    # =========================================================================
    
    # Main step-down transformer
    # In practice, this might be multiple units in N+1 configuration
    # Here we model the aggregate firm capacity
    pp.create_transformer_from_parameters(
        net,
        hv_bus=hv_bus,
        lv_bus=mv_bus,
        sn_mva=config.transformer_mva,
        vn_hv_kv=config.grid_voltage_kv,
        vn_lv_kv=config.dc_bus_voltage_kv,
        vk_percent=config.transformer_vk_percent,
        vkr_percent=config.transformer_vkr_percent,
        pfe_kw=0,  # Neglect no-load losses for planning
        i0_percent=0,  # Neglect magnetizing current for planning
        shift_degree=0,
        name="Main_Transformer",
        in_service=True
    )
    
    # =========================================================================
    # LOAD (Aggregated Data Center Load)
    # =========================================================================
    
    # Create initial load placeholder
    # Actual P/Q values are set per scenario in time-series simulation
    pp.create_load(
        net,
        bus=mv_bus,
        p_mw=50.0,  # Placeholder - updated per scenario
        q_mvar=16.4,  # Placeholder - assumes ~0.95 PF
        name="DC_Aggregated_Load",
        in_service=True
    )
    
    # =========================================================================
    # BESS (Static Generator - PQ controlled)
    # =========================================================================
    
    # BESS modeled as static generator
    # Positive P = generation (discharge), Negative P = consumption (charge)
    # Q can provide reactive support
    pp.create_sgen(
        net,
        bus=mv_bus,
        p_mw=0.0,  # Placeholder - updated per scenario
        q_mvar=0.0,  # Placeholder - updated per scenario
        name="BESS",
        type="BESS",
        in_service=True
    )
    
    # =========================================================================
    # NETWORK METADATA
    # =========================================================================
    
    # Store configuration for reference
    net.config = config
    
    # Store indices for easy access
    net.component_indices = {
        "load": 0,  # DC_Aggregated_Load
        "bess": 0,  # BESS sgen
        "transformer": 0,  # Main_Transformer
        "ext_grid": 0  # Grid_POI
    }
    
    return net


def set_load_dispatch(
    net: pp.pandapowerNet,
    p_mw: float,
    q_mvar: float
) -> None:
    """
    Set the aggregated data center load for a dispatch interval.
    
    Args:
        net: pandapower network
        p_mw: Active power demand (MW)
        q_mvar: Reactive power demand (MVAr)
    """
    load_idx = net.component_indices["load"]
    net.load.at[load_idx, "p_mw"] = p_mw
    net.load.at[load_idx, "q_mvar"] = q_mvar


def set_bess_dispatch(
    net: pp.pandapowerNet,
    p_mw: float,
    q_mvar: float = 0.0
) -> None:
    """
    Set BESS dispatch for an interval.
    
    Positive P = discharge (generation)
    Negative P = charge (consumption)
    
    Args:
        net: pandapower network
        p_mw: Active power (MW) - positive=discharge, negative=charge
        q_mvar: Reactive power (MVAr) - for voltage support
    """
    bess_idx = net.component_indices["bess"]
    net.sgen.at[bess_idx, "p_mw"] = p_mw
    net.sgen.at[bess_idx, "q_mvar"] = q_mvar


def run_powerflow(
    net: pp.pandapowerNet,
    algorithm: str = "nr",  # Newton-Raphson
    max_iteration: int = 50
) -> Dict[str, Any]:
    """
    Run AC power flow and extract results.
    
    This is a STEADY-STATE, POSITIVE-SEQUENCE power flow.
    It does NOT capture:
    - Transients
    - Unbalance
    - Harmonics
    
    Args:
        net: pandapower network
        algorithm: Power flow algorithm ('nr', 'bfsw', etc.)
        max_iteration: Maximum solver iterations
        
    Returns:
        Dictionary with power flow results
    """
    try:
        pp.runpp(
            net,
            algorithm=algorithm,
            max_iteration=max_iteration,
            calculate_voltage_angles=True,
            enforce_q_lims=False,  # Planning study
            numba=False  # Disable numba to avoid warnings
        )
        converged = True
    except Exception as e:
        converged = False
        return {
            "converged": False,
            "error": str(e)
        }
    
    # Extract results
    results = {
        "converged": converged,
        
        # Bus results
        "bus_vm_pu": net.res_bus["vm_pu"].to_dict(),
        "bus_va_degree": net.res_bus["va_degree"].to_dict(),
        
        # Transformer results
        "transformer_loading_pct": net.res_trafo["loading_percent"].to_dict(),
        "transformer_p_hv_mw": net.res_trafo["p_hv_mw"].to_dict(),
        "transformer_q_hv_mvar": net.res_trafo["q_hv_mvar"].to_dict(),
        
        # Grid exchange (external grid)
        "grid_p_mw": net.res_ext_grid["p_mw"].to_dict(),
        "grid_q_mvar": net.res_ext_grid["q_mvar"].to_dict(),
        
        # Load
        "load_p_mw": net.res_load["p_mw"].to_dict(),
        "load_q_mvar": net.res_load["q_mvar"].to_dict(),
        
        # BESS
        "bess_p_mw": net.res_sgen["p_mw"].to_dict(),
        "bess_q_mvar": net.res_sgen["q_mvar"].to_dict()
    }
    
    # Get key values for easier access
    mv_bus_idx = net.bus_indices["mv_bus"]
    results["dc_bus_voltage_pu"] = net.res_bus.at[mv_bus_idx, "vm_pu"]
    results["transformer_loading"] = net.res_trafo.at[0, "loading_percent"]
    results["grid_mw"] = float(net.res_ext_grid.at[0, "p_mw"])
    results["grid_mvar"] = float(net.res_ext_grid.at[0, "q_mvar"])
    
    return results


def check_constraints(
    results: Dict[str, Any],
    voltage_min_pu: float = 0.95,
    voltage_max_pu: float = 1.05,
    transformer_max_loading_pct: float = 100.0
) -> Dict[str, Any]:
    """
    Check for constraint violations in power flow results.
    
    Planning-level screening:
    - Voltage limits (typically ±5% for MV)
    - Transformer loading limits
    
    Args:
        results: Power flow results dictionary
        voltage_min_pu: Minimum voltage (p.u.)
        voltage_max_pu: Maximum voltage (p.u.)
        transformer_max_loading_pct: Maximum transformer loading (%)
        
    Returns:
        Dictionary with violation details
    """
    violations = []
    
    if not results.get("converged", False):
        return {
            "has_violations": True,
            "violations": [{"type": "convergence", "description": "Power flow did not converge"}]
        }
    
    # Check voltage
    dc_voltage = results.get("dc_bus_voltage_pu", 1.0)
    if dc_voltage < voltage_min_pu:
        violations.append({
            "type": "undervoltage",
            "location": "DC_Main_Bus",
            "limit_pu": voltage_min_pu,
            "actual_pu": dc_voltage,
            "severity": "warning" if dc_voltage > 0.90 else "critical"
        })
    elif dc_voltage > voltage_max_pu:
        violations.append({
            "type": "overvoltage",
            "location": "DC_Main_Bus",
            "limit_pu": voltage_max_pu,
            "actual_pu": dc_voltage,
            "severity": "warning" if dc_voltage < 1.10 else "critical"
        })
    
    # Check transformer loading
    trafo_loading = results.get("transformer_loading", 0.0)
    if trafo_loading > transformer_max_loading_pct:
        violations.append({
            "type": "overload",
            "location": "Main_Transformer",
            "limit_pct": transformer_max_loading_pct,
            "actual_pct": trafo_loading,
            "severity": "warning" if trafo_loading < 110 else "critical"
        })
    
    return {
        "has_violations": len(violations) > 0,
        "violations": violations,
        "voltage_pu": dc_voltage,
        "transformer_loading_pct": trafo_loading
    }


def get_network_summary(net: pp.pandapowerNet) -> Dict[str, Any]:
    """
    Get a summary of the network configuration.
    
    Args:
        net: pandapower network
        
    Returns:
        Dictionary with network parameters
    """
    config = net.config
    return {
        "name": config.name,
        "grid_voltage_kv": config.grid_voltage_kv,
        "dc_bus_voltage_kv": config.dc_bus_voltage_kv,
        "transformer_mva": config.transformer_mva,
        "grid_s_sc_mva": config.grid_s_sc_mva,
        "n_buses": len(net.bus),
        "n_transformers": len(net.trafo),
        "n_loads": len(net.load),
        "n_sgens": len(net.sgen)
    }


if __name__ == "__main__":
    # Quick test
    print("Creating data center network...")
    net = create_datacenter_network()
    
    print("\nNetwork summary:")
    summary = get_network_summary(net)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nRunning power flow with default load...")
    results = run_powerflow(net)
    
    if results["converged"]:
        print(f"  DC Bus Voltage: {results['dc_bus_voltage_pu']:.4f} p.u.")
        print(f"  Transformer Loading: {results['transformer_loading']:.1f}%")
        print(f"  Grid P: {results['grid_mw']:.2f} MW")
        print(f"  Grid Q: {results['grid_mvar']:.2f} MVAr")
    else:
        print(f"  Power flow failed: {results.get('error', 'Unknown error')}")
