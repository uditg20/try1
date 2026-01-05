"""
Load and BESS Scenarios
=======================

Defines planning scenarios for data center power flow studies.

SCENARIO CATEGORIES:

1. LOAD TYPE SCENARIOS:
   - Training workload: High sustained load, high utilization
   - Inference workload: Variable load, bursty patterns

2. ARCHITECTURE SCENARIOS (via Power Factor):
   - AC Architecture: Traditional, lower PF (~0.90-0.95)
   - 48V DC Architecture: Better PF, reduced conversion losses (~0.96-0.98)
   - 800V DC Architecture: Highest efficiency, best PF (~0.98-0.99)
   
   Note: The architecture differences are captured via P/Q envelopes.
   We do NOT model individual PSUs, rectifiers, or UPS explicitly.
   
3. BESS SCENARIOS:
   - Size variations (0 MW, 10 MW, 25 MW, 50 MW)
   - Dispatch profiles (peak shaving, load following, TOU arbitrage)

ABSTRACTION:
All scenarios produce P/Q profiles for the aggregated load model.
Internal DC architecture is abstracted to power factor differences.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class LoadType(Enum):
    """Data center workload type."""
    TRAINING = "training"
    INFERENCE = "inference"
    MIXED = "mixed"


class DCArchitecture(Enum):
    """
    DC power distribution architecture.
    
    Mapped to power factor ranges for P/Q envelope calculation.
    Higher voltage DC = better efficiency = higher PF.
    """
    AC_TRADITIONAL = "ac_traditional"  # AC distribution with PSU conversion
    DC_48V = "dc_48v"  # 48V DC distribution (telco-style)
    DC_800V = "dc_800v"  # 800V DC distribution (hyperscale)


# Architecture to power factor mapping
ARCHITECTURE_PF = {
    DCArchitecture.AC_TRADITIONAL: 0.92,
    DCArchitecture.DC_48V: 0.97,
    DCArchitecture.DC_800V: 0.99,
}

# Architecture to efficiency multiplier (relative losses)
ARCHITECTURE_EFFICIENCY = {
    DCArchitecture.AC_TRADITIONAL: 1.00,  # Baseline
    DCArchitecture.DC_48V: 0.97,  # ~3% less total loss
    DCArchitecture.DC_800V: 0.94,  # ~6% less total loss
}


@dataclass
class BESSConfig:
    """
    BESS configuration for scenarios.
    
    Attributes:
        power_mw: Maximum charge/discharge power (MW)
        energy_mwh: Total energy capacity (MWh)
        efficiency: Round-trip efficiency
        initial_soc: Initial state of charge (0-1)
        min_soc: Minimum SOC (0-1)
        max_soc: Maximum SOC (0-1)
    """
    power_mw: float = 25.0
    energy_mwh: float = 100.0
    efficiency: float = 0.88  # Round-trip
    initial_soc: float = 0.50
    min_soc: float = 0.10
    max_soc: float = 0.90


@dataclass
class ScenarioConfig:
    """
    Complete scenario configuration.
    
    Attributes:
        name: Scenario identifier
        load_type: Training, inference, or mixed
        architecture: DC architecture type
        base_load_mw: Base IT load (MW)
        bess_config: BESS parameters (None if no BESS)
        time_hours: Simulation duration (hours)
        interval_minutes: Time step (minutes)
    """
    name: str
    load_type: LoadType
    architecture: DCArchitecture
    base_load_mw: float
    bess_config: Optional[BESSConfig] = None
    time_hours: float = 24.0
    interval_minutes: int = 15


def generate_training_load_profile(
    base_mw: float,
    n_intervals: int,
    utilization_min: float = 0.85,
    utilization_max: float = 0.98
) -> np.ndarray:
    """
    Generate load profile for training workload.
    
    Training workloads are characterized by:
    - High sustained utilization
    - Long job durations
    - Gradual ramps (checkpointing, data loading)
    
    Args:
        base_mw: Base IT load capacity
        n_intervals: Number of time intervals
        utilization_min: Minimum utilization fraction
        utilization_max: Maximum utilization fraction
        
    Returns:
        Array of load values (MW) per interval
    """
    # Training has high, sustained load with slow variations
    # Model as high base + slow sinusoidal variation + occasional steps
    
    t = np.linspace(0, 1, n_intervals)
    
    # Base utilization with slow oscillation (job batching)
    base_util = utilization_min + 0.5 * (utilization_max - utilization_min)
    oscillation = 0.3 * (utilization_max - utilization_min) * np.sin(2 * np.pi * t * 2)
    
    # Occasional step changes (new job starts/ends)
    steps = np.zeros(n_intervals)
    step_intervals = [int(n_intervals * 0.15), int(n_intervals * 0.45), 
                      int(n_intervals * 0.7), int(n_intervals * 0.85)]
    step_values = [0.05, -0.03, 0.04, -0.02]
    
    for i, (interval, value) in enumerate(zip(step_intervals, step_values)):
        steps[interval:] += value * (utilization_max - utilization_min)
    
    utilization = np.clip(base_util + oscillation + steps, utilization_min, utilization_max)
    
    return base_mw * utilization


def generate_inference_load_profile(
    base_mw: float,
    n_intervals: int,
    utilization_min: float = 0.30,
    utilization_max: float = 0.95
) -> np.ndarray:
    """
    Generate load profile for inference workload.
    
    Inference workloads are characterized by:
    - Variable utilization based on request traffic
    - Diurnal patterns (follows user activity)
    - Bursty spikes
    
    Args:
        base_mw: Base IT load capacity
        n_intervals: Number of time intervals
        utilization_min: Minimum utilization fraction
        utilization_max: Maximum utilization fraction
        
    Returns:
        Array of load values (MW) per interval
    """
    t = np.linspace(0, 24, n_intervals)  # Hours
    
    # Diurnal pattern - peaks during business hours
    # Two peaks: morning (10am) and afternoon (3pm)
    diurnal = (
        0.3 * np.exp(-((t - 10) ** 2) / 8) +  # Morning peak
        0.4 * np.exp(-((t - 15) ** 2) / 10) +  # Afternoon peak
        0.15 * np.exp(-((t - 20) ** 2) / 6)    # Evening shoulder
    )
    diurnal = diurnal / diurnal.max()  # Normalize to 0-1
    
    # Add base load (minimum utilization)
    base_load = utilization_min + diurnal * (utilization_max - utilization_min - 0.1)
    
    # Add random bursts
    np.random.seed(42)  # Reproducibility
    bursts = np.random.choice([0, 0, 0, 0, 0.05, 0.10, 0.15], size=n_intervals)
    
    utilization = np.clip(base_load + bursts, utilization_min, utilization_max)
    
    return base_mw * utilization


def generate_mixed_load_profile(
    base_mw: float,
    n_intervals: int,
    training_fraction: float = 0.6
) -> np.ndarray:
    """
    Generate load profile for mixed training/inference workload.
    
    Args:
        base_mw: Base IT load capacity
        n_intervals: Number of time intervals
        training_fraction: Fraction of load dedicated to training
        
    Returns:
        Array of load values (MW) per interval
    """
    training_mw = base_mw * training_fraction
    inference_mw = base_mw * (1 - training_fraction)
    
    training_load = generate_training_load_profile(training_mw, n_intervals)
    inference_load = generate_inference_load_profile(inference_mw, n_intervals)
    
    return training_load + inference_load


def calculate_q_from_p(
    p_mw: np.ndarray,
    architecture: DCArchitecture
) -> np.ndarray:
    """
    Calculate reactive power from active power based on architecture.
    
    Q = P * tan(arccos(PF))
    
    Args:
        p_mw: Active power array (MW)
        architecture: DC architecture type
        
    Returns:
        Reactive power array (MVAr)
    """
    pf = ARCHITECTURE_PF[architecture]
    tan_phi = np.sqrt(1 / pf**2 - 1)
    return p_mw * tan_phi


def generate_cooling_load(
    it_load_mw: np.ndarray,
    pue: float = 1.4,
    lag_intervals: int = 2
) -> np.ndarray:
    """
    Generate cooling load based on IT load.
    
    Cooling is proportional to IT load with a time lag.
    
    Args:
        it_load_mw: IT load profile (MW)
        pue: Power Usage Effectiveness
        lag_intervals: Thermal lag in intervals
        
    Returns:
        Cooling load profile (MW)
    """
    # Cooling is (PUE - 1) * IT load
    cooling_fraction = pue - 1.0
    
    # Apply thermal lag (cooling responds slowly)
    cooling_load = cooling_fraction * it_load_mw
    
    if lag_intervals > 0:
        # Simple moving average for lag effect
        cooling_load = np.convolve(
            cooling_load, 
            np.ones(lag_intervals) / lag_intervals,
            mode='same'
        )
    
    return cooling_load


def generate_bess_dispatch_peak_shaving(
    total_load_mw: np.ndarray,
    bess_config: BESSConfig,
    peak_threshold_mw: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate BESS dispatch for peak shaving.
    
    BESS discharges when load exceeds threshold.
    BESS charges when load is below threshold and SOC allows.
    
    Args:
        total_load_mw: Total load profile (MW)
        bess_config: BESS configuration
        peak_threshold_mw: Threshold for peak shaving
        
    Returns:
        Tuple of (bess_p_mw, soc) arrays
        bess_p_mw: Positive = discharge, Negative = charge
    """
    n_intervals = len(total_load_mw)
    bess_p = np.zeros(n_intervals)
    soc = np.zeros(n_intervals)
    soc[0] = bess_config.initial_soc
    
    # Calculate energy per interval (assume 15-min intervals)
    interval_hours = 0.25
    
    for t in range(n_intervals):
        # Excess load above threshold
        excess = total_load_mw[t] - peak_threshold_mw
        
        if excess > 0:
            # Discharge to shave peak
            available_energy = (soc[t] - bess_config.min_soc) * bess_config.energy_mwh
            max_discharge = min(
                bess_config.power_mw,
                available_energy / interval_hours / np.sqrt(bess_config.efficiency),
                excess
            )
            bess_p[t] = max(0, max_discharge)
        else:
            # Charge when possible
            headroom_energy = (bess_config.max_soc - soc[t]) * bess_config.energy_mwh
            max_charge = min(
                bess_config.power_mw,
                headroom_energy / interval_hours / np.sqrt(bess_config.efficiency),
                -excess * 0.5  # Charge at half the available margin
            )
            bess_p[t] = -max(0, max_charge)
        
        # Update SOC for next interval
        if t < n_intervals - 1:
            if bess_p[t] > 0:
                # Discharging
                energy_out = bess_p[t] * interval_hours / np.sqrt(bess_config.efficiency)
            else:
                # Charging
                energy_out = bess_p[t] * interval_hours * np.sqrt(bess_config.efficiency)
            
            soc[t + 1] = np.clip(
                soc[t] - energy_out / bess_config.energy_mwh,
                bess_config.min_soc,
                bess_config.max_soc
            )
    
    return bess_p, soc


def generate_bess_dispatch_tou(
    n_intervals: int,
    bess_config: BESSConfig,
    charge_hours: List[int] = None,
    discharge_hours: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate BESS dispatch for Time-of-Use arbitrage.
    
    Charges during off-peak hours, discharges during on-peak.
    
    Args:
        n_intervals: Number of time intervals
        bess_config: BESS configuration
        charge_hours: Hours to charge (default: 0-6)
        discharge_hours: Hours to discharge (default: 17-21)
        
    Returns:
        Tuple of (bess_p_mw, soc) arrays
    """
    if charge_hours is None:
        charge_hours = [0, 1, 2, 3, 4, 5]
    if discharge_hours is None:
        discharge_hours = [17, 18, 19, 20]
    
    intervals_per_hour = n_intervals // 24
    interval_hours = 24.0 / n_intervals
    
    bess_p = np.zeros(n_intervals)
    soc = np.zeros(n_intervals)
    soc[0] = bess_config.initial_soc
    
    for t in range(n_intervals):
        hour = (t // intervals_per_hour) % 24
        
        if hour in charge_hours and soc[t] < bess_config.max_soc:
            # Charge
            headroom = (bess_config.max_soc - soc[t]) * bess_config.energy_mwh
            charge_power = min(bess_config.power_mw, headroom / interval_hours)
            bess_p[t] = -charge_power
            
        elif hour in discharge_hours and soc[t] > bess_config.min_soc:
            # Discharge
            available = (soc[t] - bess_config.min_soc) * bess_config.energy_mwh
            discharge_power = min(bess_config.power_mw, available / interval_hours)
            bess_p[t] = discharge_power
        
        # Update SOC
        if t < n_intervals - 1:
            if bess_p[t] > 0:
                energy_change = -bess_p[t] * interval_hours / np.sqrt(bess_config.efficiency)
            else:
                energy_change = -bess_p[t] * interval_hours * np.sqrt(bess_config.efficiency)
            
            soc[t + 1] = np.clip(
                soc[t] + energy_change / bess_config.energy_mwh,
                bess_config.min_soc,
                bess_config.max_soc
            )
    
    return bess_p, soc


def generate_scenario(config: ScenarioConfig) -> Dict:
    """
    Generate complete scenario data for power flow study.
    
    Returns time-series data for:
    - Load P (MW)
    - Load Q (MVAr)
    - BESS P (MW)
    - BESS SOC
    
    Args:
        config: Scenario configuration
        
    Returns:
        Dictionary with scenario data
    """
    n_intervals = int(config.time_hours * 60 / config.interval_minutes)
    
    # Generate IT load profile based on load type
    if config.load_type == LoadType.TRAINING:
        it_load = generate_training_load_profile(config.base_load_mw, n_intervals)
    elif config.load_type == LoadType.INFERENCE:
        it_load = generate_inference_load_profile(config.base_load_mw, n_intervals)
    else:  # MIXED
        it_load = generate_mixed_load_profile(config.base_load_mw, n_intervals)
    
    # Add cooling load
    cooling_load = generate_cooling_load(it_load, pue=1.4)
    
    # Adjust for architecture efficiency
    efficiency = ARCHITECTURE_EFFICIENCY[config.architecture]
    total_p = (it_load + cooling_load) * efficiency
    
    # Calculate reactive power based on architecture PF
    total_q = calculate_q_from_p(total_p, config.architecture)
    
    # Generate BESS dispatch if configured
    if config.bess_config is not None:
        # Use peak shaving strategy
        peak_threshold = np.percentile(total_p, 75)
        bess_p, bess_soc = generate_bess_dispatch_peak_shaving(
            total_p, config.bess_config, peak_threshold
        )
    else:
        bess_p = np.zeros(n_intervals)
        bess_soc = np.zeros(n_intervals)
    
    # Time array
    time_hours = np.linspace(0, config.time_hours, n_intervals)
    
    return {
        "name": config.name,
        "load_type": config.load_type.value,
        "architecture": config.architecture.value,
        "power_factor": ARCHITECTURE_PF[config.architecture],
        "n_intervals": n_intervals,
        "interval_minutes": config.interval_minutes,
        "time_hours": time_hours.tolist(),
        "it_load_mw": it_load.tolist(),
        "cooling_load_mw": cooling_load.tolist(),
        "total_load_p_mw": total_p.tolist(),
        "total_load_q_mvar": total_q.tolist(),
        "bess_p_mw": bess_p.tolist(),
        "bess_soc": bess_soc.tolist(),
        "bess_config": {
            "power_mw": config.bess_config.power_mw if config.bess_config else 0,
            "energy_mwh": config.bess_config.energy_mwh if config.bess_config else 0,
        } if config.bess_config else None
    }


def create_standard_scenarios(base_load_mw: float = 50.0) -> List[Dict]:
    """
    Create a set of standard planning scenarios.
    
    Creates combinations of:
    - 3 load types (training, inference, mixed)
    - 3 architectures (AC, 48V DC, 800V DC)
    - 3 BESS sizes (None, 10 MW, 25 MW)
    
    Args:
        base_load_mw: Base IT load capacity
        
    Returns:
        List of scenario dictionaries
    """
    scenarios = []
    
    bess_configs = [
        None,
        BESSConfig(power_mw=10, energy_mwh=40),
        BESSConfig(power_mw=25, energy_mwh=100),
    ]
    
    for load_type in LoadType:
        for arch in DCArchitecture:
            for i, bess in enumerate(bess_configs):
                bess_str = "no_bess" if bess is None else f"bess_{bess.power_mw}mw"
                config = ScenarioConfig(
                    name=f"{load_type.value}_{arch.value}_{bess_str}",
                    load_type=load_type,
                    architecture=arch,
                    base_load_mw=base_load_mw,
                    bess_config=bess
                )
                scenarios.append(generate_scenario(config))
    
    return scenarios


if __name__ == "__main__":
    # Test scenario generation
    print("Generating test scenarios...")
    
    # Single scenario test
    config = ScenarioConfig(
        name="test_training_ac",
        load_type=LoadType.TRAINING,
        architecture=DCArchitecture.AC_TRADITIONAL,
        base_load_mw=50.0,
        bess_config=BESSConfig(power_mw=25, energy_mwh=100)
    )
    
    scenario = generate_scenario(config)
    
    print(f"\nScenario: {scenario['name']}")
    print(f"  Load Type: {scenario['load_type']}")
    print(f"  Architecture: {scenario['architecture']}")
    print(f"  Power Factor: {scenario['power_factor']}")
    print(f"  Intervals: {scenario['n_intervals']}")
    print(f"  Peak Load: {max(scenario['total_load_p_mw']):.1f} MW")
    print(f"  Min Load: {min(scenario['total_load_p_mw']):.1f} MW")
    print(f"  Peak Q: {max(scenario['total_load_q_mvar']):.1f} MVAr")
    
    if scenario['bess_config']:
        print(f"  BESS Power: {scenario['bess_config']['power_mw']} MW")
        print(f"  BESS Energy: {scenario['bess_config']['energy_mwh']} MWh")
