"""
Scenario definitions for planning-level feasibility screening.

This file intentionally represents "architecture" effects only via:
- power factor (PF)
- reactive variability (conceptual volatility, realized as bounded noise)
- effective ramp severity (how quickly load changes hour-to-hour)

Topology is held fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from network import BaseSystemSpec


@dataclass(frozen=True)
class ArchitectureAssumptions:
    """
    Represents AC-side electrical behavior assumptions without modeling internal DC systems.

    Parameters are screening-level and should be aligned with facility studies when available.
    """

    name: str
    description: str

    # PF target range (lagging assumed unless explicitly changed)
    pf_min: float
    pf_max: float
    pf_sign: str = "lagging"

    # Conceptual Q volatility: implemented as PF jitter within range, plus optional Q noise
    pf_jitter_sigma: float = 0.002

    # Effective ramp severity multiplier on the baseline profile ramps (>=1 is "more spiky").
    ramp_severity: float = 1.0

    # BESS reactive capability assumed for that architecture (screening-level).
    bess_q_min_mvar: float = -15.0
    bess_q_max_mvar: float = 15.0


DEFAULT_ARCHITECTURES: dict[str, ArchitectureAssumptions] = {
    # NVIDIA-style "architecture" representations as *assumptions*, not topology changes.
    "ac_racks": ArchitectureAssumptions(
        name="AC racks (lower PF, higher Q volatility)",
        description="Represents more reactive variability and slightly lower PF.",
        pf_min=0.94,
        pf_max=0.96,
        pf_jitter_sigma=0.004,
        ramp_severity=1.15,
        bess_q_min_mvar=-15.0,
        bess_q_max_mvar=15.0,
    ),
    "dc_48v": ArchitectureAssumptions(
        name="48V DC (higher PF, lower Q volatility)",
        description="Represents improved PF and reduced reactive variability.",
        pf_min=0.97,
        pf_max=0.985,
        pf_jitter_sigma=0.002,
        ramp_severity=1.05,
        bess_q_min_mvar=-12.0,
        bess_q_max_mvar=12.0,
    ),
    "dc_800v": ArchitectureAssumptions(
        name="800V DC (near-unity PF, minimal Q volatility)",
        description="Represents near-unity PF and minimal reactive variability.",
        pf_min=0.993,
        pf_max=0.997,
        pf_jitter_sigma=0.0008,
        ramp_severity=1.0,
        bess_q_min_mvar=-8.0,
        bess_q_max_mvar=8.0,
    ),
}


@dataclass(frozen=True)
class OperatingModeAssumptions:
    """
    Facility operating mode represented only by PF envelope assumptions.

    This intentionally does NOT model internal electrical topology (UPS, rectifiers, etc.).
    """

    name: str
    pf_min: float
    pf_max: float
    pf_sign: str = "lagging"


DEFAULT_OPERATING_MODES: dict[str, OperatingModeAssumptions] = {
    "training": OperatingModeAssumptions(name="training-heavy", pf_min=0.97, pf_max=0.97, pf_sign="lagging"),
    "inference": OperatingModeAssumptions(name="inference-heavy", pf_min=0.94, pf_max=0.96, pf_sign="lagging"),
}


def _clip_pf(pf: float) -> float:
    return float(np.clip(pf, 0.80, 1.0))


def default_hourly_profile(
    *,
    idx: pd.DatetimeIndex,
    system: BaseSystemSpec,
    arch: ArchitectureAssumptions,
    seed: int = 42,
    mode: str = "mixed",
) -> pd.DataFrame:
    """
    Create an hourly "training vs inference" load profile for screening.

    Conservative intent:
    - We avoid inventing detailed IT-level behavior.
    - We use a simple diurnal variation with a modest step to represent "training-heavy" hours.
    - Architecture affects PF range and ramp severity only.
    """

    rng = np.random.default_rng(seed)
    hours = idx.hour.to_numpy()

    # Baseline 50 MW with modest diurnal shaping.
    base = system.baseline_load_p_mw
    diurnal = 1.0 + 0.06 * np.sin(2 * np.pi * (hours - 7) / 24.0)

    # "Training-heavy" windows: higher sustained utilization.
    training_hours = (hours >= 18) | (hours <= 2)
    training_adder = np.where(training_hours, 0.12, 0.0)  # +12% as a scenario knob

    p = base * diurnal * (1.0 + training_adder)

    # Add mild noise and apply ramp severity (more spiky architectures get more hour-to-hour variation).
    noise = rng.normal(0.0, 0.8, size=len(idx))  # MW
    p = p + arch.ramp_severity * noise
    p = np.maximum(0.0, p)

    # PF is treated as an assumption envelope (no internal device modeling).
    #
    # Required defaults:
    # - training-heavy: PF = 0.97
    # - inference-heavy: PF = 0.94–0.96
    #
    # Architecture assumptions can tighten/shift this envelope (Step 6).
    if mode not in {"mixed", "training", "inference"}:
        raise ValueError(f"mode must be one of mixed/training/inference, got: {mode}")

    training_mode = DEFAULT_OPERATING_MODES["training"]
    inference_mode = DEFAULT_OPERATING_MODES["inference"]

    if mode == "training":
        mode_pf_min = np.full(len(idx), training_mode.pf_min)
        mode_pf_max = np.full(len(idx), training_mode.pf_max)
        mode_sign = training_mode.pf_sign
    elif mode == "inference":
        mode_pf_min = np.full(len(idx), inference_mode.pf_min)
        mode_pf_max = np.full(len(idx), inference_mode.pf_max)
        mode_sign = inference_mode.pf_sign
    else:
        mode_pf_min = np.where(training_hours, training_mode.pf_min, inference_mode.pf_min)
        mode_pf_max = np.where(training_hours, training_mode.pf_max, inference_mode.pf_max)
        mode_sign = "lagging"

    pf_center = 0.5 * (mode_pf_min + mode_pf_max)
    pf_jitter = rng.normal(0.0, arch.pf_jitter_sigma, size=len(idx))
    pf = pf_center + pf_jitter
    # Apply operating-mode PF envelope first, then apply architecture envelope (intersection via clipping).
    pf = np.clip(pf, mode_pf_min, mode_pf_max)
    pf = np.clip(pf, arch.pf_min, arch.pf_max)
    pf = np.vectorize(_clip_pf)(pf)

    return pd.DataFrame(
        {
            "load_p_mw": p,
            "load_pf": pf,
            "load_pf_sign": mode_sign,
        },
        index=idx,
    )

