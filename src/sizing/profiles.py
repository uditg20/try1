from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Profiles:
    pv_pu: np.ndarray    # 0..1, mean ~= pv_cf after scaling
    wind_pu: np.ndarray  # 0..1, mean ~= wind_cf after scaling
    load_pu: np.ndarray  # 1.0 flat for now


def _scale_to_capacity_factor(pu: np.ndarray, target_cf: float) -> np.ndarray:
    pu = np.clip(pu, 0.0, 1.0)
    mean = float(np.mean(pu)) if pu.size else 0.0
    if mean <= 1e-9:
        return np.zeros_like(pu)
    scaled = pu * (target_cf / mean)
    return np.clip(scaled, 0.0, 1.0)


def generate_synthetic_profiles(
    horizon_hours: int = 8760,
    timestep_hours: int = 1,
    pv_cf: float = 0.25,
    wind_cf: float = 0.40,
    seed: int = 42,
) -> Profiles:
    """
    Generate simple synthetic hourly profiles for PV and wind.

    Notes:
    - PV: diurnal sine-shaped generation with mild seasonality.
    - Wind: seasonal + stochastic variability with autocorrelation.
    - These are *screening* profiles for sizing; for final design, replace with
      site-specific meteorological data.
    """
    if horizon_hours % timestep_hours != 0:
        raise ValueError("horizon_hours must be divisible by timestep_hours")

    n = horizon_hours // timestep_hours
    t = np.arange(n, dtype=float)

    # Hour-of-day and day-of-year indices
    hours = (t * timestep_hours) % 24.0
    days = (t * timestep_hours) / 24.0

    # --- PV (diurnal bell curve + seasonality) ---
    # daylight proxy: sunrise ~6, sunset ~18
    solar_angle = math.pi * (hours - 6.0) / 12.0
    diurnal = np.maximum(0.0, np.sin(solar_angle))
    diurnal = diurnal**1.8  # sharpen midday peak a bit

    # seasonality: +/-15% over year
    seasonal = 1.0 + 0.15 * np.sin(2 * math.pi * days / 365.0 - math.pi / 2)
    pv_raw = diurnal * seasonal
    pv_pu = _scale_to_capacity_factor(pv_raw, pv_cf)

    # --- Wind (seasonal + AR(1) noise) ---
    rng = np.random.default_rng(seed)
    seasonal_w = 0.55 + 0.15 * np.sin(2 * math.pi * days / 365.0 + math.pi / 3)
    daily_w = 0.05 * np.sin(2 * math.pi * hours / 24.0 + math.pi / 5)
    base = seasonal_w + daily_w

    # AR(1) noise for persistence
    eps = rng.normal(0, 0.12, size=n)
    ar = np.zeros(n, dtype=float)
    phi = 0.90
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + eps[i]
    wind_raw = base + ar
    wind_pu = _scale_to_capacity_factor(wind_raw, wind_cf)

    load_pu = np.ones(n, dtype=float)
    return Profiles(pv_pu=pv_pu, wind_pu=wind_pu, load_pu=load_pu)

