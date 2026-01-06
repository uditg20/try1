from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DispatchSimResult:
    achieved_cfe: float
    annual_load_mwh: float
    annual_grid_import_mwh: float
    annual_renewable_used_mwh: float
    annual_renewable_curtailed_mwh: float
    annual_bess_throughput_mwh: float
    max_grid_import_mw: float
    max_grid_export_mw: float
    unserved_mwh: float


def simulate_renewables_bess_grid(
    *,
    load_mw: float,
    pv_ac_mw: float,
    wind_mw: float,
    pv_pu: np.ndarray,
    wind_pu: np.ndarray,
    dt_hours: float,
    grid_import_limit_mw: float,
    grid_export_limit_mw: float,
    bess_power_mw: float,
    bess_energy_mwh: float,
    roundtrip_efficiency: float,
    soc_min: float,
    soc_max: float,
    soc_reserve: float,
    initial_soc: float,
    allow_grid_charging: bool,
) -> DispatchSimResult:
    """
    Greedy technical dispatch to screen CFE feasibility.

    Rules (simple but electrically consistent for screening):
    - Serve load with renewables first.
    - Charge BESS with excess renewables (and optionally with grid).
    - Discharge BESS to cover deficits before importing from grid.
    - Export is not used by default; if export limit > 0, we export only after BESS is full.

    CFE accounting:
    - All PV/wind generation is considered CFE.
    - BESS discharge is considered CFE only insofar as it was charged from renewables.
      (If allow_grid_charging=True, the simulation still *operates* but CFE accounting
       for BESS becomes ambiguous; we conservatively treat grid-charged energy as non-CFE.)
    """
    n = int(len(pv_pu))
    if len(wind_pu) != n:
        raise ValueError("pv_pu and wind_pu must have the same length")

    if bess_power_mw <= 1e-9 or bess_energy_mwh <= 1e-9:
        bess_power_mw = 0.0
        bess_energy_mwh = 0.0

    eta = float(np.sqrt(roundtrip_efficiency)) if roundtrip_efficiency > 0 else 0.0
    eta_c = eta if eta > 0 else 1.0
    eta_d = eta if eta > 0 else 1.0

    # SOC bounds (in MWh)
    soc_min_mwh = soc_min * bess_energy_mwh
    soc_max_mwh = soc_max * bess_energy_mwh
    soc_floor_mwh = max(soc_min_mwh, soc_reserve * bess_energy_mwh)

    soc = float(np.clip(initial_soc, 0.0, 1.0)) * bess_energy_mwh

    annual_load_mwh = load_mw * n * dt_hours
    grid_import_mwh = 0.0
    renewable_used_mwh = 0.0
    renewable_curtailed_mwh = 0.0
    bess_throughput_mwh = 0.0
    unserved_mwh = 0.0

    max_grid_import = 0.0
    max_grid_export = 0.0

    # Track how much energy in the battery is "renewable-sourced" (MWh)
    # For allow_grid_charging=False, this will simply equal SOC (above floor) up to losses.
    ren_soc = min(soc, soc_max_mwh)

    for i in range(n):
        pv = pv_ac_mw * float(pv_pu[i])
        wind = wind_mw * float(wind_pu[i])
        ren = pv + wind

        # Serve load directly from renewables (instantaneous)
        ren_to_load = min(load_mw, ren)
        renewable_used_mwh += ren_to_load * dt_hours

        remaining_load = load_mw - ren_to_load
        excess_ren = ren - ren_to_load

        # Charge BESS from excess renewables
        if bess_power_mw > 0:
            charge_power_limit = bess_power_mw
            energy_headroom_mwh = max(0.0, soc_max_mwh - soc)
            # charging power limited by headroom and efficiency
            charge_mw_headroom = energy_headroom_mwh / (dt_hours * eta_c) if dt_hours > 0 else 0.0
            charge_mw = min(excess_ren, charge_power_limit, charge_mw_headroom)
            if charge_mw > 0:
                charge_mwh_in = charge_mw * dt_hours
                stored_mwh = charge_mwh_in * eta_c
                soc += stored_mwh
                # All charging energy here is renewable-sourced
                ren_soc = min(soc, ren_soc + stored_mwh)
                bess_throughput_mwh += charge_mwh_in
                excess_ren -= charge_mw

        # Optional grid charging (only if requested and there's headroom)
        if allow_grid_charging and bess_power_mw > 0 and soc < soc_max_mwh - 1e-6:
            # only charge from grid if there is remaining grid capacity after serving load
            available_import_cap = max(0.0, grid_import_limit_mw)
            # (we haven't imported yet this hour; we will import for remaining load below)
            # For screening, don't do any proactive grid charging.
            pass

        # Discharge BESS to cover remaining load
        if remaining_load > 0 and bess_power_mw > 0:
            discharge_power_limit = bess_power_mw
            available_energy_mwh = max(0.0, soc - soc_floor_mwh)
            discharge_mw_energy = (available_energy_mwh * eta_d) / dt_hours if dt_hours > 0 else 0.0
            discharge_mw = min(remaining_load, discharge_power_limit, discharge_mw_energy)
            if discharge_mw > 0:
                discharge_mwh_out = discharge_mw * dt_hours
                soc_delta = discharge_mwh_out / eta_d
                soc -= soc_delta
                # Determine how much of discharge is renewable-sourced
                if allow_grid_charging:
                    # Conservative: only credit discharge up to ren_soc energy fraction
                    ren_available = max(0.0, ren_soc - soc_floor_mwh)
                    ren_discharge_mwh = min(discharge_mwh_out, ren_available * eta_d)
                    renewable_used_mwh += ren_discharge_mwh
                    ren_soc = max(soc_floor_mwh, ren_soc - soc_delta)
                else:
                    renewable_used_mwh += discharge_mwh_out
                    ren_soc = min(ren_soc, soc)  # keep consistent
                bess_throughput_mwh += discharge_mwh_out
                remaining_load -= discharge_mw

        # Import remaining from grid
        grid_import = min(remaining_load, grid_import_limit_mw)
        if grid_import > 0:
            grid_import_mwh += grid_import * dt_hours
            remaining_load -= grid_import
        max_grid_import = max(max_grid_import, grid_import)

        # Unserved if still remaining
        if remaining_load > 1e-9:
            unserved_mwh += remaining_load * dt_hours

        # Export or curtail leftover renewables
        export = 0.0
        if excess_ren > 1e-9 and grid_export_limit_mw > 0:
            export = min(excess_ren, grid_export_limit_mw)
            excess_ren -= export
        max_grid_export = max(max_grid_export, export)
        renewable_curtailed_mwh += max(0.0, excess_ren) * dt_hours

    achieved_cfe = 0.0 if annual_load_mwh <= 0 else float(renewable_used_mwh / annual_load_mwh)
    return DispatchSimResult(
        achieved_cfe=achieved_cfe,
        annual_load_mwh=annual_load_mwh,
        annual_grid_import_mwh=grid_import_mwh,
        annual_renewable_used_mwh=renewable_used_mwh,
        annual_renewable_curtailed_mwh=renewable_curtailed_mwh,
        annual_bess_throughput_mwh=bess_throughput_mwh,
        max_grid_import_mw=float(max_grid_import),
        max_grid_export_mw=float(max_grid_export),
        unserved_mwh=unserved_mwh,
    )

