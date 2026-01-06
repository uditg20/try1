from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple

import numpy as np

from .models import SizingInputs, SizingOutputs, ElectricalReport
from .profiles import generate_synthetic_profiles
from .simulate import simulate_renewables_bess_grid


def _required_mva(p_mw: float, min_pf: float, margin_pu: float) -> float:
    pf = max(1e-6, float(min_pf))
    return float(abs(p_mw) / pf) * float(margin_pu)


def _size_transformers(
    required_total_mva: float,
    *,
    n_transformers: int,
    redundancy: str,
    primary_kv: float,
    secondary_kv: float,
    per_unit_mva: float | None,
    design_loading_pu: float,
    margin_pu: float,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Returns: (transformer_list, total_mva, firm_mva)
    """
    n = int(n_transformers)
    if n < 1:
        n = 1

    required_total_mva = float(required_total_mva)
    design_loading_pu = max(0.1, float(design_loading_pu))

    if per_unit_mva is None:
        if redundancy == "N+1" and n >= 2:
            # firm_mva = (n-1)*unit_mva
            unit = (required_total_mva / max(1, n - 1)) / design_loading_pu
        else:
            # no redundancy considered
            unit = (required_total_mva / n) / design_loading_pu
        unit *= margin_pu
        per_unit_mva = max(1.0, unit)

    transformers: List[Dict[str, Any]] = []
    for i in range(n):
        transformers.append(
            {
                "name": f"XFMR_{i+1}",
                "mva_rating": float(per_unit_mva),
                "primary_kv": float(primary_kv),
                "secondary_kv": float(secondary_kv),
                "redundancy": redundancy,
            }
        )

    total = float(per_unit_mva) * n
    if redundancy == "N+1" and n >= 2:
        firm = float(per_unit_mva) * (n - 1)
    else:
        firm = total
    return transformers, total, firm


def size_configuration(inputs: SizingInputs) -> SizingOutputs:
    """
    Technical sizing tool (no economics):
    - sizes PV and wind (AC MW) to meet annual CFE energy target (with oversizing if needed)
    - sizes BESS power/energy (grid search) to reduce curtailment and hit CFE with losses
    - screens against grid import/export limits and POI/transformer MVA requirements
    """
    load_mw = float(inputs.load.load_mw)
    dt = float(inputs.profiles.timestep_hours)
    horizon_hours = int(inputs.profiles.horizon_hours)

    prof = generate_synthetic_profiles(
        horizon_hours=horizon_hours,
        timestep_hours=int(dt),
        pv_cf=float(inputs.profiles.pv_capacity_factor),
        wind_cf=float(inputs.profiles.wind_capacity_factor),
        seed=int(inputs.profiles.seed),
    )

    annual_load_mwh = load_mw * horizon_hours
    target_renewable_mwh = float(inputs.cfe_target) * annual_load_mwh

    pv_energy_mwh = target_renewable_mwh * float(inputs.renewable_mix.pv_energy_fraction)
    wind_energy_mwh = target_renewable_mwh * float(inputs.renewable_mix.wind_energy_fraction)

    pv_cf = float(inputs.profiles.pv_capacity_factor)
    wind_cf = float(inputs.profiles.wind_capacity_factor)

    # Initial nameplate sizes (AC MW)
    pv_ac_mw_base = 0.0 if pv_energy_mwh <= 0 else pv_energy_mwh / (pv_cf * horizon_hours)
    wind_mw_base = 0.0 if wind_energy_mwh <= 0 else wind_energy_mwh / (wind_cf * horizon_hours)

    grid_import_limit = float(inputs.grid.max_import_mw)
    grid_export_limit = float(inputs.grid.max_export_mw)

    # Search over renewable oversize factor and BESS sizes to meet achieved CFE
    oversize_factors = [1.0, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.0]

    dur_lo, dur_hi = (float(inputs.bess.duration_hours_bounds[0]), float(inputs.bess.duration_hours_bounds[1]))
    pow_lo, pow_hi = (float(inputs.bess.power_fraction_bounds[0]), float(inputs.bess.power_fraction_bounds[1]))

    # Coarse grids (kept small on purpose)
    durations = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=float)
    durations = durations[(durations >= dur_lo - 1e-9) & (durations <= dur_hi + 1e-9)]
    power_fracs = np.array([0.25, 0.5, 1.0, 1.5, 2.0], dtype=float)
    power_fracs = power_fracs[(power_fracs >= pow_lo - 1e-9) & (power_fracs <= pow_hi + 1e-9)]

    best = None
    best_sim = None

    for f in oversize_factors:
        pv_ac_mw = pv_ac_mw_base * f
        wind_mw = wind_mw_base * f

        for pfrac in power_fracs:
            bess_power = load_mw * float(pfrac)
            for dur in durations:
                bess_energy = bess_power * float(dur)

                sim = simulate_renewables_bess_grid(
                    load_mw=load_mw,
                    pv_ac_mw=pv_ac_mw,
                    wind_mw=wind_mw,
                    pv_pu=prof.pv_pu,
                    wind_pu=prof.wind_pu,
                    dt_hours=dt,
                    grid_import_limit_mw=grid_import_limit,
                    grid_export_limit_mw=grid_export_limit,
                    bess_power_mw=bess_power,
                    bess_energy_mwh=bess_energy,
                    roundtrip_efficiency=float(inputs.bess.roundtrip_efficiency),
                    soc_min=float(inputs.bess.soc_min),
                    soc_max=float(inputs.bess.soc_max),
                    soc_reserve=float(inputs.bess.reserve_soc),
                    initial_soc=float(inputs.bess.initial_soc),
                    allow_grid_charging=bool(inputs.bess.allow_grid_charging),
                )

                if not inputs.allow_unserved_load and sim.unserved_mwh > 1e-6:
                    continue

                if sim.achieved_cfe + 1e-6 < float(inputs.cfe_target):
                    continue

                # score: smallest oversize, then smallest BESS energy, then smallest power
                score = (f, bess_energy, bess_power)
                if best is None or score < best:
                    best = score
                    best_sim = sim
                    best_tuple = (pv_ac_mw, wind_mw, bess_power, bess_energy)

        if best is not None:
            break

    # If we couldn't satisfy CFE without unserved load, pick best achievable (for reporting)
    if best is None:
        # pick a reasonable "max" configuration from bounds for a best-effort report
        pv_ac_mw = pv_ac_mw_base * oversize_factors[-1]
        wind_mw = wind_mw_base * oversize_factors[-1]
        bess_power = load_mw * float(power_fracs[-1]) if power_fracs.size else 0.0
        bess_energy = bess_power * float(durations[-1]) if durations.size else 0.0
        best_sim = simulate_renewables_bess_grid(
            load_mw=load_mw,
            pv_ac_mw=pv_ac_mw,
            wind_mw=wind_mw,
            pv_pu=prof.pv_pu,
            wind_pu=prof.wind_pu,
            dt_hours=dt,
            grid_import_limit_mw=grid_import_limit,
            grid_export_limit_mw=grid_export_limit,
            bess_power_mw=bess_power,
            bess_energy_mwh=bess_energy,
            roundtrip_efficiency=float(inputs.bess.roundtrip_efficiency),
            soc_min=float(inputs.bess.soc_min),
            soc_max=float(inputs.bess.soc_max),
            soc_reserve=float(inputs.bess.reserve_soc),
            initial_soc=float(inputs.bess.initial_soc),
            allow_grid_charging=bool(inputs.bess.allow_grid_charging),
        )
    else:
        pv_ac_mw, wind_mw, bess_power, bess_energy = best_tuple

    # Electrical sizing: POI MVA and transformers
    # Use worst-case of import/export (real power) under PF requirement.
    required_poi_mva = max(
        _required_mva(grid_import_limit, inputs.grid.poi_min_power_factor, inputs.design_margin_pu),
        _required_mva(grid_export_limit, inputs.grid.poi_min_power_factor, inputs.design_margin_pu),
    )
    poi_mva = float(inputs.grid.poi_mva_rating or required_poi_mva)

    # Transformers sized to the same screening requirement by default.
    transformers, total_xfmr_mva, firm_xfmr_mva = _size_transformers(
        required_total_mva=poi_mva,
        n_transformers=inputs.transformer.n_transformers,
        redundancy=inputs.transformer.redundancy,
        primary_kv=inputs.transformer.primary_kv or inputs.grid.poi_voltage_kv,
        secondary_kv=inputs.transformer.secondary_kv,
        per_unit_mva=inputs.transformer.per_unit_mva,
        design_loading_pu=inputs.transformer.design_loading_pu,
        margin_pu=inputs.design_margin_pu,
    )

    # BESS PCS MVA (conservative with inverter PF)
    inverter_pf = float(inputs.bess.inverter_min_pf)
    bess_pcs_mva = 0.0 if bess_power <= 1e-9 else (bess_power / max(1e-6, inverter_pf)) * float(inputs.design_margin_pu)

    # Build warnings/violations
    warnings: List[str] = []
    violations: List[str] = []

    if poi_mva + 1e-9 < required_poi_mva:
        violations.append(
            f"POI MVA rating too small: have {poi_mva:.1f} MVA, need >= {required_poi_mva:.1f} MVA (PF={inputs.grid.poi_min_power_factor})."
        )

    # Firm transformer capacity should cover POI MVA for N+1 (screening)
    if inputs.transformer.redundancy == "N+1" and firm_xfmr_mva + 1e-9 < poi_mva:
        violations.append(
            f"Transformer N+1 firm MVA too small: firm {firm_xfmr_mva:.1f} MVA < POI {poi_mva:.1f} MVA."
        )

    if best_sim.unserved_mwh > 1e-6 and not inputs.allow_unserved_load:
        violations.append(
            f"Unserved load in screening dispatch: {best_sim.unserved_mwh:.2f} MWh. Increase grid limit, renewables, or add firm generation."
        )

    if best_sim.max_grid_import_mw > grid_import_limit + 1e-6:
        violations.append(
            f"Grid import limit violated in screening: max {best_sim.max_grid_import_mw:.2f} MW > limit {grid_import_limit:.2f} MW."
        )

    if best_sim.achieved_cfe + 1e-6 < float(inputs.cfe_target):
        warnings.append(
            f"CFE target not met in screening: achieved {best_sim.achieved_cfe*100:.1f}% < target {inputs.cfe_target*100:.1f}%."
        )

    if bess_energy > 0 and (bess_energy / max(1e-9, bess_power)) < 1.0:
        warnings.append("BESS duration < 1 hour; may be insufficient for overnight shifting depending on site.")

    feasible = len(violations) == 0

    report = ElectricalReport(
        feasible=feasible,
        warnings=warnings,
        violations=violations,
        max_grid_import_mw=float(best_sim.max_grid_import_mw),
        grid_import_limit_mw=grid_import_limit,
        max_grid_export_mw=float(best_sim.max_grid_export_mw),
        grid_export_limit_mw=grid_export_limit,
        required_poi_mva=float(required_poi_mva),
        required_transformer_mva_total=float(poi_mva),
        firm_transformer_mva=float(firm_xfmr_mva),
        achieved_cfe=float(best_sim.achieved_cfe),
        annual_load_mwh=float(best_sim.annual_load_mwh),
        annual_grid_import_mwh=float(best_sim.annual_grid_import_mwh),
        annual_renewable_used_mwh=float(best_sim.annual_renewable_used_mwh),
        annual_renewable_curtailed_mwh=float(best_sim.annual_renewable_curtailed_mwh),
        annual_bess_throughput_mwh=float(best_sim.annual_bess_throughput_mwh),
    )

    # Emit a "case" structure aligned with the repo's JSON conventions.
    # Note: dispatch engine doesn't yet optimize PV/wind; this is a *configuration output*.
    case_json: Dict[str, Any] = {
        "name": inputs.name,
        "iso": "GENERIC",
        "time": {"horizon_hours": 24, "interval_minutes": 60},
        "topology": {
            "poi": {
                "name": "POI",
                "max_import_mw": grid_import_limit,
                "max_export_mw": grid_export_limit,
                "mva_rating": poi_mva,
                "min_power_factor": float(inputs.grid.poi_min_power_factor),
                "voltage_kv": float(inputs.grid.poi_voltage_kv),
            },
            "transformers": transformers,
        },
        "resources": {
            "loads": [
                {
                    "name": "DataCenterLoad",
                    "type": "critical",
                    "base_mw": load_mw,
                    "power_factor": float(inputs.load.power_factor),
                }
            ],
            "bess": {
                "name": "BESS",
                "power_mw": float(bess_power),
                "energy_mwh": float(bess_energy),
                "pcs_mva": float(bess_pcs_mva),
                "efficiency_charge": float(np.sqrt(inputs.bess.roundtrip_efficiency)),
                "efficiency_discharge": float(np.sqrt(inputs.bess.roundtrip_efficiency)),
                "soc_min": float(inputs.bess.soc_min),
                "soc_max": float(inputs.bess.soc_max),
                "soc_reserve": float(inputs.bess.reserve_soc),
                "initial_soc": float(inputs.bess.initial_soc),
                "degradation_cost_per_mwh": 0.0,
            },
            "renewables": {
                "pv": {
                    "ac_mw": float(pv_ac_mw),
                    "profile": "synthetic",
                    "capacity_factor": float(inputs.profiles.pv_capacity_factor),
                },
                "wind": {
                    "mw": float(wind_mw),
                    "profile": "synthetic",
                    "capacity_factor": float(inputs.profiles.wind_capacity_factor),
                },
            },
        },
        "planning_targets": {"cfe_target": float(inputs.cfe_target)},
        "notes": [
            "This case JSON is a sizing output. The current dispatch MILP in this repo does not yet co-optimize PV/wind.",
            "Replace synthetic profiles with site-specific time series for final engineering.",
        ],
    }

    return SizingOutputs(
        inputs=inputs,
        pv_ac_mw=float(pv_ac_mw),
        wind_mw=float(wind_mw),
        bess_power_mw=float(bess_power),
        bess_energy_mwh=float(bess_energy),
        bess_pcs_mva=float(bess_pcs_mva),
        poi_mva_rating=float(poi_mva),
        transformers=transformers,
        case_json=case_json,
        report=report,
    )

