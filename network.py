"""
Planning-level pandapower network model (positive-sequence / RMS).
===============================================================

Scope:
- Balanced three-phase, steady-state power flow only (RMS / positive-sequence abstraction).
- Utility-style feasibility screening: voltages, transformer loading, grid P/Q import.

Out of scope (by design):
- EMT / switching dynamics, inverter firmware, PLL behavior
- Protection coordination, harmonics, flicker
- Unbalanced / negative-sequence / zero-sequence effects
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, tan
from typing import Literal

import pandapower as pp


PowerFactorSign = Literal["lagging", "leading"]


@dataclass(frozen=True)
class TransformerSpec:
    """
    Two-winding transformer parameters suitable for planning studies.

    Notes:
    - `vk_percent` is the short-circuit impedance magnitude in percent on transformer base.
    - `vkr_percent` is the resistive component in percent.
    - Values below are *typical* for HV/MV GSUs; replace with interconnection-quality data
      when available. Keep conservative when uncertain.
    """

    sn_mva: float = 150.0
    vn_hv_kv: float = 230.0
    vn_lv_kv: float = 34.5
    vk_percent: float = 10.5
    vkr_percent: float = 0.35
    pfe_kw: float = 80.0
    i0_percent: float = 0.15
    shift_degree: float = 0.0


@dataclass(frozen=True)
class BaseSystemSpec:
    """Single-transformer grid-to-data-center electrical model."""

    grid_v_kv: float = 230.0
    dc_v_kv: float = 34.5
    trafo: TransformerSpec = TransformerSpec()

    # Planning-level defaults (examples; adjust to match project)
    baseline_load_p_mw: float = 50.0
    baseline_load_pf: float = 0.97  # training-heavy default
    baseline_load_pf_sign: PowerFactorSign = "lagging"

    bess_p_max_mw: float = 30.0
    bess_p_min_mw: float = -30.0  # negative means charging (net consumption)
    bess_q_max_mvar: float = 15.0
    bess_q_min_mvar: float = -15.0


def q_from_pf(p_mw: float, pf: float, sign: PowerFactorSign = "lagging") -> float:
    """
    Convert P and PF to Q using Q = P * tan(arccos(PF)).

    Conventions used:
    - Load element: positive P/Q represent consumption (inductive Q > 0 for lagging PF).
    - BESS `sgen`: positive P/Q represent injection.
    """

    if pf <= 0.0 or pf > 1.0:
        raise ValueError(f"Power factor must be in (0, 1], got pf={pf}")

    # pf can be 1.0 -> acos(1)=0 -> Q=0
    q_mag = abs(p_mw) * tan(acos(pf))
    if sign == "lagging":
        return q_mag
    if sign == "leading":
        return -q_mag
    raise ValueError(f"Unknown PF sign: {sign}")


def build_base_network(
    spec: BaseSystemSpec = BaseSystemSpec(),
    *,
    load_p_mw: float | None = None,
    load_pf: float | None = None,
    load_pf_sign: PowerFactorSign | None = None,
    bess_p_mw: float = 0.0,
    bess_q_mvar: float = 0.0,
) -> pp.pandapowerNet:
    """
    Build the base planning-level pandapower network:
    Grid (slack) -> HV/MV transformer -> data center MV bus -> load and BESS (PQ).
    """

    load_p = spec.baseline_load_p_mw if load_p_mw is None else float(load_p_mw)
    pf = spec.baseline_load_pf if load_pf is None else float(load_pf)
    pf_sign = spec.baseline_load_pf_sign if load_pf_sign is None else load_pf_sign
    load_q = q_from_pf(load_p, pf, pf_sign)

    net = pp.create_empty_network(sn_mva=spec.trafo.sn_mva)

    # Buses
    bus_grid = pp.create_bus(net, vn_kv=spec.grid_v_kv, name="GRID_230kV")
    bus_dc = pp.create_bus(net, vn_kv=spec.dc_v_kv, name="DATA_CENTER_34p5kV")

    # Grid infinite source (utility / transmission system)
    pp.create_ext_grid(net, bus=bus_grid, vm_pu=1.02, name="GRID_SLACK")

    # HV/MV transformer (two-winding)
    pp.create_transformer_from_parameters(
        net,
        hv_bus=bus_grid,
        lv_bus=bus_dc,
        sn_mva=spec.trafo.sn_mva,
        vn_hv_kv=spec.trafo.vn_hv_kv,
        vn_lv_kv=spec.trafo.vn_lv_kv,
        vk_percent=spec.trafo.vk_percent,
        vkr_percent=spec.trafo.vkr_percent,
        pfe_kw=spec.trafo.pfe_kw,
        i0_percent=spec.trafo.i0_percent,
        shift_degree=spec.trafo.shift_degree,
        name="XFMR_230_34p5kV",
        tap_side="hv",
        tap_neutral=0,
        tap_min=-10,
        tap_max=10,
        tap_step_percent=1.25,
    )

    # Aggregated data center load (planning PQ representation)
    pp.create_load(
        net,
        bus=bus_dc,
        p_mw=load_p,
        q_mvar=load_q,
        name="DATA_CENTER_LOAD",
    )

    # BESS inverter as PQ-controlled injection (sgen). Dispatch set externally.
    sgen_idx = pp.create_sgen(
        net,
        bus=bus_dc,
        p_mw=float(bess_p_mw),
        q_mvar=float(bess_q_mvar),
        name="BESS_INVERTER",
    )
    # Store capability bounds explicitly (pandapower PF does not enforce these).
    net.sgen.loc[sgen_idx, "p_mw_min"] = spec.bess_p_min_mw
    net.sgen.loc[sgen_idx, "p_mw_max"] = spec.bess_p_max_mw
    net.sgen.loc[sgen_idx, "q_mvar_min"] = spec.bess_q_min_mvar
    net.sgen.loc[sgen_idx, "q_mvar_max"] = spec.bess_q_max_mvar

    return net


def run_powerflow(net: pp.pandapowerNet) -> None:
    """Run a single power flow with standard options."""

    # `numba=False` avoids optional dependency warnings and keeps execution reproducible.
    pp.runpp(net, algorithm="nr", init="auto", enforce_q_lims=False, numba=False)


def summarize_base_results(net: pp.pandapowerNet) -> dict:
    """
    Extract a concise, utility-style summary:
    - Bus voltages (pu)
    - Transformer loading (%)
    - Grid import (MW/MVAr)
    """

    if net.get("res_bus", None) is None or net.res_bus.empty:
        raise RuntimeError("No power flow results found. Run `run_powerflow(net)` first.")

    bus = net.res_bus[["vm_pu", "va_degree"]].copy()
    bus.index = net.bus["name"].astype(str)

    trafo = net.res_trafo[["loading_percent", "p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar"]].copy()
    trafo.index = net.trafo["name"].astype(str)

    grid = net.res_ext_grid[["p_mw", "q_mvar"]].copy()
    grid.index = net.ext_grid["name"].astype(str)

    return {"bus": bus, "trafo": trafo, "grid": grid}

