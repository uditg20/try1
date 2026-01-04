"""
Rule-based (shadow) dispatch for feasibility screening.

Dispatch intent:
- Not economics-driven
- No closed-loop controls
- Conservative, explainable logic

Method:
- Propose a candidate BESS P setpoint based on load level
- Screen it by running a trial power flow and verifying:
  - Voltage is within [0.95, 1.05] pu
  - Transformer loading is below a threshold (e.g. 90%)
  - Load ramp is below a threshold
- If it fails screening, back off the BESS setpoint toward zero
"""

from __future__ import annotations

from dataclasses import dataclass

import pandapower as pp

from network import run_powerflow, summarize_base_results
from scenarios import ArchitectureAssumptions
from network import BaseSystemSpec


@dataclass(frozen=True)
class DispatchLimits:
    bess_p_min_mw: float
    bess_p_max_mw: float
    bess_q_min_mvar: float
    bess_q_max_mvar: float


def _screen_operating_point(
    net: pp.pandapowerNet,
    *,
    vmin_pu: float,
    vmax_pu: float,
    trafo_loading_max_pct: float,
) -> tuple[bool, str]:
    run_powerflow(net)
    s = summarize_base_results(net)

    vm = float(s["bus"].loc["DATA_CENTER_34p5kV", "vm_pu"])
    loading = float(s["trafo"].iloc[0]["loading_percent"])

    if vm < vmin_pu or vm > vmax_pu:
        return False, "blocked_voltage"
    if loading >= trafo_loading_max_pct:
        return False, "blocked_transformer_loading"
    return True, "ok"


def dispatch_bess_rule_based(
    *,
    net: pp.pandapowerNet,
    system: BaseSystemSpec,
    arch: ArchitectureAssumptions,
    load_p_mw: float,
    load_ramp_mw_per_h: float | None,
    limits: DispatchLimits,
    vmin_pu: float,
    vmax_pu: float,
    trafo_loading_max_pct: float,
    load_ramp_max_mw_per_h: float,
) -> tuple[float, float, str]:
    """
    Conservative shadow dispatch:
    - If load ramp exceeds threshold: do not dispatch (hold BESS at 0 MW / 0 MVAr).
    - Otherwise: attempt a modest discharge during higher-load hours and a modest charge during lower-load hours.
    - Screen the candidate by running PF; if it violates constraints, reduce toward zero.

    Returns: (bess_p_mw, bess_q_mvar, decision_reason)
    """

    # Ramp gating (conceptual: avoid compounding fast facility ramps with BESS actions).
    if load_ramp_mw_per_h is not None and abs(load_ramp_mw_per_h) > load_ramp_max_mw_per_h:
        return 0.0, 0.0, "blocked_ramp"

    # Candidate setpoint is intentionally simple and explainable:
    # - Discharge when load is above baseline
    # - Charge when load is below baseline
    # This is NOT meant to optimize; it simply exercises feasibility under plausible dispatch directions.
    delta = load_p_mw - system.baseline_load_p_mw
    candidate = 0.0
    if delta > 0:
        candidate = min(limits.bess_p_max_mw, 0.5 * delta)  # discharge up to half the deviation
    elif delta < 0:
        candidate = max(limits.bess_p_min_mw, 0.5 * delta)  # charge (negative P)

    # Reactive setpoint: keep at 0 MVAr by default to avoid implying voltage control.
    # Reactive capability is represented as bounds for future sensitivity studies.
    q_set = 0.0
    q_set = float(max(limits.bess_q_min_mvar, min(limits.bess_q_max_mvar, q_set)))

    # Screen by backing off toward zero if needed.
    # We evaluate a sequence: full candidate -> 75% -> 50% -> 25% -> 0
    steps = [1.0, 0.75, 0.50, 0.25, 0.0]

    for k in steps:
        p_try = float(k * candidate)

        # Apply setpoints (mutate net in-place)
        net.sgen.loc[:, "p_mw"] = p_try
        net.sgen.loc[:, "q_mvar"] = q_set

        ok, reason = _screen_operating_point(
            net,
            vmin_pu=vmin_pu,
            vmax_pu=vmax_pu,
            trafo_loading_max_pct=trafo_loading_max_pct,
        )
        if ok:
            if p_try == 0.0:
                return 0.0, 0.0, "idle_ok"
            return p_try, q_set, "dispatched_ok"

        # If the candidate violates constraints, continue backing off.
        last_reason = reason

    return 0.0, 0.0, last_reason if "last_reason" in locals() else "blocked_unknown"

