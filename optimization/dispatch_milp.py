from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo

from iso.base import ISOAdapter, ReserveProduct
from topology.screening import linear_mva_facets


@dataclass(frozen=True)
class DispatchBuildInputs:
    time: List[str]
    dt_hours: float
    big_m: float
    mva_facets: List[Tuple[float, float]]


def _tan_from_pf(pf_min: float) -> float:
    if pf_min <= 0 or pf_min > 1.0:
        raise ValueError("pf_min must be in (0, 1]")
    return float(np.tan(np.arccos(pf_min)))


def build_dispatch_model(
    *,
    adapter: ISOAdapter,
    case: dict,
    topology: Any,
    bess: Any,
    gas_gen: Any,
    load: Any,
    solver_facets: int = 12,
) -> Tuple[pyo.ConcreteModel, DispatchBuildInputs]:
    """
    Build the production-grade microgrid dispatch MILP (Pyomo).

    Topology (explicit, minimal):
      Grid POI -> Transformer bank -> Microgrid bus (resources + load)

    Electrical limits (linearized):
    - POI apparent power (MVA) via polygon facets
    - POI minimum PF via |Q| <= tan(arccos(pf_min))*|P|
    - Transformer bank MVA screening (N+1 effective capacity) via same facets
    - BESS PCS MVA via same facets using |P|, |Q|
    """

    topology.validate()
    bess.validate()
    gas_gen.validate()
    load.validate()
    adapter.validate_case(case)

    time = case["time"]
    if len(time) != len(load.p_critical_mw):
        raise ValueError("case.time length must match load series length")

    dt_min = case.get("dispatch_interval_minutes", adapter.dispatch_interval_minutes())
    dt_hours = float(dt_min) / 60.0

    facets = linear_mva_facets(solver_facets)
    big_m = float(case.get("big_m", adapter.default_big_m()))

    m = pyo.ConcreteModel("microgrid_dispatch_milp")
    m.T = pyo.RangeSet(0, len(time) - 1)
    m.R = pyo.Set(initialize=[rp.name for rp in adapter.reserve_products()], ordered=True)

    # -----------------------------
    # Parameters (time series)
    # -----------------------------
    energy_key = adapter.energy_price_series_key()
    if energy_key not in case:
        raise ValueError(f"case missing required energy price series: {energy_key}")
    if len(case[energy_key]) != len(time):
        raise ValueError("energy price series length must match time")

    reserve_keys = adapter.reserve_price_series_keys()
    reserve_price = {}
    for r in m.R:
        k = reserve_keys.get(r)
        if k is None:
            raise ValueError(f"adapter missing reserve price series key for product {r}")
        if k not in case:
            raise ValueError(f"case missing required reserve price series: {k}")
        if len(case[k]) != len(time):
            raise ValueError(f"reserve price series {k} length must match time")
        reserve_price[r] = case[k]

    voll_critical = float(case.get("voll_critical_$per_mwh", 10000.0))
    voll_noncritical = float(case.get("voll_noncritical_$per_mwh", 1000.0))
    allow_export = bool(case.get("allow_export", True))
    ride_through_minutes = int(case.get("ride_through_minutes", max(0, gas_gen.start_time_minutes)))

    # Reliability reserve energy requirement: ride-through for critical load
    # SOC >= market reserve + (critical_load * ride_through_hours)/eta_discharge
    ride_through_hours = float(ride_through_minutes) / 60.0

    # -----------------------------
    # Decision variables
    # -----------------------------
    # Grid import/export (MW)
    m.p_grid_import = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_grid_export = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # POI reactive power (MVAr) via pos/neg
    m.q_poi_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.q_poi_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # BESS power (MW) split, plus no-simultaneous binary
    m.p_bess_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_bess_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.z_bess_ch = pyo.Var(m.T, domain=pyo.Binary)
    m.z_bess_dis = pyo.Var(m.T, domain=pyo.Binary)

    # BESS reactive (MVAr) via pos/neg
    m.q_bess_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.q_bess_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # BESS SOC (MWh)
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Generator commitment as integer units + start/stop with explicit start delay
    m.n_on = pyo.Var(m.T, domain=pyo.NonNegativeIntegers, bounds=(0, gas_gen.n_units))
    m.n_start = pyo.Var(m.T, domain=pyo.NonNegativeIntegers, bounds=(0, gas_gen.n_units))
    m.n_stop = pyo.Var(m.T, domain=pyo.NonNegativeIntegers, bounds=(0, gas_gen.n_units))
    m.n_available = pyo.Var(m.T, domain=pyo.NonNegativeIntegers, bounds=(0, gas_gen.n_units))

    # Generator active and reactive
    m.p_gen = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.q_gen_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.q_gen_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Load served/unserved (MW)
    m.p_crit_served = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_noncrit_served = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_crit_unserved = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_noncrit_unserved = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Curtailment (non-critical) (MW)
    m.p_curtail = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Reserves (MW) per product, split by resource for explainability
    m.res_bess = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)
    m.res_gen = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals)

    # -----------------------------
    # Helper expressions
    # -----------------------------
    def _p_poi_abs(t):
        # |P_poi| = import + export (non-overlapping is not required; model can use both but will be penalized by prices)
        return m.p_grid_import[t] + m.p_grid_export[t]

    def _q_poi_abs(t):
        return m.q_poi_pos[t] + m.q_poi_neg[t]

    def _p_bess_abs(t):
        # With no-simultaneous enforced, |P| = P_ch + P_dis.
        return m.p_bess_ch[t] + m.p_bess_dis[t]

    def _q_bess_abs(t):
        return m.q_bess_pos[t] + m.q_bess_neg[t]

    def _q_gen_abs(t):
        return m.q_gen_pos[t] + m.q_gen_neg[t]

    def _q_load(t):
        # Fixed PF load model; assume lagging (consuming Q positive).
        tanphi = _tan_from_pf(load.pf_load)
        return (load.p_critical_mw[t] + load.p_noncritical_mw[t]) * tanphi

    # -----------------------------
    # Core constraints
    # -----------------------------
    # Load accounting
    def crit_balance_rule(_, t):
        return m.p_crit_served[t] + m.p_crit_unserved[t] == float(load.p_critical_mw[t])

    m.crit_balance = pyo.Constraint(m.T, rule=crit_balance_rule)

    def noncrit_balance_rule(_, t):
        return m.p_noncrit_served[t] + m.p_noncrit_unserved[t] == float(load.p_noncritical_mw[t])

    m.noncrit_balance = pyo.Constraint(m.T, rule=noncrit_balance_rule)

    def curtail_cap_rule(_, t):
        return m.p_curtail[t] <= float(load.p_curtailable_max_mw[t])

    m.curtail_cap = pyo.Constraint(m.T, rule=curtail_cap_rule)

    def noncrit_served_curtail_rule(_, t):
        # Curtailment reduces served noncritical load.
        return m.p_noncrit_served[t] == float(load.p_noncritical_mw[t]) - m.p_curtail[t]

    m.noncrit_served_curtail = pyo.Constraint(m.T, rule=noncrit_served_curtail_rule)

    # Microgrid active power balance at microgrid bus:
    # grid_import - grid_export + gen + bess_dis - bess_ch = served_load_total
    def p_balance_rule(_, t):
        return (
            m.p_grid_import[t]
            - m.p_grid_export[t]
            + m.p_gen[t]
            + m.p_bess_dis[t]
            - m.p_bess_ch[t]
            == m.p_crit_served[t] + m.p_noncrit_served[t]
        )

    m.p_balance = pyo.Constraint(m.T, rule=p_balance_rule)

    # Reactive power balance (MVAr) at POI (explicit and auditable):
    # q_poi = q_load - q_gen - q_bess
    def q_balance_rule(_, t):
        q_poi = m.q_poi_pos[t] - m.q_poi_neg[t]
        q_gen = m.q_gen_pos[t] - m.q_gen_neg[t]
        q_bess = m.q_bess_pos[t] - m.q_bess_neg[t]
        return q_poi == _q_load(t) - q_gen - q_bess

    m.q_balance = pyo.Constraint(m.T, rule=q_balance_rule)

    # POI import/export caps
    def import_cap_rule(_, t):
        return m.p_grid_import[t] <= topology.poi.p_import_max_mw

    m.poi_import_cap = pyo.Constraint(m.T, rule=import_cap_rule)

    def export_cap_rule(_, t):
        if allow_export:
            return m.p_grid_export[t] <= topology.poi.p_export_max_mw
        return m.p_grid_export[t] == 0.0

    m.poi_export_cap = pyo.Constraint(m.T, rule=export_cap_rule)

    # POI MVA (polygon facets): a*|P| + b*|Q| <= Smax
    m.poi_mva_facets = pyo.ConstraintList()
    for t in range(len(time)):
        for (a, b) in facets:
            m.poi_mva_facets.add(a * _p_poi_abs(t) + b * _q_poi_abs(t) <= topology.poi.s_max_mva)

    # POI minimum PF screening: |Q| <= tanphi * |P|
    poi_tan = _tan_from_pf(topology.poi.pf_min)
    def poi_pf_rule(_, t):
        return _q_poi_abs(t) <= poi_tan * _p_poi_abs(t)

    m.poi_pf = pyo.Constraint(m.T, rule=poi_pf_rule)

    # Transformer MVA screening using effective N+1 capacity
    tx_smax = topology.transformer_total_effective_mva()
    m.tx_mva_facets = pyo.ConstraintList()
    for t in range(len(time)):
        for (a, b) in facets:
            m.tx_mva_facets.add(a * _p_poi_abs(t) + b * _q_poi_abs(t) <= tx_smax)

    # BESS no simultaneous charge/discharge + bounds
    def bess_mode_rule(_, t):
        return m.z_bess_ch[t] + m.z_bess_dis[t] <= 1

    m.bess_mode = pyo.Constraint(m.T, rule=bess_mode_rule)

    def bess_ch_cap_rule(_, t):
        return m.p_bess_ch[t] <= bess.p_charge_max_mw * m.z_bess_ch[t]

    def bess_dis_cap_rule(_, t):
        return m.p_bess_dis[t] <= bess.p_discharge_max_mw * m.z_bess_dis[t]

    m.bess_ch_cap = pyo.Constraint(m.T, rule=bess_ch_cap_rule)
    m.bess_dis_cap = pyo.Constraint(m.T, rule=bess_dis_cap_rule)

    # BESS PCS MVA screening (polygon facets): a*|P| + b*|Q| <= pcs_mva
    m.bess_pcs_mva_facets = pyo.ConstraintList()
    for t in range(len(time)):
        for (a, b) in facets:
            m.bess_pcs_mva_facets.add(a * _p_bess_abs(t) + b * _q_bess_abs(t) <= bess.pcs_mva)

    # Generator PF screening at terminals: |Q_gen| <= tan(pf_min) * P_gen
    gen_tan = _tan_from_pf(gas_gen.pf_min)
    def gen_pf_rule(_, t):
        return _q_gen_abs(t) <= gen_tan * m.p_gen[t]

    m.gen_pf = pyo.Constraint(m.T, rule=gen_pf_rule)

    # Generator availability with explicit start delay (units started need time before available)
    start_delay_steps = int(np.ceil(gas_gen.start_time_minutes / dt_min)) if dt_min > 0 else 0
    n_on_init = int(case.get("gen_initial_units_on", 0))
    if n_on_init < 0 or n_on_init > gas_gen.n_units:
        raise ValueError("gen_initial_units_on must be within [0, n_units]")

    def on_units_transition_rule(_, t):
        if t == 0:
            return m.n_on[t] == n_on_init + m.n_start[t] - m.n_stop[t]
        return m.n_on[t] == m.n_on[t - 1] + m.n_start[t] - m.n_stop[t]

    m.gen_on_transition = pyo.Constraint(m.T, rule=on_units_transition_rule)

    def available_units_rule(_, t):
        if start_delay_steps == 0:
            return m.n_available[t] == m.n_on[t]

        # Units that have started within the last start_delay_steps are "on but not yet available".
        recent_starts = []
        for k in range(max(0, t - start_delay_steps + 1), t + 1):
            recent_starts.append(m.n_start[k])
        return m.n_available[t] >= m.n_on[t] - sum(recent_starts)

    m.gen_available_lb = pyo.Constraint(m.T, rule=available_units_rule)

    def available_units_ub_rule(_, t):
        return m.n_available[t] <= m.n_on[t]

    m.gen_available_ub = pyo.Constraint(m.T, rule=available_units_ub_rule)

    # Generator power bounds based on available units (startup units produce 0 until available)
    def gen_p_cap_rule(_, t):
        return m.p_gen[t] <= gas_gen.unit_p_max_mw * m.n_available[t]

    m.gen_p_cap = pyo.Constraint(m.T, rule=gen_p_cap_rule)

    def gen_p_min_rule(_, t):
        # If units are available, enforce aggregate minimum. If none available, P=0 satisfies.
        return m.p_gen[t] >= gas_gen.unit_p_min_mw * m.n_available[t]

    m.gen_p_min = pyo.Constraint(m.T, rule=gen_p_min_rule)

    # Generator ramping (simplified but explicit)
    ramp_mw_per_step = gas_gen.ramp_mw_per_min * dt_min
    def gen_ramp_up_rule(_, t):
        if t == 0:
            return pyo.Constraint.Skip
        # Allow extra ramp when new units become available.
        return m.p_gen[t] - m.p_gen[t - 1] <= ramp_mw_per_step * m.n_available[t - 1] + gas_gen.unit_p_max_mw * m.n_start[t]

    def gen_ramp_down_rule(_, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.p_gen[t - 1] - m.p_gen[t] <= ramp_mw_per_step * m.n_available[t] + gas_gen.unit_p_max_mw * m.n_stop[t]

    m.gen_ramp_up = pyo.Constraint(m.T, rule=gen_ramp_up_rule)
    m.gen_ramp_down = pyo.Constraint(m.T, rule=gen_ramp_down_rule)

    # N+1 adequacy (simplified but explicit):
    # Require enough available online capacity to serve critical load after losing one largest unit.
    # Equivalent: total available capacity >= critical load + largest unit capacity.
    def gen_nplus1_rule(_, t):
        return gas_gen.unit_p_max_mw * m.n_available[t] + bess.p_discharge_max_mw >= float(load.p_critical_mw[t]) + gas_gen.unit_p_max_mw

    m.gen_nplus1 = pyo.Constraint(m.T, rule=gen_nplus1_rule)

    # SOC dynamics + bounds (including market + ride-through reserve)
    def soc_init_rule(_, t):
        if t != 0:
            return pyo.Constraint.Skip
        return m.soc[t] == float(bess.soc_init_mwh)

    m.soc_init = pyo.Constraint(m.T, rule=soc_init_rule)

    def soc_dyn_rule(_, t):
        if t == 0:
            return pyo.Constraint.Skip
        return (
            m.soc[t]
            == m.soc[t - 1]
            + bess.eta_charge * m.p_bess_ch[t] * dt_hours
            - (1.0 / bess.eta_discharge) * m.p_bess_dis[t] * dt_hours
        )

    m.soc_dyn = pyo.Constraint(m.T, rule=soc_dyn_rule)

    def soc_bounds_rule(_, t):
        return pyo.inequality(float(bess.soc_min_mwh), m.soc[t], float(bess.soc_max_mwh))

    m.soc_bounds = pyo.Constraint(m.T, rule=soc_bounds_rule)

    def soc_reliability_floor_rule(_, t):
        floor = float(bess.soc_market_reserve_mwh) + float(load.p_critical_mw[t]) * ride_through_hours / float(bess.eta_discharge)
        return m.soc[t] >= floor

    m.soc_reliability_floor = pyo.Constraint(m.T, rule=soc_reliability_floor_rule)

    # BESS ramping (explicit on net power)
    bess_ramp_mw_per_step = bess.ramp_mw_per_min * dt_min
    def bess_ramp_up_rule(_, t):
        if t == 0:
            return pyo.Constraint.Skip
        # |P| changes bounded; we approximate with abs(P)=ch+dis
        return (_p_bess_abs(t) - _p_bess_abs(t - 1)) <= bess_ramp_mw_per_step

    def bess_ramp_down_rule(_, t):
        if t == 0:
            return pyo.Constraint.Skip
        return (_p_bess_abs(t - 1) - _p_bess_abs(t)) <= bess_ramp_mw_per_step

    m.bess_ramp_up = pyo.Constraint(m.T, rule=bess_ramp_up_rule)
    m.bess_ramp_down = pyo.Constraint(m.T, rule=bess_ramp_down_rule)

    # Reserves: headroom + energy backing for BESS, headroom for generator
    rp_by_name: Dict[str, ReserveProduct] = {rp.name: rp for rp in adapter.reserve_products()}

    def bess_reserve_headroom_rule(_, r, t):
        rp = rp_by_name[r]
        if rp.direction == "up":
            # can ramp up by increasing discharge and/or reducing charge
            return m.res_bess[r, t] <= bess.p_discharge_max_mw - m.p_bess_dis[t] + m.p_bess_ch[t]
        # down: increase charge and/or reduce discharge
        return m.res_bess[r, t] <= bess.p_charge_max_mw - m.p_bess_ch[t] + m.p_bess_dis[t]

    m.bess_reserve_headroom = pyo.Constraint(m.R, m.T, rule=bess_reserve_headroom_rule)

    def bess_reserve_energy_backing_rule(_, r, t):
        rp = rp_by_name[r]
        dur_h = float(rp.duration_minutes) / 60.0
        floor = float(bess.soc_market_reserve_mwh) + float(load.p_critical_mw[t]) * ride_through_hours / float(bess.eta_discharge)
        if rp.direction == "up":
            # res * dur <= (soc - floor) * eta_discharge
            return m.res_bess[r, t] * dur_h <= (m.soc[t] - floor) * bess.eta_discharge
        # down: res * dur <= (soc_max - soc)/eta_charge
        return m.res_bess[r, t] * dur_h <= (float(bess.soc_max_mwh) - m.soc[t]) / bess.eta_charge

    m.bess_reserve_energy_backing = pyo.Constraint(m.R, m.T, rule=bess_reserve_energy_backing_rule)

    def gen_reserve_headroom_rule(_, r, t):
        rp = rp_by_name[r]
        if rp.direction == "up":
            return m.res_gen[r, t] <= gas_gen.unit_p_max_mw * m.n_available[t] - m.p_gen[t]
        return m.res_gen[r, t] <= m.p_gen[t] - gas_gen.unit_p_min_mw * m.n_available[t]

    m.gen_reserve_headroom = pyo.Constraint(m.R, m.T, rule=gen_reserve_headroom_rule)

    # -----------------------------
    # Objective: maximize net value
    # -----------------------------
    def objective_rule(_):
        expr = 0.0
        for t in range(len(time)):
            price_e = float(case[energy_key][t])
            expr += (m.p_grid_export[t] - m.p_grid_import[t]) * price_e * dt_hours

            # Reserve revenue
            for r in m.R:
                price_r = float(reserve_price[r][t])
                expr += (m.res_bess[r, t] + m.res_gen[r, t]) * price_r * dt_hours

            # Generator costs
            fuel_marg = gas_gen.marginal_fuel_cost_per_mwh()
            expr -= m.p_gen[t] * (fuel_marg + gas_gen.vom_cost_per_mwh) * dt_hours
            expr -= m.n_start[t] * float(gas_gen.start_cost)

            # BESS degradation cost on throughput
            expr -= (m.p_bess_ch[t] + m.p_bess_dis[t]) * dt_hours * float(bess.degradation_cost_per_mwh_throughput)

            # Unserved energy penalties (VoLL)
            expr -= m.p_crit_unserved[t] * voll_critical * dt_hours
            expr -= m.p_noncrit_unserved[t] * voll_noncritical * dt_hours
        return expr

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    build_inputs = DispatchBuildInputs(
        time=time,
        dt_hours=dt_hours,
        big_m=big_m,
        mva_facets=facets,
    )
    return m, build_inputs


def solve_dispatch_model(
    model: pyo.ConcreteModel,
    *,
    solver_preference: Optional[List[str]] = None,
    tee: bool = False,
    time_limit_s: Optional[int] = None,
) -> pyo.SolverResults:
    """
    Solve MILP. Prefer HiGHS (pip-installable via highspy), fall back to GLPK if available.
    """

    if solver_preference is None:
        solver_preference = ["appsi_highs", "highs", "cbc", "glpk"]

    last_err: Optional[Exception] = None
    for s in solver_preference:
        try:
            opt = pyo.SolverFactory(s)
            if opt is None or not opt.available(exception_flag=False):
                continue
            if time_limit_s is not None:
                # best-effort across solvers
                try:
                    opt.options["time_limit"] = time_limit_s
                except Exception:
                    pass
            return opt.solve(model, tee=tee)
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    raise RuntimeError(f"No MILP solver available via Pyomo. Last error: {last_err}")


def extract_results(
    model: pyo.ConcreteModel,
    build: DispatchBuildInputs,
    *,
    adapter: ISOAdapter,
    case: dict,
    topology: Any,
    bess: Any,
    gas_gen: Any,
    load: Any,
    slack_tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Extract structured results for UI, including constraint slacks for explainability.
    """

    time = build.time
    dt_hours = build.dt_hours

    def v(x):
        return float(pyo.value(x))

    # Core series
    out: Dict[str, Any] = {
        "meta": {
            "dt_hours": dt_hours,
            "n_steps": len(time),
            "solver": str(getattr(model, "solver", "pyomo")),
        },
        "time": time,
        "poi": {
            "p_import_mw": [v(model.p_grid_import[t]) for t in model.T],
            "p_export_mw": [v(model.p_grid_export[t]) for t in model.T],
            "q_poi_mvar": [v(model.q_poi_pos[t] - model.q_poi_neg[t]) for t in model.T],
            "q_abs_mvar": [v(model.q_poi_pos[t] + model.q_poi_neg[t]) for t in model.T],
        },
        "bess": {
            "p_charge_mw": [v(model.p_bess_ch[t]) for t in model.T],
            "p_discharge_mw": [v(model.p_bess_dis[t]) for t in model.T],
            "q_mvar": [v(model.q_bess_pos[t] - model.q_bess_neg[t]) for t in model.T],
            "soc_mwh": [v(model.soc[t]) for t in model.T],
        },
        "gen": {
            "p_mw": [v(model.p_gen[t]) for t in model.T],
            "q_mvar": [v(model.q_gen_pos[t] - model.q_gen_neg[t]) for t in model.T],
            "n_on": [v(model.n_on[t]) for t in model.T],
            "n_available": [v(model.n_available[t]) for t in model.T],
            "n_start": [v(model.n_start[t]) for t in model.T],
            "n_stop": [v(model.n_stop[t]) for t in model.T],
        },
        "load": {
            "p_critical_mw": [float(x) for x in load.p_critical_mw],
            "p_noncritical_mw": [float(x) for x in load.p_noncritical_mw],
            "p_critical_served_mw": [v(model.p_crit_served[t]) for t in model.T],
            "p_noncritical_served_mw": [v(model.p_noncrit_served[t]) for t in model.T],
            "p_critical_unserved_mw": [v(model.p_crit_unserved[t]) for t in model.T],
            "p_noncritical_unserved_mw": [v(model.p_noncrit_unserved[t]) for t in model.T],
            "p_curtail_mw": [v(model.p_curtail[t]) for t in model.T],
        },
        "reserves": {
            "products": list(model.R.data()),
            "bess_mw": {r: [v(model.res_bess[r, t]) for t in model.T] for r in model.R},
            "gen_mw": {r: [v(model.res_gen[r, t]) for t in model.T] for r in model.R},
        },
    }

    # Electrical loading calculations (MVA from extracted P/Q)
    p_abs = np.array(out["poi"]["p_import_mw"]) + np.array(out["poi"]["p_export_mw"])
    q = np.array(out["poi"]["q_poi_mvar"])
    s = np.sqrt(p_abs**2 + q**2)

    # BESS PCS loading
    p_bess_abs = np.array(out["bess"]["p_charge_mw"]) + np.array(out["bess"]["p_discharge_mw"])
    q_bess = np.array(out["bess"]["q_mvar"])
    s_bess = np.sqrt(p_bess_abs**2 + q_bess**2)

    out["electrical"] = {
        "poi_s_mva": s.tolist(),
        "poi_s_limit_mva": float(topology.poi.s_max_mva),
        "transformer_s_mva": s.tolist(),  # same flow in this explicit 1-branch topology
        "transformer_s_limit_mva": float(topology.transformer_total_effective_mva()),
        "bess_pcs_s_mva": s_bess.tolist(),
        "bess_pcs_limit_mva": float(bess.pcs_mva),
        "poi_pf": (np.where(s > 1e-9, p_abs / s, 1.0)).tolist(),
        "poi_pf_min": float(topology.poi.pf_min),
    }

    # Economics breakdown (structured, UI-ready)
    energy_key = adapter.energy_price_series_key()
    reserve_keys = adapter.reserve_price_series_keys()
    fuel_marg = float(gas_gen.marginal_fuel_cost_per_mwh())
    voll_critical = float(case.get("voll_critical_$per_mwh", 10000.0))
    voll_noncritical = float(case.get("voll_noncritical_$per_mwh", 1000.0))

    energy_price = np.array([float(x) for x in case[energy_key]])
    p_import = np.array(out["poi"]["p_import_mw"])
    p_export = np.array(out["poi"]["p_export_mw"])

    energy_import_cost = p_import * energy_price * dt_hours
    energy_export_revenue = p_export * energy_price * dt_hours

    reserve_revenue = np.zeros(len(time))
    for r in out["reserves"]["products"]:
        price = np.array([float(x) for x in case[reserve_keys[r]]])
        res_tot = np.array(out["reserves"]["bess_mw"][r]) + np.array(out["reserves"]["gen_mw"][r])
        reserve_revenue += res_tot * price * dt_hours

    p_gen = np.array(out["gen"]["p_mw"])
    fuel_and_vom_cost = p_gen * (fuel_marg + float(gas_gen.vom_cost_per_mwh)) * dt_hours
    start_cost = np.array(out["gen"]["n_start"]) * float(gas_gen.start_cost)

    p_bess_ch = np.array(out["bess"]["p_charge_mw"])
    p_bess_dis = np.array(out["bess"]["p_discharge_mw"])
    degradation_cost = (p_bess_ch + p_bess_dis) * dt_hours * float(bess.degradation_cost_per_mwh_throughput)

    unserved_crit = np.array(out["load"]["p_critical_unserved_mw"])
    unserved_noncrit = np.array(out["load"]["p_noncritical_unserved_mw"])
    voll_penalty = unserved_crit * voll_critical * dt_hours + unserved_noncrit * voll_noncritical * dt_hours

    out["economics"] = {
        "energy_import_cost_$": energy_import_cost.tolist(),
        "energy_export_revenue_$": energy_export_revenue.tolist(),
        "reserve_revenue_$": reserve_revenue.tolist(),
        "fuel_and_vom_cost_$": fuel_and_vom_cost.tolist(),
        "start_cost_$": start_cost.tolist(),
        "degradation_cost_$": degradation_cost.tolist(),
        "voll_penalty_$": voll_penalty.tolist(),
        "totals_$": {
            "energy_import_cost_$": float(energy_import_cost.sum()),
            "energy_export_revenue_$": float(energy_export_revenue.sum()),
            "reserve_revenue_$": float(reserve_revenue.sum()),
            "fuel_and_vom_cost_$": float(fuel_and_vom_cost.sum()),
            "start_cost_$": float(start_cost.sum()),
            "degradation_cost_$": float(degradation_cost.sum()),
            "voll_penalty_$": float(voll_penalty.sum()),
            "net_value_$": float(
                (energy_export_revenue.sum() + reserve_revenue.sum())
                - (energy_import_cost.sum() + fuel_and_vom_cost.sum() + start_cost.sum() + degradation_cost.sum() + voll_penalty.sum())
            ),
        },
    }

    # Explainability: compute slacks for a curated set of constraints per timestep.
    # Note: MILP duals are not reliable; we use primal slacks and economic signals (prices vs costs).
    expl: Dict[str, Any] = {"binding_constraints": {}, "notes": {}}

    def slack_leq(lhs, rhs) -> float:
        return float(rhs - lhs)

    def slack_geq(lhs, rhs) -> float:
        return float(lhs - rhs)

    for t in range(len(time)):
        bindings: List[Dict[str, Any]] = []

        # POI import/export caps
        s1 = slack_leq(out["poi"]["p_import_mw"][t], float(topology.poi.p_import_max_mw))
        if s1 <= slack_tol:
            bindings.append({"constraint": "poi_import_cap", "slack": s1})
        s2 = slack_leq(out["poi"]["p_export_mw"][t], float(topology.poi.p_export_max_mw))
        if s2 <= slack_tol and bool(case.get("allow_export", True)):
            bindings.append({"constraint": "poi_export_cap", "slack": s2})

        # POI MVA and PF
        s3 = slack_leq(out["electrical"]["poi_s_mva"][t], float(topology.poi.s_max_mva))
        if s3 <= slack_tol:
            bindings.append({"constraint": "poi_mva_limit", "slack": s3})
        s4 = slack_geq(out["electrical"]["poi_pf"][t], float(topology.poi.pf_min))
        if s4 <= slack_tol:
            bindings.append({"constraint": "poi_pf_min", "slack": s4})

        # Transformer
        s5 = slack_leq(out["electrical"]["transformer_s_mva"][t], float(topology.transformer_total_effective_mva()))
        if s5 <= slack_tol:
            bindings.append({"constraint": "transformer_mva_screening", "slack": s5})

        # BESS PCS
        s6 = slack_leq(out["electrical"]["bess_pcs_s_mva"][t], float(bess.pcs_mva))
        if s6 <= slack_tol:
            bindings.append({"constraint": "bess_pcs_mva_limit", "slack": s6})

        # SOC reserve floor (market + ride-through)
        ride_through_minutes = int(case.get("ride_through_minutes", max(0, gas_gen.start_time_minutes)))
        ride_through_hours = float(ride_through_minutes) / 60.0
        soc_floor = float(bess.soc_market_reserve_mwh) + float(load.p_critical_mw[t]) * ride_through_hours / float(bess.eta_discharge)
        s7 = slack_geq(out["bess"]["soc_mwh"][t], soc_floor)
        if s7 <= slack_tol:
            bindings.append({"constraint": "soc_reliability_floor", "slack": s7})

        # Generator N+1 adequacy
        n_av = out["gen"]["n_available"][t]
        lhs = float(gas_gen.unit_p_max_mw) * float(n_av) + float(bess.p_discharge_max_mw)
        rhs = float(load.p_critical_mw[t]) + float(gas_gen.unit_p_max_mw)
        s8 = slack_geq(lhs, rhs)
        if s8 <= slack_tol:
            bindings.append({"constraint": "gen_nplus1", "slack": s8})

        expl["binding_constraints"][time[t]] = bindings

    # Lightweight economic context for decision explainer (opportunity cost scaffolding)
    expl["notes"]["marginals"] = {
        "fuel_$per_mwh": float(gas_gen.marginal_fuel_cost_per_mwh()),
        "bess_degradation_$per_mwh_throughput": float(bess.degradation_cost_per_mwh_throughput),
        "voll_critical_$per_mwh": float(case.get("voll_critical_$per_mwh", 10000.0)),
        "energy_price_series_key": energy_key,
    }
    out["explainability"] = expl

    return out

