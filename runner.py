from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pyomo.environ as pyo

from iso import ERCOTAdapter
from optimization.dispatch_milp import build_dispatch_model, extract_results, solve_dispatch_model
from resources import BESS, GasGenFleet, Load
from topology import POI, Topology, Transformer


def select_adapter(iso_name: str):
    iso_name = (iso_name or "").strip().upper()
    if iso_name in {"ERCOT"}:
        return ERCOTAdapter()
    raise ValueError(f"Unsupported ISO '{iso_name}'. Implement additional adapters under iso/.")


def build_objects_from_case(case: Dict[str, Any]):
    # ISO adapter
    adapter = select_adapter(case.get("iso", "ERCOT"))

    # Topology
    poi_d = case["topology"]["poi"]
    poi = POI(
        p_import_max_mw=float(poi_d["p_import_max_mw"]),
        p_export_max_mw=float(poi_d["p_export_max_mw"]),
        s_max_mva=float(poi_d["s_max_mva"]),
        pf_min=float(poi_d["pf_min"]),
    )
    transformers = []
    for td in case["topology"]["transformers"]:
        transformers.append(
            Transformer(
                name=str(td["name"]),
                count=int(td["count"]),
                mva_rating=float(td["mva_rating"]),
                n_plus_one=bool(td.get("n_plus_one", True)),
            )
        )
    topology = Topology(poi=poi, transformers=transformers)

    # Resources
    bd = case["resources"]["bess"]
    bess = BESS(
        name=str(bd.get("name", "BESS")),
        p_charge_max_mw=float(bd["p_charge_max_mw"]),
        p_discharge_max_mw=float(bd["p_discharge_max_mw"]),
        e_max_mwh=float(bd["e_max_mwh"]),
        pcs_mva=float(bd["pcs_mva"]),
        eta_charge=float(bd["eta_charge"]),
        eta_discharge=float(bd["eta_discharge"]),
        ramp_mw_per_min=float(bd["ramp_mw_per_min"]),
        soc_init_mwh=float(bd["soc_init_mwh"]),
        soc_min_mwh=float(bd["soc_min_mwh"]),
        soc_max_mwh=float(bd["soc_max_mwh"]),
        soc_market_reserve_mwh=float(bd["soc_market_reserve_mwh"]),
        degradation_cost_per_mwh_throughput=float(bd["degradation_cost_$per_mwh_throughput"]),
    )

    gd = case["resources"]["gas_gen"]
    gas_gen = GasGenFleet(
        name=str(gd.get("name", "GasGen")),
        n_units=int(gd["n_units"]),
        unit_p_max_mw=float(gd["unit_p_max_mw"]),
        unit_p_min_mw=float(gd["unit_p_min_mw"]),
        ramp_mw_per_min=float(gd["ramp_mw_per_min"]),
        start_time_minutes=int(gd["start_time_minutes"]),
        heat_rate_mmbtu_per_mwh=float(gd["heat_rate_mmbtu_per_mwh"]),
        fuel_cost_per_mmbtu=float(gd["fuel_cost_$per_mmbtu"]),
        vom_cost_per_mwh=float(gd["vom_cost_$per_mwh"]),
        start_cost=float(gd["start_cost_$"]),
        pf_min=float(gd["pf_min"]),
    )

    ld = case["resources"]["load"]
    load = Load(
        name=str(ld.get("name", "Load")),
        p_critical_mw=[float(x) for x in ld["p_critical_mw"]],
        p_noncritical_mw=[float(x) for x in ld["p_noncritical_mw"]],
        p_curtailable_max_mw=[float(x) for x in ld["p_curtailable_max_mw"]],
        pf_load=float(ld["pf_load"]),
    )

    return adapter, topology, bess, gas_gen, load


def run_case(case: Dict[str, Any], *, tee: bool = False) -> Dict[str, Any]:
    adapter, topology, bess, gas_gen, load = build_objects_from_case(case)
    model, build = build_dispatch_model(
        adapter=adapter,
        case=case,
        topology=topology,
        bess=bess,
        gas_gen=gas_gen,
        load=load,
        solver_facets=int(case.get("mva_linear_facets", 12)),
    )
    results = solve_dispatch_model(model, tee=tee, time_limit_s=case.get("time_limit_s"))

    # Attach solve status for UI
    solve_meta = {
        "status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
    }
    try:
        solve_meta["objective"] = float(pyo.value(model.obj))  # type: ignore[name-defined]
    except Exception:
        pass

    out = extract_results(
        model,
        build,
        adapter=adapter,
        case=case,
        topology=topology,
        bess=bess,
        gas_gen=gas_gen,
        load=load,
    )
    out["meta"]["iso"] = adapter.iso_name
    out["meta"]["solve"] = solve_meta
    out["meta"]["case_name"] = case.get("case_name", "unnamed")
    out["meta"]["ride_through_minutes"] = int(case.get("ride_through_minutes", max(0, gas_gen.start_time_minutes)))
    out["meta"]["gen_start_time_minutes"] = int(gas_gen.start_time_minutes)
    out["meta"]["soc_market_reserve_mwh"] = float(bess.soc_market_reserve_mwh)
    out["meta"]["bess_eta_discharge"] = float(bess.eta_discharge)
    out["meta"]["topology_summary"] = {
        "poi": {
            "p_import_max_mw": float(topology.poi.p_import_max_mw),
            "p_export_max_mw": float(topology.poi.p_export_max_mw),
            "s_max_mva": float(topology.poi.s_max_mva),
            "pf_min": float(topology.poi.pf_min),
        },
        "transformers": [
            {
                "name": t.name,
                "count": int(t.count),
                "mva_rating": float(t.mva_rating),
                "n_plus_one": bool(t.n_plus_one),
                "effective_mva_capacity": float(t.effective_mva_capacity()),
            }
            for t in topology.transformers
        ],
        "transformer_total_effective_mva": float(topology.transformer_total_effective_mva()),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Run microgrid sizing+dispatch case (Pyomo MILP).")
    ap.add_argument("case_json", type=str, help="Path to case JSON")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path")
    ap.add_argument("--tee", action="store_true", help="Stream solver output")
    args = ap.parse_args()

    case_path = Path(args.case_json)
    case = json.loads(case_path.read_text())
    out = run_case(case, tee=args.tee)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

