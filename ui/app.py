from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from runner import run_case


def _to_df(results: Dict[str, Any]) -> pd.DataFrame:
    t = results["time"]
    df = pd.DataFrame(
        {
            "time": t,
            "p_import_mw": results["poi"]["p_import_mw"],
            "p_export_mw": results["poi"]["p_export_mw"],
            "q_poi_mvar": results["poi"]["q_poi_mvar"],
            "soc_mwh": results["bess"]["soc_mwh"],
            "p_bess_ch_mw": results["bess"]["p_charge_mw"],
            "p_bess_dis_mw": results["bess"]["p_discharge_mw"],
            "p_gen_mw": results["gen"]["p_mw"],
            "n_gen_available": results["gen"]["n_available"],
            "p_crit_served_mw": results["load"]["p_critical_served_mw"],
            "p_noncrit_served_mw": results["load"]["p_noncritical_served_mw"],
            "p_crit_unserved_mw": results["load"]["p_critical_unserved_mw"],
            "p_noncrit_unserved_mw": results["load"]["p_noncritical_unserved_mw"],
        }
    )
    return df


def _plot_timeseries(df: pd.DataFrame, series: List[str], title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(x=df["time"], y=df[s], mode="lines", name=s))
    fig.update_layout(title=title, xaxis_title="time", yaxis_title=ytitle, legend_orientation="h")
    return fig


def _plot_stacked_dispatch(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["time"], y=df["p_gen_mw"], name="Gen (MW)"))
    fig.add_trace(go.Bar(x=df["time"], y=df["p_bess_dis_mw"], name="BESS discharge (MW)"))
    fig.add_trace(go.Bar(x=df["time"], y=-df["p_bess_ch_mw"], name="BESS charge (MW)"))
    fig.add_trace(go.Bar(x=df["time"], y=df["p_import_mw"], name="Grid import (MW)"))
    fig.add_trace(go.Bar(x=df["time"], y=-df["p_export_mw"], name="Grid export (MW)"))
    fig.update_layout(barmode="relative", title="Dispatch power stack (sign convention shown)", xaxis_title="time", yaxis_title="MW")
    return fig


def _plot_pq_scatter(results: Dict[str, Any]) -> go.Figure:
    p = np.array(results["poi"]["p_import_mw"]) - np.array(results["poi"]["p_export_mw"])
    q = np.array(results["poi"]["q_poi_mvar"])
    smax = float(results["electrical"]["poi_s_limit_mva"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=q, mode="markers+lines", name="POI operating points"))

    # Draw MVA circle (reference only; optimization uses linear facets)
    th = np.linspace(0, 2 * np.pi, 361)
    fig.add_trace(go.Scatter(x=smax * np.cos(th), y=smax * np.sin(th), mode="lines", name="POI MVA limit (circle)", line=dict(dash="dash")))
    fig.update_layout(title="POI P-Q envelope (reference)", xaxis_title="P at POI (MW, import positive)", yaxis_title="Q at POI (MVAr)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _binding_table(results: Dict[str, Any], ts: str) -> pd.DataFrame:
    rows = results.get("explainability", {}).get("binding_constraints", {}).get(ts, [])
    if not rows:
        return pd.DataFrame(columns=["constraint", "slack"])
    return pd.DataFrame(rows)


st.set_page_config(page_title="Data Center Energy Platform", layout="wide")
st.title("Data Center Energy Platform — Planning + Dispatch (MILP)")

with st.sidebar:
    st.header("Case")
    case_path = st.text_input("Case JSON path", value="cases/example_ercot.json")
    run_button = st.button("Run MILP", type="primary")
    page = st.radio(
        "Pages",
        [
            "System Overview",
            "Electrical Feasibility",
            "Dispatch Timeline",
            "Reliability & Reserves",
            "Economics",
            "Decision Explainer",
        ],
    )

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_button:
    case = json.loads(Path(case_path).read_text())
    with st.spinner("Solving MILP (Pyomo)…"):
        st.session_state["results"] = run_case(case, tee=False)

results = st.session_state["results"]
if results is None:
    st.info("Load a case and click **Run MILP**.")
    st.stop()

df = _to_df(results)

iso_name = results["meta"].get("iso", "UNKNOWN")
solve_meta = results["meta"].get("solve", {})

if page == "System Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("ISO", iso_name)
    c2.metric("Solver status", solve_meta.get("status", ""))
    c3.metric("Termination", solve_meta.get("termination_condition", ""))

    st.subheader("Topology summary")
    st.write(results["meta"].get("topology_summary", {}))

    st.subheader("Resource stack")
    st.write(
        {
            "Gen_max_MW": float(np.max(df["p_gen_mw"])),
            "BESS_max_discharge_MW": float(np.max(df["p_bess_dis_mw"])),
            "BESS_max_charge_MW": float(np.max(df["p_bess_ch_mw"])),
            "BESS_soc_min_MWh": float(np.min(df["soc_mwh"])),
            "BESS_soc_max_MWh": float(np.max(df["soc_mwh"])),
        }
    )

    st.subheader("Ride-through guarantee")
    st.write(
        {
            "ride_through_minutes_configured": int(results["meta"].get("ride_through_minutes", 0)),
            "generator_start_time_minutes": int(results["meta"].get("gen_start_time_minutes", 0)),
            "note": "SOC reliability floor enforced as market SOC reserve + critical load ride-through energy.",
        }
    )

elif page == "Electrical Feasibility":
    st.subheader("POI operating envelope (MVA + PF screening)")
    st.plotly_chart(_plot_pq_scatter(results), use_container_width=True)
    fig1 = _plot_timeseries(df.assign(poi_s_mva=results["electrical"]["poi_s_mva"]), ["poi_s_mva"], "POI apparent power", "MVA")
    fig1.add_hline(y=results["electrical"]["poi_s_limit_mva"], line_dash="dash", annotation_text="POI MVA limit")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = _plot_timeseries(df.assign(poi_pf=results["electrical"]["poi_pf"]), ["poi_pf"], "POI power factor (magnitude)", "PF")
    fig2.add_hline(y=results["electrical"]["poi_pf_min"], line_dash="dash", annotation_text="POI PF min")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Transformer loading over time (screening)")
    fig3 = _plot_timeseries(df.assign(tx_s_mva=results["electrical"]["transformer_s_mva"]), ["tx_s_mva"], "Transformer apparent power", "MVA")
    fig3.add_hline(y=results["electrical"]["transformer_s_limit_mva"], line_dash="dash", annotation_text="Effective MVA limit (N+1)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("PCS loading vs MVA limit")
    fig4 = _plot_timeseries(df.assign(pcs_s_mva=results["electrical"]["bess_pcs_s_mva"]), ["pcs_s_mva"], "BESS PCS apparent power", "MVA")
    fig4.add_hline(y=results["electrical"]["bess_pcs_limit_mva"], line_dash="dash", annotation_text="PCS MVA limit")
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Dispatch Timeline":
    st.subheader("Dispatch stack")
    st.plotly_chart(_plot_stacked_dispatch(df), use_container_width=True)

    st.subheader("SOC")
    st.plotly_chart(_plot_timeseries(df, ["soc_mwh"], "BESS SOC", "MWh"), use_container_width=True)

    st.subheader("Grid import/export")
    st.plotly_chart(_plot_timeseries(df, ["p_import_mw", "p_export_mw"], "POI power", "MW"), use_container_width=True)

elif page == "Reliability & Reserves":
    st.subheader("Critical vs non-critical served")
    st.plotly_chart(
        _plot_timeseries(df, ["p_crit_served_mw", "p_noncrit_served_mw"], "Load served", "MW"),
        use_container_width=True,
    )

    st.subheader("Unserved energy flags (prominent)")
    unserved = df["p_crit_unserved_mw"].to_numpy()
    if np.max(unserved) > 1e-6:
        st.error(f"Unserved CRITICAL load detected. Max unserved MW: {float(np.max(unserved)):.3f}")
    else:
        st.success("No unserved critical load.")

    st.plotly_chart(
        _plot_timeseries(df, ["p_crit_unserved_mw", "p_noncrit_unserved_mw"], "Unserved load", "MW"),
        use_container_width=True,
    )

    st.subheader("SOC reserve line (never violated)")
    st.caption("This is the enforced floor: market SOC reserve + critical-load ride-through energy.")
    soc = np.array(results["bess"]["soc_mwh"])
    crit = np.array(results["load"]["p_critical_mw"])
    eta_dis = float(results["meta"].get("bess_eta_discharge", 0.97))
    ride_h = float(results["meta"].get("ride_through_minutes", 0)) / 60.0
    soc_floor = float(results["meta"].get("soc_market_reserve_mwh", 0)) if "soc_market_reserve_mwh" in results["meta"] else None
    if soc_floor is None:
        # fall back: infer from minimum slack of SOC floor by reading binding logic is not persisted; show only SOC.
        st.plotly_chart(_plot_timeseries(df, ["soc_mwh"], "BESS SOC", "MWh"), use_container_width=True)
    else:
        floor_series = soc_floor + crit * ride_h / eta_dis
        sdf = pd.DataFrame({"time": results["time"], "soc_mwh": soc, "soc_floor_mwh": floor_series})
        st.plotly_chart(_plot_timeseries(sdf, ["soc_mwh", "soc_floor_mwh"], "BESS SOC vs reserve floor", "MWh"), use_container_width=True)

    st.subheader("Reserve provision by product")
    products = results["reserves"]["products"]
    for r in products:
        s = pd.DataFrame(
            {
                "time": results["time"],
                "bess": results["reserves"]["bess_mw"][r],
                "gen": results["reserves"]["gen_mw"][r],
            }
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["time"], y=s["bess"], mode="lines", name=f"{r} (BESS)"))
        fig.add_trace(go.Scatter(x=s["time"], y=s["gen"], mode="lines", name=f"{r} (Gen)"))
        fig.update_layout(title=f"{r} reserve provision", xaxis_title="time", yaxis_title="MW", legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Economics":
    st.subheader("Net value (objective) — planning-grade summary")
    st.write(
        {
            "objective_value_$": results["meta"]["solve"].get("objective", "n/a"),
            "note": "Objective includes energy settlement, reserve revenue, fuel+VOM+start, degradation, and VoLL penalties.",
        }
    )

    st.subheader("Revenue + cost stacks")
    econ = results.get("economics", {})
    totals = econ.get("totals_$", {})
    if totals:
        c1, c2, c3 = st.columns(3)
        c1.metric("Net value ($)", f"{totals.get('net_value_$', 0.0):,.0f}")
        c2.metric("Energy export revenue ($)", f"{totals.get('energy_export_revenue_$', 0.0):,.0f}")
        c3.metric("Reserve revenue ($)", f"{totals.get('reserve_revenue_$', 0.0):,.0f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Energy import cost ($)", f"{totals.get('energy_import_cost_$', 0.0):,.0f}")
        c5.metric("Fuel+VOM+start ($)", f"{(totals.get('fuel_and_vom_cost_$', 0.0) + totals.get('start_cost_$', 0.0)):,.0f}")
        c6.metric("Degradation + VoLL ($)", f"{(totals.get('degradation_cost_$', 0.0) + totals.get('voll_penalty_$', 0.0)):,.0f}")

        tdf = pd.DataFrame(
            {
                "time": results["time"],
                "energy_export_revenue_$": econ.get("energy_export_revenue_$", []),
                "reserve_revenue_$": econ.get("reserve_revenue_$", []),
                "energy_import_cost_$": [-x for x in econ.get("energy_import_cost_$", [])],
                "fuel_and_vom_cost_$": [-x for x in econ.get("fuel_and_vom_cost_$", [])],
                "start_cost_$": [-x for x in econ.get("start_cost_$", [])],
                "degradation_cost_$": [-x for x in econ.get("degradation_cost_$", [])],
                "voll_penalty_$": [-x for x in econ.get("voll_penalty_$", [])],
            }
        )
        fig = go.Figure()
        for col in [c for c in tdf.columns if c != "time"]:
            fig.add_trace(go.Bar(x=tdf["time"], y=tdf[col], name=col))
        fig.update_layout(barmode="relative", title="Economics stack by timestep (revenues positive, costs negative)", xaxis_title="time", yaxis_title="$")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensitivity toggles (what-if)")
    st.caption("These toggles re-run the MILP by modifying the case in-memory.")
    fuel_mult = st.slider("Fuel price multiplier", 0.5, 2.0, 1.0, 0.05)
    soc_reserve_add = st.slider("Add SOC market reserve (MWh)", 0.0, 50.0, 0.0, 1.0)
    ride_through_min = st.slider("Ride-through minutes", 0, 120, 30, 5)

    if st.button("Re-run with sensitivity"):
        case = json.loads(Path(case_path).read_text())
        case["resources"]["gas_gen"]["fuel_cost_$per_mmbtu"] = float(case["resources"]["gas_gen"]["fuel_cost_$per_mmbtu"]) * float(fuel_mult)
        case["resources"]["bess"]["soc_market_reserve_mwh"] = float(case["resources"]["bess"]["soc_market_reserve_mwh"]) + float(soc_reserve_add)
        case["ride_through_minutes"] = int(ride_through_min)
        with st.spinner("Re-solving MILP…"):
            st.session_state["results"] = run_case(case, tee=False)
        st.rerun()

elif page == "Decision Explainer":
    st.subheader("Decision explainer (binding constraints + opportunity cost)")
    ts = st.selectbox("Timestep", results["time"], index=0)

    st.markdown("**Binding constraints at selected timestep**")
    st.dataframe(_binding_table(results, ts), use_container_width=True)

    st.markdown("**Context (prices and marginals)**")
    st.write(results.get("explainability", {}).get("notes", {}))

    st.markdown("**Why didn’t the battery discharge? (example logic)**")
    idx = results["time"].index(ts)
    p_dis = float(results["bess"]["p_discharge_mw"][idx])
    if p_dis <= 1e-6:
        bindings = results.get("explainability", {}).get("binding_constraints", {}).get(ts, [])
        reasons = []
        if any(b["constraint"] == "soc_reliability_floor" for b in bindings):
            reasons.append("SOC reliability floor was binding (market + ride-through reserve).")
        if any(b["constraint"] == "bess_pcs_mva_limit" for b in bindings):
            reasons.append("PCS MVA limit was binding (P/Q coupling).")
        if any(b["constraint"] in {"poi_mva_limit", "transformer_mva_screening"} for b in bindings):
            reasons.append("Upstream MVA screening was binding (POI/transformer).")
        if not reasons:
            reasons.append("Dispatch economics did not justify discharge vs alternative (energy price vs fuel + degradation + reserve opportunity).")
        st.write({"explanation": reasons})
    else:
        st.write({"explanation": ["Battery was discharging at this timestep."]})

    st.markdown("**Why did generation start? (example logic)**")
    n_start = float(results["gen"]["n_start"][idx])
    if n_start >= 1e-6:
        bindings = results.get("explainability", {}).get("binding_constraints", {}).get(ts, [])
        reasons = []
        reasons.append("A generator start decision occurred (explicit start cost + start delay are modeled).")
        if any(b["constraint"] == "gen_nplus1" for b in bindings):
            reasons.append("N+1 adequacy constraint was binding (critical load + largest unit margin).")
        if any(b["constraint"] in {"poi_import_cap", "transformer_mva_screening", "poi_mva_limit"} for b in bindings):
            reasons.append("Upstream import/MVA screening limited the ability to serve load from the grid, pushing on-site generation.")
        if not reasons:
            reasons.append("Economics favored on-site generation vs grid energy price (fuel+VOM vs LMP).")
        st.write({"explanation": reasons})
    else:
        st.write({"explanation": ["No generator start at this timestep."]})

