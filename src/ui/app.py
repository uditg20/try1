"""
Data Center Energy Platform - Streamlit UI
===========================================

Professional planning and operations interface for data center
microgrid dispatch optimization.

Pages:
1. System Overview
2. Electrical Feasibility
3. Dispatch Timeline
4. Reliability & Reserves
5. Economics
6. Decision Explainer
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from runner import run_case, load_case_file


# Page configuration
st.set_page_config(
    page_title="Data Center Energy Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_results() -> Optional[Dict[str, Any]]:
    """Load optimization results from session state or run optimization."""
    if "results" in st.session_state:
        return st.session_state["results"]
    return None


def run_optimization(case_file: str) -> Dict[str, Any]:
    """Run optimization and store results."""
    np.random.seed(42)  # For reproducibility
    results = run_case(case_file)
    st.session_state["results"] = results
    st.session_state["case_file"] = case_file
    return results


def page_system_overview(results: Dict[str, Any]):
    """System Overview page."""
    st.header("📊 System Overview")
    
    case = results["case"]
    topology = results["topology"]
    resources = results["resources"]
    opt_results = results["results"]
    
    # Top-level info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ISO", case["iso"])
    with col2:
        st.metric("Case", case["name"])
    with col3:
        status = "✅ Solved" if opt_results["solved"] else "❌ Failed"
        st.metric("Status", status)
    
    st.divider()
    
    # Topology Summary
    st.subheader("🔌 Electrical Topology")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Point of Interconnection (POI)**")
        poi = topology["poi"]
        poi_df = pd.DataFrame({
            "Parameter": ["Max Import", "Max Export", "MVA Rating", "Min PF", "Voltage"],
            "Value": [
                f"{poi['max_import_mw']:.1f} MW",
                f"{poi['max_export_mw']:.1f} MW",
                f"{poi['mva_rating']:.1f} MVA",
                f"{poi['min_pf']:.2f}",
                f"{poi['voltage_kv']:.1f} kV"
            ]
        })
        st.dataframe(poi_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Transformers**")
        xfmr = topology["transformers"]
        st.metric("Total MVA", f"{xfmr['total_mva']:.1f} MVA")
        st.metric("Firm MVA (N-1)", f"{xfmr['firm_mva']:.1f} MVA")
        st.metric("Units", f"{xfmr['count']} ({xfmr['units'][0]['config'] if xfmr['units'] else 'N/A'})")
    
    st.divider()
    
    # Resource Stack
    st.subheader("⚡ Resource Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**BESS**")
        if resources["bess"]:
            bess = resources["bess"]
            st.metric("Power", f"{bess['power_mw']:.1f} MW")
            st.metric("Energy", f"{bess['energy_mwh']:.1f} MWh")
            st.metric("Duration", f"{bess['duration_hours']:.1f} hours")
            st.metric("SOC Reserve", f"{bess['soc_reserve']*100:.0f}%")
        else:
            st.info("No BESS configured")
    
    with col2:
        st.markdown("**Generators**")
        if resources["generators"]:
            gens = resources["generators"]
            st.metric("Total Capacity", f"{gens['total_capacity_mw']:.1f} MW")
            st.metric("Firm Capacity (N+1)", f"{gens['firm_capacity_mw']:.1f} MW")
            st.metric("Units", f"{gens['n_units']}")
            st.metric("Fastest Start", f"{gens['fastest_start_min']:.1f} min")
        else:
            st.info("No generators configured")
    
    with col3:
        st.markdown("**Loads**")
        loads = resources["loads"]
        total = sum(l["base_mw"] for l in loads)
        critical = sum(l["base_mw"] for l in loads if l["type"] == "critical")
        st.metric("Total Load", f"{total:.1f} MW")
        st.metric("Critical Load", f"{critical:.1f} MW")
        st.metric("Load Categories", f"{len(loads)}")
    
    st.divider()
    
    # Ride-through Guarantee
    st.subheader("🛡️ Ride-Through Capability")
    
    if resources["bess"] and resources["generators"]:
        bess = resources["bess"]
        gens = resources["generators"]
        critical_mw = sum(l["base_mw"] for l in loads if l["type"] == "critical")
        
        # Calculate ride-through
        available_soc = bess["initial_soc"] - bess["soc_reserve"]
        available_mwh = available_soc * bess["energy_mwh"]
        
        if critical_mw > 0:
            ride_through_min = (available_mwh / critical_mw) * 60
            gen_start_min = gens["fastest_start_min"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BESS Ride-Through", f"{ride_through_min:.1f} min")
            with col2:
                st.metric("Generator Start Time", f"{gen_start_min:.1f} min")
            with col3:
                margin = ride_through_min - gen_start_min
                if margin > 0:
                    st.success(f"✅ Margin: {margin:.1f} min")
                else:
                    st.error(f"⚠️ Gap: {abs(margin):.1f} min")


def page_electrical_feasibility(results: Dict[str, Any]):
    """Electrical Feasibility page."""
    st.header("🔧 Electrical Feasibility")
    
    topology = results["topology"]
    dispatch = results["dispatch_case"]
    opt_results = results["results"]
    
    if not opt_results["solved"]:
        st.error("Optimization not solved - cannot display feasibility analysis")
        return
    
    ts = opt_results["timeseries"]
    hours = ts["hours"]
    
    # POI Operating Envelope
    st.subheader("POI Operating Envelope (P vs MVA)")
    
    poi = topology["poi"]
    p_import = np.array(ts["grid_import_mw"])
    p_export = np.array(ts["grid_export_mw"])
    p_net = p_import - p_export
    
    # Generate envelope
    p_range = np.linspace(-poi["max_export_mw"], poi["max_import_mw"], 100)
    s_limit = poi["mva_rating"]
    
    fig = go.Figure()
    
    # MVA circle (simplified as horizontal line for P limit)
    fig.add_trace(go.Scatter(
        x=[poi["max_import_mw"]] * 2,
        y=[0, s_limit],
        mode='lines',
        name='Import Limit',
        line=dict(color='red', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[-poi["max_export_mw"]] * 2,
        y=[0, s_limit],
        mode='lines',
        name='Export Limit',
        line=dict(color='red', dash='dash')
    ))
    
    # MVA limit line
    fig.add_hline(y=s_limit, line_dash="dash", line_color="orange",
                  annotation_text=f"MVA Limit: {s_limit}")
    
    # Operating points
    s_operating = np.abs(p_net)  # Simplified - assuming Q ≈ 0
    fig.add_trace(go.Scatter(
        x=p_net,
        y=s_operating,
        mode='markers',
        name='Operating Points',
        marker=dict(size=5, color=hours, colorscale='Viridis',
                   colorbar=dict(title='Hour'))
    ))
    
    fig.update_layout(
        xaxis_title="Real Power P (MW) [+Import/-Export]",
        yaxis_title="Apparent Power S (MVA)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Transformer Loading
    st.subheader("Transformer Loading Over Time")
    
    xfmr = topology["transformers"]
    total_mva = xfmr["total_mva"]
    
    loading_pct = (np.abs(p_net) / total_mva) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=loading_pct,
        mode='lines',
        name='Loading %',
        fill='tozeroy',
        line=dict(color='#1f77b4')
    ))
    
    # Threshold lines
    fig.add_hline(y=80, line_dash="dash", line_color="yellow",
                  annotation_text="80% (Normal)")
    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  annotation_text="100% (Rated)")
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Transformer Loading (%)",
        height=300,
        yaxis=dict(range=[0, max(120, max(loading_pct) * 1.1)])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Warnings
    max_loading = max(loading_pct)
    if max_loading > 100:
        st.error(f"⚠️ Transformer overload detected! Max loading: {max_loading:.1f}%")
    elif max_loading > 80:
        st.warning(f"⚡ Elevated transformer loading: {max_loading:.1f}%")
    else:
        st.success(f"✅ Transformer loading within limits: {max_loading:.1f}%")
    
    st.divider()
    
    # PCS Loading
    st.subheader("PCS Loading vs MVA Limit")
    
    pcs = topology["pcs"]
    if pcs["count"] > 0:
        bess_charge = np.array(ts["bess_charge_mw"])
        bess_discharge = np.array(ts["bess_discharge_mw"])
        bess_power = np.maximum(bess_charge, bess_discharge)
        
        pcs_mva = pcs["total_mva"]
        pcs_loading = (bess_power / pcs_mva) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=pcs_loading,
            mode='lines',
            name='PCS Loading %',
            fill='tozeroy',
            line=dict(color='#2ca02c')
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color="red",
                      annotation_text="100% (MVA Limit)")
        
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="PCS Loading (%)",
            height=300,
            yaxis=dict(range=[0, max(120, max(pcs_loading) * 1.1)])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        max_pcs = max(pcs_loading)
        if max_pcs > 100:
            st.error(f"⚠️ PCS MVA limit exceeded! Max: {max_pcs:.1f}%")
        else:
            st.success(f"✅ PCS within MVA limits: {max_pcs:.1f}%")
    else:
        st.info("No PCS configured")


def page_dispatch_timeline(results: Dict[str, Any]):
    """Dispatch Timeline page."""
    st.header("📈 Dispatch Timeline")
    
    opt_results = results["results"]
    dispatch = results["dispatch_case"]
    
    if not opt_results["solved"]:
        st.error("Optimization not solved")
        return
    
    ts = opt_results["timeseries"]
    hours = ts["hours"]
    
    # Energy Price
    st.subheader("Energy Price")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=dispatch["prices"]["energy"],
        mode='lines',
        name='LMP ($/MWh)',
        line=dict(color='#1f77b4'),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Price ($/MWh)",
        height=250
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # BESS Dispatch
    st.subheader("BESS Charge/Discharge & SOC")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Charge (negative) and Discharge (positive)
    charge = np.array(ts["bess_charge_mw"])
    discharge = np.array(ts["bess_discharge_mw"])
    net = discharge - charge
    
    # Create bar colors based on charge/discharge
    colors = ['#d62728' if n < 0 else '#2ca02c' for n in net]
    
    fig.add_trace(
        go.Bar(x=hours, y=net, name='BESS Power (MW)',
               marker_color=colors),
        secondary_y=False
    )
    
    # SOC
    soc = np.array(ts["bess_soc"]) * 100
    fig.add_trace(
        go.Scatter(x=hours, y=soc, name='SOC (%)',
                   line=dict(color='#ff7f0e', width=2)),
        secondary_y=True
    )
    
    # SOC Reserve line
    bess_info = results["resources"]["bess"]
    if bess_info:
        reserve_pct = bess_info["soc_reserve"] * 100
        fig.add_hline(y=reserve_pct, line_dash="dash", line_color="red",
                      annotation_text=f"Reserve Floor: {reserve_pct:.0f}%",
                      secondary_y=True)
    
    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Power (MW)", secondary_y=False)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 100], secondary_y=True)
    fig.update_xaxes(title_text="Hour")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Generator Output
    st.subheader("Generator Output")
    
    gen_output = np.array(ts["gen_output_mw"])
    gen_units = np.array(ts["gen_units_online"])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=hours, y=gen_output, name='Gen Output (MW)',
                   fill='tozeroy', line=dict(color='#9467bd')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=hours, y=gen_units, name='Units Online',
                   line=dict(color='#8c564b', dash='dot'), mode='lines'),
        secondary_y=True
    )
    
    fig.update_layout(height=300)
    fig.update_yaxes(title_text="Output (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Units Online", secondary_y=True)
    fig.update_xaxes(title_text="Hour")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Grid Import/Export
    st.subheader("Grid Import/Export")
    
    grid_import = np.array(ts["grid_import_mw"])
    grid_export = np.array(ts["grid_export_mw"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=grid_import,
        name='Import (MW)',
        fill='tozeroy',
        line=dict(color='#1f77b4')
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=-grid_export,
        name='Export (MW)',
        fill='tozeroy',
        line=dict(color='#2ca02c')
    ))
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Power (MW) [+Import/-Export]",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Load vs Supply Stack
    st.subheader("Load vs Supply Stack")
    
    total_load = np.array(ts["total_load_mw"])
    
    fig = go.Figure()
    
    # Stack: Grid + Gen + BESS discharge
    fig.add_trace(go.Scatter(
        x=hours, y=grid_import,
        name='Grid Import',
        fill='tozeroy',
        stackgroup='supply'
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=gen_output,
        name='Generation',
        fill='tonexty',
        stackgroup='supply'
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=discharge,
        name='BESS Discharge',
        fill='tonexty',
        stackgroup='supply'
    ))
    
    # Load line
    fig.add_trace(go.Scatter(
        x=hours, y=total_load,
        name='Total Load',
        line=dict(color='black', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Power (MW)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def page_reliability_reserves(results: Dict[str, Any]):
    """Reliability & Reserves page."""
    st.header("🛡️ Reliability & Reserves")
    
    opt_results = results["results"]
    
    if not opt_results["solved"]:
        st.error("Optimization not solved")
        return
    
    ts = opt_results["timeseries"]
    hours = ts["hours"]
    reliability = opt_results["reliability"]
    bess_info = results["resources"]["bess"]
    
    # Reliability Metrics
    st.subheader("Reliability Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unserved = reliability["unserved_mwh"]
        color = "normal" if unserved < 0.001 else "inverse"
        st.metric("Unserved Energy", f"{unserved:.3f} MWh", 
                  delta="0" if unserved < 0.001 else f"-{unserved:.3f}",
                  delta_color=color)
    
    with col2:
        st.metric("Max Unserved", f"{reliability['max_unserved_mw']:.2f} MW")
    
    with col3:
        intervals_curtail = reliability["intervals_with_curtailment"]
        st.metric("Intervals w/ Curtailment", intervals_curtail)
    
    with col4:
        achieved = "✅ Yes" if reliability["ride_through_achieved"] else "❌ No"
        st.metric("Ride-Through Achieved", achieved)
    
    if reliability["unserved_mwh"] > 0.001:
        st.error("⚠️ CRITICAL: Unserved energy detected! System reliability compromised.")
    else:
        st.success("✅ All load served throughout horizon.")
    
    st.divider()
    
    # SOC Reserve Line
    st.subheader("SOC vs Reserve Floor")
    
    soc = np.array(ts["bess_soc"]) * 100
    
    fig = go.Figure()
    
    # SOC
    fig.add_trace(go.Scatter(
        x=hours, y=soc,
        name='Actual SOC',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    # Reserve floor
    if bess_info:
        reserve = bess_info["soc_reserve"] * 100
        fig.add_hline(y=reserve, line_dash="dash", line_color="red",
                      annotation_text=f"Reserve Floor: {reserve:.0f}%")
        
        # Min SOC floor
        soc_min = bess_info["soc_min"] * 100
        fig.add_hline(y=soc_min, line_dash="dot", line_color="gray",
                      annotation_text=f"Absolute Min: {soc_min:.0f}%")
    
    # Max SOC
    if bess_info:
        soc_max = bess_info["soc_max"] * 100
        fig.add_hline(y=soc_max, line_dash="dash", line_color="green",
                      annotation_text=f"Max SOC: {soc_max:.0f}%")
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="State of Charge (%)",
        yaxis=dict(range=[0, 100]),
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Check reserve violations
    if bess_info:
        reserve = bess_info["soc_reserve"] * 100
        violations = sum(1 for s in soc if s < reserve - 0.1)
        if violations > 0:
            st.warning(f"⚠️ SOC dropped below reserve floor in {violations} intervals")
        else:
            st.success("✅ SOC reserve maintained throughout horizon")
    
    st.divider()
    
    # Load Served Breakdown
    st.subheader("Critical vs Non-Critical Load Served")
    
    total_load = np.array(ts["total_load_mw"])
    critical_load = np.array(ts["critical_load_mw"])
    curtailed = np.array(ts["curtailed_mw"])
    unserved = np.array(ts["unserved_mw"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=critical_load,
        name='Critical Load',
        fill='tozeroy',
        line=dict(color='#d62728')
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=total_load,
        name='Total Load',
        line=dict(color='black', dash='dash')
    ))
    
    if max(curtailed) > 0.01:
        fig.add_trace(go.Scatter(
            x=hours, y=curtailed,
            name='Curtailed',
            line=dict(color='orange', width=2)
        ))
    
    if max(unserved) > 0.01:
        fig.add_trace(go.Scatter(
            x=hours, y=unserved,
            name='UNSERVED',
            line=dict(color='red', width=3)
        ))
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load (MW)",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Reserve Provision
    st.subheader("Reserve Provision by Product")
    
    reg_up = np.array(ts["reg_up_mw"])
    reg_down = np.array(ts["reg_down_mw"])
    spin = np.array(ts["spin_mw"])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=reg_up,
        name='Reg-Up',
        fill='tozeroy',
        stackgroup='reserves'
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=reg_down,
        name='Reg-Down',
        fill='tonexty',
        stackgroup='reserves'
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=spin,
        name='Spinning',
        fill='tonexty',
        stackgroup='reserves'
    ))
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Reserve Provision (MW)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def page_economics(results: Dict[str, Any]):
    """Economics page."""
    st.header("💰 Economics")
    
    opt_results = results["results"]
    
    if not opt_results["solved"]:
        st.error("Optimization not solved")
        return
    
    economics = opt_results["economics"]
    ts = opt_results["timeseries"]
    hours = ts["hours"]
    
    # Summary Metrics
    st.subheader("Economic Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Revenue",
                  f"${economics['revenue']['total_revenue']:,.0f}")
        st.caption("Energy Export + AS")
    
    with col2:
        st.metric("Total Cost",
                  f"${economics['costs']['total_cost']:,.0f}")
        st.caption("Import + Fuel + Degradation")
    
    with col3:
        net = economics["net_value"]
        color = "normal" if net >= 0 else "inverse"
        st.metric("Net Value",
                  f"${net:,.0f}",
                  delta=f"${net/economics['horizon_hours']:,.0f}/hr",
                  delta_color=color)
    
    st.divider()
    
    # Revenue Breakdown
    st.subheader("Revenue Stack")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        revenue = economics["revenue"]
        fig = go.Figure(data=[
            go.Bar(
                x=["Energy Export", "Ancillary Services"],
                y=[revenue["energy_export"], revenue["ancillary_services"]],
                marker_color=['#2ca02c', '#1f77b4']
            )
        ])
        fig.update_layout(
            yaxis_title="Revenue ($)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Revenue Breakdown**")
        st.write(f"Energy Export: ${revenue['energy_export']:,.2f}")
        st.write(f"Ancillary Services: ${revenue['ancillary_services']:,.2f}")
        st.write(f"**Total: ${revenue['total_revenue']:,.2f}**")
    
    st.divider()
    
    # Cost Breakdown
    st.subheader("Cost Stack")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        costs = economics["costs"]
        fig = go.Figure(data=[
            go.Bar(
                x=["Energy Import", "Fuel", "Degradation", "Curtailment"],
                y=[costs["energy_import"], costs["fuel"], 
                   costs["degradation"], costs["curtailment"]],
                marker_color=['#d62728', '#ff7f0e', '#9467bd', '#8c564b']
            )
        ])
        fig.update_layout(
            yaxis_title="Cost ($)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Cost Breakdown**")
        st.write(f"Energy Import: ${costs['energy_import']:,.2f}")
        st.write(f"Fuel: ${costs['fuel']:,.2f}")
        st.write(f"Degradation: ${costs['degradation']:,.2f}")
        st.write(f"Curtailment: ${costs['curtailment']:,.2f}")
        st.write(f"**Total: ${costs['total_cost']:,.2f}**")
    
    st.divider()
    
    # Cumulative Value Over Time
    st.subheader("Cumulative Net Value Over Time")
    
    intervals = opt_results["intervals"]
    cumulative = []
    running_total = 0
    
    for interval in intervals:
        net = (
            interval["energy_revenue"] + interval["as_revenue"]
            - interval["energy_cost"] - interval["fuel_cost"]
            - interval["degradation_cost"] - interval["curtailment_cost"]
        )
        running_total += net
        cumulative.append(running_total)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=cumulative,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2ca02c' if cumulative[-1] >= 0 else '#d62728')
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Cumulative Net Value ($)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Sensitivity Analysis
    st.subheader("Sensitivity Toggles")
    
    st.info("Adjust parameters below to see impact on economics (re-run required)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fuel_price = st.slider("Fuel Price ($/MMBtu)", 1.0, 10.0, 3.0, 0.5)
    
    with col2:
        soc_reserve = st.slider("SOC Reserve (%)", 10, 50, 25, 5)
    
    with col3:
        ride_through = st.slider("Ride-Through (min)", 5, 30, 15, 5)
    
    if st.button("Re-run with New Parameters"):
        st.info("Parameter sensitivity would trigger re-optimization...")


def page_decision_explainer(results: Dict[str, Any]):
    """Decision Explainer page."""
    st.header("🔍 Decision Explainer")
    
    opt_results = results["results"]
    
    if not opt_results["solved"]:
        st.error("Optimization not solved")
        return
    
    ts = opt_results["timeseries"]
    hours = ts["hours"]
    intervals = opt_results["intervals"]
    
    # Timeline selector
    st.subheader("Select Timestep for Analysis")
    
    n_intervals = len(intervals)
    selected_idx = st.slider("Interval", 0, n_intervals - 1, n_intervals // 2)
    
    selected = intervals[selected_idx]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Hour", f"{selected['timestamp_hours']:.2f}")
        st.metric("Grid Import", f"{selected['grid_import_mw']:.1f} MW")
        st.metric("Grid Export", f"{selected['grid_export_mw']:.1f} MW")
        st.metric("BESS Charge", f"{selected['bess_charge_mw']:.1f} MW")
        st.metric("BESS Discharge", f"{selected['bess_discharge_mw']:.1f} MW")
        st.metric("SOC", f"{selected['bess_soc']*100:.1f}%")
        st.metric("Gen Output", f"{selected['gen_output_mw']:.1f} MW")
    
    with col2:
        st.markdown("### Why These Decisions?")
        
        binding = selected["binding_constraints"]
        
        # Generate explanations
        explanations = []
        
        # BESS explanations
        if selected["bess_discharge_mw"] > 0.1:
            if "energy_price_high" in binding:
                explanations.append(
                    "**Battery Discharging**: Energy price is above average, "
                    "making discharge economically attractive."
                )
            else:
                explanations.append(
                    "**Battery Discharging**: Required for load balance or "
                    "reserve provision."
                )
        elif selected["bess_charge_mw"] > 0.1:
            if "energy_price_low" in binding:
                explanations.append(
                    "**Battery Charging**: Energy price is below average, "
                    "storing energy for later discharge."
                )
            else:
                explanations.append(
                    "**Battery Charging**: Building SOC for reserve requirements."
                )
        else:
            if "soc_reserve" in binding:
                explanations.append(
                    "**Battery Idle**: SOC at reserve floor - cannot discharge "
                    "for market without violating ride-through requirement."
                )
            elif "soc_max" in binding:
                explanations.append(
                    "**Battery Idle**: SOC at maximum - cannot charge further."
                )
            else:
                explanations.append(
                    "**Battery Idle**: Optimal strategy is to hold energy "
                    "for higher-value periods or future reserve needs."
                )
        
        # Generator explanations
        if selected["gen_output_mw"] > 0.1:
            if "grid_import_max" in binding:
                explanations.append(
                    "**Generator Running**: Grid import limit is binding. "
                    "Generator needed to meet load."
                )
            else:
                explanations.append(
                    "**Generator Running**: Grid energy cost exceeds generation "
                    "cost, or providing reliability margin."
                )
        else:
            explanations.append(
                "**Generator Offline**: Grid import sufficient and more economic "
                "than running generators."
            )
        
        # Grid explanations
        if selected["grid_export_mw"] > 0.1:
            explanations.append(
                "**Exporting to Grid**: Excess generation being sold at "
                f"current market price."
            )
        
        # Curtailment
        if selected["curtailed_mw"] > 0.01:
            explanations.append(
                f"**⚠️ Load Curtailed**: {selected['curtailed_mw']:.1f} MW "
                "curtailed due to supply constraints or economics."
            )
        
        if selected["unserved_mw"] > 0.01:
            explanations.append(
                f"**🚨 UNSERVED LOAD**: {selected['unserved_mw']:.1f} MW "
                "could not be served! Check capacity and constraints."
            )
        
        for exp in explanations:
            st.markdown(exp)
            st.write("")
        
        st.divider()
        
        st.markdown("### Binding Constraints")
        if binding:
            for constraint in binding:
                st.code(constraint)
        else:
            st.info("No significant constraints binding at this interval.")
    
    st.divider()
    
    # Constraint binding frequency
    st.subheader("Constraint Binding Frequency")
    
    binding_counts = opt_results.get("binding_constraints", {})
    
    if binding_counts:
        df = pd.DataFrame({
            "Constraint": list(binding_counts.keys()),
            "Intervals Binding": list(binding_counts.values())
        }).sort_values("Intervals Binding", ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df["Constraint"],
                y=df["Intervals Binding"],
                marker_color='#1f77b4'
            )
        ])
        
        fig.update_layout(
            xaxis_title="Constraint",
            yaxis_title="Number of Intervals Binding",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No binding constraint data available")


def main():
    """Main application."""
    st.title("⚡ Data Center Energy Platform")
    st.caption("Production-Grade Planning and Dispatch Tool")
    
    # Sidebar
    with st.sidebar:
        st.header("Case Selection")
        
        # File upload or example
        case_option = st.radio(
            "Load case from:",
            ["Example Case", "Upload JSON"]
        )
        
        if case_option == "Example Case":
            example_path = Path(__file__).parent.parent.parent / "examples" / "case_ercot_50mw.json"
            if example_path.exists():
                case_file = str(example_path)
                st.success(f"Using: {example_path.name}")
            else:
                st.error("Example case not found")
                case_file = None
        else:
            uploaded = st.file_uploader("Upload JSON case file", type=["json"])
            if uploaded:
                # Save to temp file
                temp_path = Path("/tmp/uploaded_case.json")
                with open(temp_path, "w") as f:
                    f.write(uploaded.read().decode())
                case_file = str(temp_path)
                st.success("File uploaded")
            else:
                case_file = None
        
        st.divider()
        
        if case_file:
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Solving..."):
                    try:
                        results = run_optimization(case_file)
                        st.success("Optimization complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            [
                "📊 System Overview",
                "🔧 Electrical Feasibility",
                "📈 Dispatch Timeline",
                "🛡️ Reliability & Reserves",
                "💰 Economics",
                "🔍 Decision Explainer"
            ]
        )
    
    # Main content
    results = load_results()
    
    if results is None:
        st.info("👈 Select a case and run optimization to see results.")
        
        st.markdown("""
        ## Welcome to the Data Center Energy Platform
        
        This tool provides **production-grade planning and dispatch optimization** for data centers with:
        
        - ⚡ **Grid Interconnection** (POI with MW/MVA limits)
        - 🔋 **Battery Energy Storage** (BESS with SOC management)
        - 🏭 **On-site Generation** (Gas turbines/engines with start-up logic)
        - 📊 **Load Flexibility** (Critical, non-critical, curtailable loads)
        - 💹 **Market Participation** (Energy and ancillary services)
        
        ### Getting Started
        
        1. Select an example case or upload your own JSON case file
        2. Click "Run Optimization" to solve the dispatch problem
        3. Explore the results through the various analysis pages
        
        ### Features
        
        - **MILP Optimization** using Pyomo with HiGHS solver
        - **ISO-Agnostic** design with adapter pattern (ERCOT, Generic, extensible)
        - **Electrical Feasibility** checks (POI, transformer, PCS limits)
        - **Reliability Constraints** (SOC reserve, ride-through, N+1 logic)
        - **Full Explainability** - understand why every decision was made
        """)
        
    else:
        # Route to selected page
        if "System Overview" in page:
            page_system_overview(results)
        elif "Electrical Feasibility" in page:
            page_electrical_feasibility(results)
        elif "Dispatch Timeline" in page:
            page_dispatch_timeline(results)
        elif "Reliability" in page:
            page_reliability_reserves(results)
        elif "Economics" in page:
            page_economics(results)
        elif "Decision Explainer" in page:
            page_decision_explainer(results)


if __name__ == "__main__":
    main()
