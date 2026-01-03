import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runner import Runner

st.set_page_config(layout="wide", page_title="Data Center Energy Platform")

# Sidebar
st.sidebar.title("Configuration")
case_file = st.sidebar.text_input("Case File", "example_case.json")

if st.sidebar.button("Run Dispatch"):
    with st.spinner("Running Optimization..."):
        try:
            runner = Runner(case_file)
            output = runner.run()
            st.session_state['results'] = output['results']
            st.session_state['inputs'] = output['inputs']
            st.session_state['status'] = output['status']
            st.success("Optimization Successful!")
        except Exception as e:
            st.error(f"Error: {e}")

if 'results' not in st.session_state:
    st.info("Please run the dispatch to see results.")
    st.stop()

results = st.session_state['results']
inputs = st.session_state['inputs']

# Navigation
page = st.sidebar.radio("Navigate", [
    "System Overview", 
    "Electrical Feasibility", 
    "Dispatch Timeline", 
    "Reliability & Reserves", 
    "Economics",
    "Decision Explainer"
])

def plot_area_stack(df, x_col, y_cols, title, y_axis_title):
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], stackgroup='one', name=col
        ))
    fig.update_layout(title=title, yaxis_title=y_axis_title)
    st.plotly_chart(fig, use_container_width=True)

if page == "System Overview":
    st.title("System Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ISO", inputs['iso'])
        st.metric("Horizon", f"{len(results)} Intervals")
    with col2:
        st.metric("BESS Capacity", f"{inputs['bess'].capacity_mw} MW / {inputs['bess'].capacity_mwh} MWh")
        st.metric("Gen Capacity", f"{inputs['gen'].total_capacity_mw} MW")
    with col3:
        st.metric("Peak Load", f"{max(inputs['load'].forecast_mw):.2f} MW")
        st.metric("POI Limit", f"{inputs['topology'].poi.import_limit_mw} MW")

    st.subheader("Resource Stack")
    # Simple table or chart of static resources
    
    st.subheader("Summary Stats")
    total_load = results['load_served'].sum()
    total_import = results['grid_import'].sum()
    total_export = results['grid_export'].sum()
    total_gen = results['gen_mw'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Load Served (MWh)", f"{total_load/4:.1f}") # Assuming 15m intervals
    col2.metric("Grid Import (MWh)", f"{total_import/4:.1f}")
    col3.metric("Grid Export (MWh)", f"{total_export/4:.1f}")
    col4.metric("Gen Output (MWh)", f"{total_gen/4:.1f}")

elif page == "Electrical Feasibility":
    st.title("Electrical Feasibility")
    
    st.subheader("POI Loading")
    # Import vs Limit, Export vs Limit
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['t'], y=results['grid_import'], name="Import", fill='tozeroy'))
    fig.add_trace(go.Scatter(x=results['t'], y=results['grid_export'], name="Export", fill='tozeroy'))
    
    # Add limits
    limit_imp = inputs['topology'].poi.import_limit_mw
    limit_exp = inputs['topology'].poi.export_limit_mw
    fig.add_hline(y=limit_imp, line_dash="dash", annotation_text="Import Limit")
    fig.add_hline(y=limit_exp, line_dash="dash", annotation_text="Export Limit") # Note: Export is usually positive in my model but might need checking
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Transformer Loading")
    st.write("Assuming single path for simplicity (Sum of flows)")
    # Flow = Import - Export ? Or just max(Import, Export)?
    # Actually current through transformer is net flow.
    net_flow = results['grid_import'] - results['grid_export']
    fig_tx = go.Figure()
    fig_tx.add_trace(go.Scatter(x=results['t'], y=net_flow.abs(), name="|Net Flow|"))
    fig_tx.add_hline(y=inputs['topology'].transformers[0].capacity_mva, line_color="red", annotation_text="TX Limit MVA")
    st.plotly_chart(fig_tx, use_container_width=True)

elif page == "Dispatch Timeline":
    st.title("Dispatch Timeline")
    
    st.subheader("Power Balance")
    # Stacked Area Chart
    # Supply = Import + Gen + Discharge
    # Demand = Load + Charge + Export
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['t'], y=results['grid_import'], stackgroup='supply', name='Import'))
    fig.add_trace(go.Scatter(x=results['t'], y=results['gen_mw'], stackgroup='supply', name='Gen'))
    fig.add_trace(go.Scatter(x=results['t'], y=results['bess_discharge'], stackgroup='supply', name='BESS Discharge'))
    
    fig.add_trace(go.Scatter(x=results['t'], y=results['load_served'] * -1, stackgroup='demand', name='Load'))
    fig.add_trace(go.Scatter(x=results['t'], y=results['bess_charge'] * -1, stackgroup='demand', name='BESS Charge'))
    fig.add_trace(go.Scatter(x=results['t'], y=results['grid_export'] * -1, stackgroup='demand', name='Export'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("BESS SOC")
    fig_soc = px.line(results, x='t', y='bess_soc', title="State of Charge (MWh)")
    fig_soc.add_hline(y=inputs['bess'].capacity_mwh, line_dash="dash", annotation_text="Max Capacity")
    fig_soc.add_hline(y=inputs['bess'].soc_min_reserve * inputs['bess'].capacity_mwh, line_color="red", annotation_text="Min Reserve")
    st.plotly_chart(fig_soc, use_container_width=True)

elif page == "Reliability & Reserves":
    st.title("Reliability & Reserves")
    
    st.subheader("Unserved Load")
    if results['load_unserved'].sum() > 0:
        st.error(f"Warning: {results['load_unserved'].sum()} MW unserved load detected!")
        st.bar_chart(results['load_unserved'])
    else:
        st.success("No unserved load.")

    st.subheader("Ancillary Services Provision")
    as_cols = [c for c in results.columns if c.startswith("as_")]
    if as_cols:
        plot_area_stack(results, 't', as_cols, "AS Schedule", "MW")
    else:
        st.write("No AS products modeled.")

elif page == "Economics":
    st.title("Economics")
    st.write("Detailed financial breakdown would go here.")
    # Calculate revenue/cost again for visualization if not stored
    # Simplified view
    st.metric("Objective Function Value", "See optimization logs")

elif page == "Decision Explainer":
    st.title("Decision Explainer")
    
    selected_time = st.slider("Select Time Step", min_value=0, max_value=len(results)-1, value=0)
    
    row = results.iloc[selected_time]
    
    st.markdown(f"### Snapshot at t={selected_time}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**State**")
        st.write(f"Load: {row['load_served']:.2f} MW")
        st.write(f"Gen: {row['gen_mw']:.2f} MW")
        st.write(f"Import: {row['grid_import']:.2f} MW")
        st.write(f"BESS SOC: {row['bess_soc']:.2f} MWh")
    
    with col2:
        st.write("**Constraints Analysis**")
        # Logic to explain
        if row['bess_soc'] <= (inputs['bess'].soc_min_reserve * inputs['bess'].capacity_mwh) + 0.1:
            st.warning("BESS at minimum SOC reserve.")
        
        if row['gen_mw'] >= (inputs['gen'].total_capacity_mw - 0.1):
            st.warning("Generator at max capacity.")
            
        if row['grid_import'] >= (inputs['topology'].poi.import_limit_mw - 0.1):
            st.warning("Import limit binding.")
            
    st.info("In a full implementation, shadow prices (dual variables) from the solver would be displayed here to show the marginal value of each constraint.")

