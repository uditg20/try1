"""
Data Center Energy Platform
===========================

Production-grade planning and dispatch tool for data centers with:
- On-site generation (gas turbines, diesel gensets)
- Battery Energy Storage Systems (BESS)
- Load flexibility (critical, non-critical, curtailable, shiftable)
- Grid interconnection with ISO market participation

Architecture:
- iso/: ISO market adapters (ERCOT, CAISO, PJM, etc.)
- topology/: Electrical feasibility layer
- resources/: Asset models (BESS, generators, loads)
- optimization/: MILP dispatch optimization
- ui/: Streamlit visualization interface
"""

__version__ = "1.0.0"
