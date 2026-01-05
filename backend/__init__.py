"""
Data Center Power Flow Backend
==============================

Planning-level, positive-sequence power flow simulation for data centers
with BESS integration using pandapower.

SCOPE LIMITATIONS:
- Planning-level / positive-sequence / RMS abstraction ONLY
- Balanced three-phase system assumption
- No EMT, no inverter firmware, no protection coordination
- No operational controls or real-time simulation
- Results are EXPLANATORY, not for operational use

This module provides:
- network.py: pandapower network definition
- scenarios.py: Load and BESS scenario generators
- run_simulations.py: Time-series power flow execution
- export_results.py: JSON export for static UI
"""

__version__ = "1.0.0"
