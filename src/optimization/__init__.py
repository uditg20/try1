"""
Optimization Module
===================

MILP-based dispatch optimization using Pyomo:
- Rolling horizon capable
- Multi-objective: energy, AS, fuel, degradation
- Constraint-driven with full explainability
"""

from .dispatch_milp import DispatchOptimizer
from .results import OptimizationResults

__all__ = ["DispatchOptimizer", "OptimizationResults"]
