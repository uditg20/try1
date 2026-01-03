from __future__ import annotations

import math
from typing import List, Tuple


def pf_q_limit_from_pfmin(p_mw: float, pf_min: float) -> float:
    """
    For given active power P and minimum PF magnitude, compute |Q| limit:
      PF = |P| / sqrt(P^2 + Q^2) >= pf_min  =>  |Q| <= |P| * tan(arccos(pf_min))
    """

    if pf_min <= 0 or pf_min > 1.0:
        raise ValueError("pf_min must be in (0, 1]")
    return abs(p_mw) * math.tan(math.acos(pf_min))


def linear_mva_facets(n_facets: int = 12) -> List[Tuple[float, float]]:
    """
    Return (a, b) pairs defining a polygon approximation of the unit circle:
        a*P + b*Q <= 1  for each facet
    for P,Q >= 0 quadrant; we apply via abs(P), abs(Q) linearization externally.

    Implementation uses angles from 0..pi/2.
    """

    if n_facets < 4:
        raise ValueError("n_facets should be >= 4 for a reasonable approximation")

    facets: List[Tuple[float, float]] = []
    for k in range(n_facets):
        theta1 = (k / n_facets) * (math.pi / 2)
        theta2 = ((k + 1) / n_facets) * (math.pi / 2)
        # Supporting line to circle between theta1/theta2: use normal at midpoint angle
        theta = 0.5 * (theta1 + theta2)
        a = math.cos(theta)
        b = math.sin(theta)
        # For unit circle, max of a*P + b*Q over circle equals 1 (since (a,b) unit).
        facets.append((a, b))
    return facets

