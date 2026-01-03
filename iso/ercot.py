from __future__ import annotations

from typing import Dict, List

from .base import ISOAdapter, ReserveProduct


class ERCOTAdapter(ISOAdapter):
    """
    ERCOT-like adapter (simplified but explicit).

    Products implemented:
    - REG_UP: regulation up (short duration energy backing)
    - REG_DOWN: regulation down
    - RRS: responsive reserve service (contingency), longer duration

    Settlement:
    - energy: $/MWh
    - reserves: $/MW-h (case provides series; runner scales by dt)
    """

    iso_name = "ERCOT"

    def dispatch_interval_minutes(self) -> int:
        # Keep 15-min default for planning runs; case may override via runner if desired.
        return 15

    def reserve_products(self) -> List[ReserveProduct]:
        return [
            ReserveProduct(name="REG_UP", direction="up", duration_minutes=15),
            ReserveProduct(name="REG_DOWN", direction="down", duration_minutes=15),
            ReserveProduct(name="RRS", direction="up", duration_minutes=30),
        ]

    def reserve_price_series_keys(self) -> Dict[str, str]:
        return {
            "REG_UP": "reg_up_price_$per_mw_h",
            "REG_DOWN": "reg_down_price_$per_mw_h",
            "RRS": "rrs_price_$per_mw_h",
        }

