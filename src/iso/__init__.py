"""
ISO Adapters
============

Each ISO adapter defines market-specific parameters:
- Dispatch interval (5-min / 15-min / hourly)
- Reserve and ancillary service products
- Energy-backing duration requirements
- Headroom rules (up/down regulation)
"""

from .base import ISOAdapter, ASProduct, ASProductSpec, MarketInterval
from .ercot import ERCOTAdapter
from .generic import GenericAdapter, get_iso_adapter

__all__ = [
    "ISOAdapter", 
    "ASProduct",
    "ASProductSpec",
    "MarketInterval",
    "ERCOTAdapter", 
    "GenericAdapter",
    "get_iso_adapter"
]
