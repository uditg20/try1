"""
Sizing / configuration toolchain.

This package provides a technical (non-economic) configurator that converts
high-level data center requirements (MW, grid limits, voltage, CFE target, etc.)
into an electrically credible PV / wind / BESS / grid interconnect configuration.
"""

from .models import SizingInputs, SizingOutputs
from .sizer import size_configuration

