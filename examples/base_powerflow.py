"""
Base pandapower power flow (single snapshot).

This is a minimal, planning-level example:
Grid slack -> HV/MV transformer -> data center MV bus -> PQ load + PQ BESS.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path when running from /examples
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network import BaseSystemSpec, build_base_network, run_powerflow, summarize_base_results


def main() -> None:
    net = build_base_network(BaseSystemSpec(), bess_p_mw=0.0, bess_q_mvar=0.0)
    run_powerflow(net)
    s = summarize_base_results(net)

    print("=== Bus voltages (pu) ===")
    print(s["bus"].to_string())
    print("\n=== Transformer loading (%) ===")
    print(s["trafo"][["loading_percent"]].to_string())
    print("\n=== Grid import (MW / MVAr) ===")
    print(s["grid"].to_string())


if __name__ == "__main__":
    main()

