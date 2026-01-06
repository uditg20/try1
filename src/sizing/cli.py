from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

from .models import SizingInputs
from .sizer import size_configuration


def _prompt_float(prompt: str, *, min_v: float | None = None, max_v: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            v = float(raw)
        except ValueError:
            print("Please enter a number.", file=sys.stderr)
            continue
        if min_v is not None and v < min_v:
            print(f"Must be >= {min_v}.", file=sys.stderr)
            continue
        if max_v is not None and v > max_v:
            print(f"Must be <= {max_v}.", file=sys.stderr)
            continue
        return v


def load_inputs(path: str | None, *, interactive: bool) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input JSON not found: {path}")
        data = json.loads(p.read_text())

    # Ask for required fields if missing
    if interactive:
        if "cfe_target" not in data:
            data["cfe_target"] = _prompt_float("Target CFE (0-1): ", min_v=0.0, max_v=1.0)

        load = data.get("load", {})
        if "load_mw" not in load:
            load["load_mw"] = _prompt_float("Data center load (MW): ", min_v=0.0001)
        data["load"] = load

        grid = data.get("grid", {})
        if "max_import_mw" not in grid:
            grid["max_import_mw"] = _prompt_float("Grid import limit at POI (MW): ", min_v=0.0001)
        if "poi_voltage_kv" not in grid:
            grid["poi_voltage_kv"] = _prompt_float("POI voltage (kV): ", min_v=0.1)
        data["grid"] = grid

    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Technical PV/Wind/BESS/Grid configuration tool (non-economic)."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to sizing input JSON. If omitted, use --interactive prompts.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sizing_output.json",
        help="Path to write sizing output JSON.",
    )
    parser.add_argument(
        "--case-output",
        default="case_sized.json",
        help="Path to write a case JSON with the recommended configuration.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing required inputs (asks for CFE).",
    )

    args = parser.parse_args(argv)

    try:
        raw = load_inputs(args.input, interactive=args.interactive)
        inputs = SizingInputs.model_validate(raw)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 2
    except ValidationError as e:
        print("Input validation error:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 2

    out = size_configuration(inputs)

    # Full output
    Path(args.output).write_text(out.model_dump_json(indent=2))
    # Case JSON output
    Path(args.case_output).write_text(json.dumps(out.case_json, indent=2))

    # Minimal console summary
    r = out.report
    print(f"Feasible: {r.feasible}")
    print(f"Achieved CFE: {r.achieved_cfe*100:.1f}% (target {inputs.cfe_target*100:.1f}%)")
    print(f"PV (AC): {out.pv_ac_mw:.1f} MW, Wind: {out.wind_mw:.1f} MW")
    print(f"BESS: {out.bess_power_mw:.1f} MW / {out.bess_energy_mwh:.1f} MWh (PCS {out.bess_pcs_mva:.1f} MVA)")
    print(f"Grid max import: {inputs.grid.max_import_mw:.1f} MW (max observed {r.max_grid_import_mw:.1f} MW)")
    print(f"POI MVA: {out.poi_mva_rating:.1f} MVA, Transformer firm MVA: {r.firm_transformer_mva:.1f} MVA")
    if r.violations:
        print("\nViolations:", file=sys.stderr)
        for v in r.violations:
            print(f"- {v}", file=sys.stderr)
    if r.warnings:
        print("\nWarnings:", file=sys.stderr)
        for w in r.warnings:
            print(f"- {w}", file=sys.stderr)

    return 0 if r.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())

