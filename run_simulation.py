"""
Run a planning-level feasibility simulation (hourly time series).

This script is intentionally conservative:
- Positive-sequence / RMS only (balanced three-phase abstraction)
- No inverter controls or EMT
- Rule-based dispatch constrained by voltage and transformer loading
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp

from dispatch import DispatchLimits, dispatch_bess_rule_based
from network import BaseSystemSpec, build_base_network, q_from_pf, run_powerflow, summarize_base_results
from scenarios import (
    ArchitectureAssumptions,
    DEFAULT_ARCHITECTURES,
    default_hourly_profile,
)


@dataclass(frozen=True)
class SimulationConfig:
    hours: int = 24
    start: str = "2026-01-01 00:00:00"
    tz: str | None = None

    # Dispatch screening thresholds (planning constraints)
    vmin_pu: float = 0.95
    vmax_pu: float = 1.05
    trafo_loading_max_pct: float = 90.0
    load_ramp_max_mw_per_h: float = 10.0


def _apply_load(net: pp.pandapowerNet, p_mw: float, pf: float, pf_sign: str) -> None:
    q_mvar = q_from_pf(p_mw, pf, pf_sign)  # for loads: lagging => +Q consumption
    net.load.loc[:, "p_mw"] = float(p_mw)
    net.load.loc[:, "q_mvar"] = float(q_mvar)


def _apply_bess(net: pp.pandapowerNet, p_mw: float, q_mvar: float) -> None:
    net.sgen.loc[:, "p_mw"] = float(p_mw)
    net.sgen.loc[:, "q_mvar"] = float(q_mvar)


def run_hourly_simulation(
    *,
    system: BaseSystemSpec,
    arch: ArchitectureAssumptions,
    sim: SimulationConfig,
    dispatch_limits: DispatchLimits,
    seed: int = 42,
    mode: str = "mixed",
    pf_override: float | None = None,
    pf_sign_override: str | None = None,
) -> pd.DataFrame:
    """
    Hourly loop:
    - Update load (P, PF)
    - Compute BESS dispatch (rule-based)
    - Run power flow
    - Log voltages, transformer loading, grid P/Q
    """

    idx = pd.date_range(sim.start, periods=sim.hours, freq="h", tz=sim.tz)
    prof = default_hourly_profile(idx=idx, system=system, arch=arch, seed=seed, mode=mode)
    if pf_override is not None:
        prof.loc[:, "load_pf"] = float(pf_override)
    if pf_sign_override is not None:
        prof.loc[:, "load_pf_sign"] = str(pf_sign_override)

    # Build once; mutate element setpoints each timestep.
    net = build_base_network(system)

    rows: list[dict] = []
    prev_p = None

    for ts, r in prof.iterrows():
        load_p = float(r["load_p_mw"])
        load_pf = float(r["load_pf"])
        pf_sign = str(r["load_pf_sign"])

        # Ramp screening is based on the underlying load profile (not "net-of-bess").
        ramp_mw = None if prev_p is None else (load_p - prev_p)

        # Set load first (so dispatch can evaluate feasibility with the same load)
        _apply_load(net, load_p, load_pf, pf_sign)

        # Propose a feasibility-driven BESS setpoint (open-loop per hour).
        bess_p, bess_q, decision = dispatch_bess_rule_based(
            net=net,
            system=system,
            arch=arch,
            load_p_mw=load_p,
            load_ramp_mw_per_h=ramp_mw,
            limits=dispatch_limits,
            vmin_pu=sim.vmin_pu,
            vmax_pu=sim.vmax_pu,
            trafo_loading_max_pct=sim.trafo_loading_max_pct,
            load_ramp_max_mw_per_h=sim.load_ramp_max_mw_per_h,
        )

        _apply_bess(net, bess_p, bess_q)

        # Run PF and record results
        run_powerflow(net)
        summary = summarize_base_results(net)

        vm_dc = float(summary["bus"].loc["DATA_CENTER_34p5kV", "vm_pu"])
        vm_grid = float(summary["bus"].loc["GRID_230kV", "vm_pu"])
        trafo_loading = float(summary["trafo"].iloc[0]["loading_percent"])
        grid_p = float(summary["grid"].iloc[0]["p_mw"])
        grid_q = float(summary["grid"].iloc[0]["q_mvar"])

        rows.append(
            {
                "time": ts,
                "arch": arch.name,
                "load_p_mw": load_p,
                "load_pf": load_pf,
                "load_q_mvar": float(net.load.loc[0, "q_mvar"]),
                "bess_p_mw": bess_p,
                "bess_q_mvar": bess_q,
                "vm_pu_dc": vm_dc,
                "vm_pu_grid": vm_grid,
                "trafo_loading_pct": trafo_loading,
                "grid_p_mw": grid_p,
                "grid_q_mvar": grid_q,
                "dispatch_decision": decision,
                "load_ramp_mw_per_h": ramp_mw if ramp_mw is not None else 0.0,
            }
        )

        prev_p = load_p

    df = pd.DataFrame(rows).set_index("time")
    return df


def plot_results(df: pd.DataFrame, *, out_prefix: str | None = None) -> None:
    """Utility-style time plots (voltage, transformer loading, grid P/Q)."""

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df.index, df["vm_pu_dc"], label="Data center bus (pu)")
    ax1.axhline(0.95, linestyle="--", linewidth=1, color="k", alpha=0.5)
    ax1.axhline(1.05, linestyle="--", linewidth=1, color="k", alpha=0.5)
    ax1.set_ylabel("Voltage (pu)")
    ax1.set_title("Data center MV bus voltage")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df.index, df["trafo_loading_pct"], label="Transformer loading (%)")
    ax2.axhline(90.0, linestyle="--", linewidth=1, color="k", alpha=0.5, label="90% screen")
    ax2.set_ylabel("Loading (%)")
    ax2.set_title("HV/MV transformer loading")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df.index, df["grid_p_mw"], label="Grid P (MW)")
    ax3.plot(df.index, df["grid_q_mvar"], label="Grid Q (MVAr)")
    ax3.set_ylabel("Import (+) / Export (-)")
    ax3.set_title("Grid import at slack bus (P/Q)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()

    if out_prefix:
        fig1.savefig(f"{out_prefix}_voltage.png", dpi=150)
        fig2.savefig(f"{out_prefix}_trafo_loading.png", dpi=150)
        fig3.savefig(f"{out_prefix}_grid_pq.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Planning-level pandapower feasibility screening")
    parser.add_argument("--arch", default="ac_racks", choices=sorted(DEFAULT_ARCHITECTURES.keys()))
    parser.add_argument("--mode", default="mixed", choices=["mixed", "training", "inference"])
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output prefix for CSV/PNG (no extension)")
    parser.add_argument("--pf", type=float, default=None, help="Override load power factor (constant for all hours)")
    parser.add_argument(
        "--pf-sign",
        default=None,
        choices=["lagging", "leading"],
        help="Override load PF sign (default: lagging)",
    )
    parser.add_argument(
        "--pf-sweep",
        default=None,
        help="Comma-separated PF values to run sensitivity (requires --out). Example: 0.94,0.96,0.97,0.995",
    )
    args = parser.parse_args()

    system = BaseSystemSpec()
    arch = DEFAULT_ARCHITECTURES[args.arch]
    sim = SimulationConfig(hours=args.hours)

    dispatch_limits = DispatchLimits(
        bess_p_min_mw=system.bess_p_min_mw,
        bess_p_max_mw=system.bess_p_max_mw,
        bess_q_min_mvar=arch.bess_q_min_mvar,
        bess_q_max_mvar=arch.bess_q_max_mvar,
    )

    if args.pf_sweep and not args.out:
        raise SystemExit("--pf-sweep requires --out so results can be saved per case.")

    pf_sweep = None
    if args.pf_sweep:
        pf_sweep = [float(x.strip()) for x in args.pf_sweep.split(",") if x.strip()]
        if not pf_sweep:
            raise SystemExit("--pf-sweep was provided but no PF values were parsed.")

    runs: list[tuple[str, pd.DataFrame]] = []
    if pf_sweep is None:
        tag = f"{args.arch}_{args.mode}"
        df = run_hourly_simulation(
            system=system,
            arch=arch,
            sim=sim,
            dispatch_limits=dispatch_limits,
            seed=args.seed,
            mode=args.mode,
            pf_override=args.pf,
            pf_sign_override=args.pf_sign,
        )
        runs.append((tag, df))
    else:
        for pf in pf_sweep:
            tag = f"{args.arch}_{args.mode}_pf{pf:.3f}"
            df = run_hourly_simulation(
                system=system,
                arch=arch,
                sim=sim,
                dispatch_limits=dispatch_limits,
                seed=args.seed,
                mode=args.mode,
                pf_override=pf,
                pf_sign_override=args.pf_sign,
            )
            runs.append((tag, df))

    for tag, df in runs:
        # Print a compact engineering table (first/last few rows)
        cols = [
            "load_p_mw",
            "load_pf",
            "bess_p_mw",
            "vm_pu_dc",
            "trafo_loading_pct",
            "grid_p_mw",
            "grid_q_mvar",
            "dispatch_decision",
        ]
        print(f"\n=== Case: {tag} ===")
        print("\nFeasibility summary (head):")
        print(df[cols].head(6).to_string())
        print("\nFeasibility summary (tail):")
        print(df[cols].tail(6).to_string())

        # Interpret what limited dispatch
        decision_counts = df["dispatch_decision"].value_counts().to_dict()
        print("\nDispatch limiting reasons (counts):")
        for k, v in decision_counts.items():
            print(f"- {k}: {v}")

        vmin = df["vm_pu_dc"].min()
        vmax = df["vm_pu_dc"].max()
        tmax = df["trafo_loading_pct"].max()
        print("\nKey screening extrema:")
        print(f"- Min/Max DC bus voltage (pu): {vmin:.4f} / {vmax:.4f}")
        print(f"- Max transformer loading (%): {tmax:.2f}")

        # Simple interpretation: which screen was "closest" to binding
        vmin_margin = vmin - sim.vmin_pu
        vmax_margin = sim.vmax_pu - vmax
        trafo_margin = sim.trafo_loading_max_pct - tmax
        closest = sorted(
            [
                ("voltage_low_margin_pu", vmin_margin),
                ("voltage_high_margin_pu", vmax_margin),
                ("transformer_loading_margin_pct", trafo_margin),
            ],
            key=lambda x: x[1],
        )[0]
        print("\nDispatch interpretation (screening):")
        print(f"- Closest constraint margin: {closest[0]} = {closest[1]:.4f}")

        if args.out:
            prefix = f"{args.out}_{tag}"
            df.to_csv(f"{prefix}.csv")
            plot_results(df, out_prefix=prefix)
        else:
            plot_results(df, out_prefix=None)
            plt.show()


if __name__ == "__main__":
    main()

