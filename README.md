# Data Center Power System Sizing & Dispatch Tool

ISO-agnostic planning and dispatch engine for data centers with on-site
generation, BESS, load flexibility, and grid interconnection.

This is a **production-style planning + dispatch platform** with:

- **MILP optimization (Pyomo)** for dispatch and reserve provision
- **ISO adapter/plugin pattern** (ERCOT implemented end-to-end)
- **Explicit electrical screening** (MW/MVA/PF/Q, transformer and PCS MVA)
- **Operational realism** (generator start delay, ramps, no simultaneous charge/discharge)
- **Reliability** (critical load VoLL, SOC ride-through floor, simplified N+1 logic)
- **UI-first outputs** (structured JSON results + constraint-binding explainability)

## Repo architecture (required)

- `iso/`: ISO adapters (ERCOT implemented)
- `topology/`: electrical feasibility layer (POI and transformer screening)
- `resources/`: BESS / gas gen / load models
- `optimization/`: Pyomo MILP (`dispatch_milp.py`)
- `ui/`: Streamlit UI (cleanly separated from optimization)
- `runner.py`: loads a JSON case, runs adapter + model, returns structured results

## Quickstart

### Install

```bash
python3 -m pip install -r requirements.txt
```

### Run a case (CLI)

```bash
python3 runner.py cases/example_ercot.json --out cases/example_ercot_results.json
```

### Run the UI (Streamlit)

```bash
python3 -m streamlit run ui/app.py
```

Then set the case path to `cases/example_ercot.json` and click **Run MILP**.

## What the MILP explicitly enforces

- **Microgrid balance**: grid import/export + generation + BESS = served load
- **BESS**:
  - SOC dynamics with charge/discharge efficiencies
  - no simultaneous charge/discharge (binary)
  - PCS MVA screening (linearized \(P/Q\) coupling)
  - SOC floor = **market SOC reserve + ride-through energy for critical load**
- **Generator**:
  - integer unit commitment with **explicit start delay**
  - min/max output by available units
  - ramp limits (simplified but explicit)
  - **N+1 adequacy**: capacity >= critical load + largest unit margin (with BESS support)
- **POI + transformer**:
  - MW import/export caps
  - MVA screening (linearized polygon facets)
  - PF screening at POI (minimum PF magnitude)
- **Reserves**:
  - up/down headroom constraints for both gen and BESS
  - **energy-backing** constraints for BESS reserve products

## Outputs and explainability

`runner.py` returns a structured dict (JSON-serializable) containing:

- `dispatch` time series (grid, BESS, gen, load served/unserved)
- `electrical` screening time series (POI/transformer/PCS loading)
- `economics` breakdown (revenue and cost stacks)
- `explainability.binding_constraints[time]`: binding constraints by timestep (via primal slacks)
