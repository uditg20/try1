# ⚡ Data Center Energy Platform

**Production-Grade Planning and Dispatch Tool for Data Centers**

A professional, ISO-agnostic platform for sizing and dispatching data center microgrids with on-site generation, battery energy storage (BESS), load flexibility, and grid interconnection.

## 🎯 Overview

This platform provides:

- **MILP Optimization** using Pyomo with HiGHS solver
- **Electrical Credibility** - enforces real limits (MW, MVA, PF, transformer ratings)
- **Operational Realism** - generator start-up logic, N+1 redundancy, SOC dynamics
- **Full Explainability** - understand why every dispatch decision was made
- **Professional UI** - Streamlit-based interface for planning teams, utilities, and lenders

## 🏗️ Architecture

```
├── src/
│   ├── iso/                  # ISO market adapters
│   │   ├── base.py          # Abstract base class
│   │   ├── ercot.py         # ERCOT-specific rules
│   │   └── generic.py       # Generic adapter + factory
│   │
│   ├── topology/             # Electrical feasibility layer
│   │   ├── poi.py           # Point of Interconnection
│   │   ├── transformer.py   # Transformer MVA screening
│   │   ├── bus.py           # Buses and PCS models
│   │   └── network.py       # Complete microgrid topology
│   │
│   ├── resources/            # Asset models
│   │   ├── bess.py          # Battery storage with SOC dynamics
│   │   ├── gas_gen.py       # Gas generators with start-up logic
│   │   └── load.py          # Critical/non-critical/curtailable loads
│   │
│   ├── optimization/         # MILP dispatch engine
│   │   ├── dispatch_milp.py # Pyomo optimization model
│   │   └── results.py       # Structured results container
│   │
│   └── ui/                   # Streamlit interface
│       └── app.py           # Multi-page dashboard
│
├── examples/
│   └── case_ercot_50mw.json # Example 50MW data center case
│
├── runner.py                 # Main entry point
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd data-center-energy-platform

# Install dependencies
pip install -r requirements.txt
```

## 🔌 Planning-Level Electrical Feasibility (pandapower)

This repo includes a **planning / positive-sequence / RMS** feasibility screening model for a data center interconnected through a **single HV/MV transformer**:

**Topology (fixed)**
- Grid (infinite source / slack bus)
- HV/MV transformer (e.g., 230 kV → 34.5 kV)
- Data center MV bus
  - Aggregated data center load (PQ, configurable PF)
  - BESS inverter (PQ setpoints; dispatch is rule-based and screened)

**Explicitly out of scope (by design)**
- EMT / switching transients / sub-cycle dynamics
- Inverter firmware (PLL, current control), voltage regulation loops
- Harmonics, flicker
- Protection coordination
- Unbalanced / negative-sequence effects

### Run the base power flow

The base case is a 50 MW aggregated data center load at PF≈0.97 (lagging) with BESS at 0 MW / 0 MVAr.

```bash
python3 examples/base_powerflow.py
```

Or as a one-liner:

```bash
python3 -c "from network import BaseSystemSpec, build_base_network, run_powerflow, summarize_base_results; net=build_base_network(BaseSystemSpec()); run_powerflow(net); s=summarize_base_results(net); print(s['bus']); print(s['trafo'][['loading_percent']]); print(s['grid'])"
```

### Run the hourly feasibility simulation

This runs an hourly loop that updates load (training vs inference assumptions), applies conservative BESS dispatch, runs a power flow each hour, and logs:
- bus voltages
- transformer loading
- grid MW/MVAr import

```bash
python3 run_simulation.py --arch ac_racks --mode mixed --hours 24 --out feasibility_demo
```

### PF sensitivity (screening)

```bash
python3 run_simulation.py --arch dc_800v --mode mixed --hours 24 --pf-sweep 0.94,0.97,0.995 --out pf_sensitivity
```

### Run Optimization

```bash
# Run with example case
python runner.py examples/case_ercot_50mw.json

# Save results to JSON
python runner.py examples/case_ercot_50mw.json --output results.json
```

### Launch UI

```bash
streamlit run src/ui/app.py
```

## 📊 UI Pages

### 1. System Overview
- ISO selection and case info
- Topology summary (POI, transformers, redundancy)
- Resource stack (Gen MW, BESS MW/MWh/MVA, loads)
- Ride-through guarantee calculation

### 2. Electrical Feasibility
- POI operating envelope (P vs MVA)
- Transformer loading over time
- PCS loading vs MVA limit
- Clear warnings when constraints bind

### 3. Dispatch Timeline
- Energy price signal
- BESS charge/discharge and SOC
- Generator output and units online
- Grid import/export
- Load vs supply stack

### 4. Reliability & Reserves
- SOC vs reserve floor visualization
- Critical vs non-critical load served
- Unserved energy flagging
- Reserve provision by product (Reg-Up, Reg-Down, Spin)

### 5. Economics
- Revenue stack (energy, AS)
- Cost stack (fuel, degradation, import, curtailment)
- Net value and cumulative value over time
- Sensitivity toggles

### 6. Decision Explainer
- Per-interval analysis
- "Why did the battery charge/discharge?"
- "Why did the generator start?"
- Binding constraint identification and frequency

## ⚙️ Optimization Model

### Objective Function

**Maximize Net Value:**
```
Revenue:
+ Energy export × LMP
+ Ancillary services (Reg-Up, Reg-Down, Spinning)

Costs:
- Energy import × LMP
- Generator fuel + VOM
- Battery degradation
- Curtailment cost
- Value of Lost Load (VoLL) penalty
```

### Key Constraints

| Constraint | Description |
|-----------|-------------|
| Power Balance | Gen + BESS + Grid = Load at each interval |
| SOC Dynamics | State of charge evolves with charge/discharge |
| SOC Limits | Min ≤ SOC ≤ Max |
| SOC Reserve | Market dispatch cannot violate ride-through reserve |
| No Simultaneous Charge/Discharge | Binary variable enforcement |
| Generator Bounds | Min/max output when online |
| Generator Start-up | Unit commitment and minimum run time |
| POI Limits | Import/export MW caps |
| Transformer MVA | Apparent power screening |
| PCS MVA | Inverter P-Q capability |
| Reserve Headroom | AS provision requires MW headroom |
| Reserve Energy Backing | AS provision requires energy to sustain |

## 🔌 ISO Adapters

The platform uses a clean adapter pattern for ISO-specific rules:

```python
from src.iso.generic import get_iso_adapter

# Get ERCOT adapter
adapter = get_iso_adapter("ERCOT")

# Access market parameters
interval = adapter.market_interval.rtm_interval_min  # 5 minutes
products = adapter.as_products  # RRS, ECRS, Reg-Up, Reg-Down, Non-Spin

# Calculate reserve requirements
req = adapter.get_reserve_requirement(
    product=ASProduct.RRS,
    capacity_mw=25,
    soc_mwh=50
)
```

### Implemented ISOs
- ✅ **ERCOT** - Full implementation with 5-min dispatch, all AS products
- ✅ **GENERIC** - Configurable adapter for other markets

### Planned ISOs
- 🔲 CAISO
- 🔲 PJM
- 🔲 MISO
- 🔲 NYISO
- 🔲 ISO-NE
- 🔲 SPP

## 📝 Case File Format

Cases are defined in JSON:

```json
{
  "name": "ERCOT 50MW Data Center",
  "iso": "ERCOT",
  
  "time": {
    "horizon_hours": 24,
    "interval_minutes": 15
  },
  
  "topology": {
    "poi": {
      "max_import_mw": 60,
      "max_export_mw": 20,
      "mva_rating": 75,
      "min_power_factor": 0.95
    },
    "transformers": [...]
  },
  
  "resources": {
    "bess": {
      "power_mw": 25,
      "energy_mwh": 100,
      "soc_reserve": 0.25
    },
    "generators": {...},
    "loads": [...]
  },
  
  "reliability": {
    "ride_through_minutes": 15,
    "voll_per_mwh": 50000
  },
  
  "prices": {
    "base_price": 35,
    "peak_hours": [7, 8, 9, 17, 18, 19, 20],
    "as_prices": {...}
  }
}
```

## 🛡️ Reliability Features

### SOC Reserve
- Configurable portion of BESS energy reserved for ride-through
- Market dispatch CANNOT violate this floor
- Emergency-only access below reserve

### Ride-Through Calculation
```
Ride-through time = (SOC - Reserve) × Energy / Critical Load
Must exceed: Generator start time
```

### N+1 Generator Logic
- Fleet configured with redundancy
- Firm capacity = Total - Largest unit
- Start-up sequencing constraints

## 🔍 Explainability

Every dispatch decision can be explained by:

1. **Binding Constraint** - which limit is active
2. **Opportunity Cost** - price signal driving behavior
3. **Reliability Requirement** - reserve/ride-through need

The Decision Explainer page shows:
- Which constraints bound at each interval
- Why the battery charged/discharged or was idle
- Why generators started or stayed offline
- Frequency of constraint binding across the horizon

## 📈 Example Results

Running the 50MW ERCOT example:

```
Optimization horizon: 24.0 hours
Intervals: 96
BESS: 25 MW / 100 MWh
Generators: 4 units, 60 MW total

Solver status: optimal
Solve time: 0.33 seconds

Objective (Net Value): $3,466.90

Economics Summary:
  Energy Revenue: $27,511.55
  AS Revenue: $17,565.42
  Energy Cost: $11,291.82
  Fuel Cost: $27,837.66
  Degradation: $480.60
  Net Value: $5,466.90

Reliability Summary:
  Unserved Energy: 0.000 MWh
  Ride-through Achieved: True
```

## 🔧 Development

### Running Tests

```bash
# Run the optimization
python runner.py examples/case_ercot_50mw.json

# Check for lint errors
ruff check src/
```

### Adding a New ISO

1. Create `src/iso/new_iso.py` extending `ISOAdapter`
2. Implement required properties and methods
3. Register in `src/iso/generic.py` factory
4. Update `src/iso/__init__.py` exports

## 📄 License

[License information here]

## 🤝 Contributing

[Contribution guidelines here]
