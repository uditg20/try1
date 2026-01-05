# ⚡ Data Center Power Flow Planning Study

**Planning-Level Electrical Simulation for Data Centers with BESS**

[![Static Badge](https://img.shields.io/badge/Abstraction-Planning%20Level-blue)](https://github.com)
[![Static Badge](https://img.shields.io/badge/Power%20Flow-Positive%20Sequence-green)](https://github.com)
[![Static Badge](https://img.shields.io/badge/UI-Static%20React-orange)](https://github.com)

---

## 🎯 Purpose

This tool provides **planning-level** power flow analysis for data center electrical systems with battery energy storage (BESS). It is designed for:

- Sizing and capacity planning studies
- Architecture comparison (AC vs DC distribution)
- BESS integration impact assessment
- Voltage and loading screening studies

---

## ⚠️ IMPORTANT DISCLAIMERS

### What This Tool IS

- ✅ **Planning-level analysis** using positive-sequence, balanced 3-phase, steady-state power flow
- ✅ **Educational and explanatory** tool for understanding system behavior
- ✅ **Scenario comparison** for sizing and architecture decisions
- ✅ **Open-source and reproducible** analysis methodology

### What This Tool IS NOT

- ❌ **NOT for operational dispatch decisions**
- ❌ **NOT for interconnection approval or utility submittals**
- ❌ **NOT for protection coordination or relay settings**
- ❌ **NOT for transient/EMT analysis**
- ❌ **NOT for inverter control design or firmware validation**
- ❌ **NOT a substitute for professional engineering review**

### Abstraction Level

| Aspect | This Tool | Full Study |
|--------|-----------|------------|
| Power Flow | Positive-sequence AC | Unbalanced 3-phase |
| Time Domain | Steady-state | EMT + steady-state |
| Load Model | Aggregated P/Q | Explicit device models |
| BESS Model | PQ-controlled source | Full inverter dynamics |
| Protection | Not modeled | Full coordination study |

---

## 🏗️ Architecture

### System Topology

```
Grid (External Grid - Slack Bus)
│
└── HV/MV Transformer (138kV/13.8kV, 80 MVA)
    │
    └── Data Center Main Bus (13.8 kV MV)
        │
        ├── Aggregated Data Center Load (P + Q)
        │   (Represents ALL: IT, cooling, lighting via P/Q envelope)
        │
        └── BESS Inverter (PQ-controlled)
            (Charge/discharge with optional reactive support)
```

### What We Model

- **Grid Connection**: External grid as voltage source (slack bus)
- **Transformer**: Step-down transformer with impedance
- **Load**: Aggregated P/Q load representing entire data center
- **BESS**: Static generator with controllable P/Q dispatch

### What We DO NOT Model (Intentionally)

These are abstracted into the P/Q envelope:

- Individual GPUs, servers, or racks
- Power supply units (PSUs)
- Uninterruptible power supplies (UPS)
- Rectifiers and DC-DC converters
- Inverter control dynamics
- Protection relays and coordination
- Harmonics and power quality

---

## 📊 Scenarios

### Load Types

| Load Type | Characteristics | Utilization Pattern |
|-----------|-----------------|---------------------|
| **Training** | High sustained load, GPU clusters | 85-98%, slow variation |
| **Inference** | Variable, request-driven | 30-95%, diurnal pattern |
| **Mixed** | Combination of training + inference | Blended profile |

### DC Architectures (via Power Factor)

| Architecture | Power Factor | Relative Efficiency | Description |
|--------------|--------------|---------------------|-------------|
| **AC Traditional** | 0.92 | Baseline | AC distribution with PSU conversion |
| **48V DC** | 0.97 | +3% | 48V DC bus (telco-style) |
| **800V DC** | 0.99 | +6% | 800V DC bus (hyperscale) |

*Note: Architecture differences are captured via power factor and efficiency multipliers. We do NOT model individual power conversion stages.*

### BESS Configurations

| Option | Power (MW) | Energy (MWh) | Duration |
|--------|------------|--------------|----------|
| No BESS | 0 | 0 | - |
| Small | 10 | 40 | 4 hours |
| Large | 25 | 100 | 4 hours |

---

## 🔧 Technical Stack

### Backend (Offline Python)

```
/backend/
├── network.py          # pandapower network definition
├── scenarios.py        # Load and BESS scenario generators
├── run_simulations.py  # Time-series power flow execution
├── export_results.py   # JSON export for static UI
└── main.py             # Main entry point
```

**Dependencies:**
- Python 3.8+
- pandapower >= 2.14.0
- numpy >= 1.24.0
- pandas >= 2.0.0

### Frontend (Static React)

```
/frontend/
├── public/
│   ├── index.html
│   └── data/
│       ├── simulation_results.json  # Precomputed results
│       └── scenarios/               # Individual scenario files
└── src/
    ├── App.jsx         # Main React component
    └── index.css       # Styling
```

**Dependencies:**
- React 18
- Plotly.js
- Plain CSS (no framework)

---

## 🚀 Usage

### Running the Backend

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run full simulation suite
python main.py

# Custom options
python main.py --base-load 75          # 75 MW data center
python main.py --transformer-mva 100   # 100 MVA transformer
python main.py --output ./custom_dir   # Custom output directory
```

### Building the Frontend

```bash
# Install dependencies
cd frontend
npm install

# Development server
npm start

# Production build for GitHub Pages
npm run build
```

### Deploying to GitHub Pages

1. Build the frontend: `npm run build`
2. Copy `build/` contents to `docs/` folder
3. Enable GitHub Pages from `docs/` folder in repository settings

---

## 📈 Output Schema

### Combined Results JSON

```json
{
  "metadata": {
    "version": "1.0.0",
    "name": "Data Center Power Flow Study",
    "generated": "2024-01-15T12:00:00Z",
    "disclaimer": "PLANNING-LEVEL RESULTS ONLY...",
    "scope": {...},
    "limitations": [...],
    "not_suitable_for": [...]
  },
  "scenarios": [
    {
      "name": "training_ac_bess_25mw",
      "load_type": "training",
      "architecture": "ac_traditional",
      "power_factor": 0.92,
      "time": [0.0, 0.25, 0.5, ...],
      "voltage_pu": [0.9685, 0.9682, ...],
      "transformer_loading_pct": [85.2, 86.1, ...],
      "grid_mw": [52.3, 53.1, ...],
      "grid_mvar": [22.4, 22.8, ...],
      "bess_mw": [0.0, -5.2, 8.3, ...],
      "bess_soc": [0.50, 0.52, 0.48, ...],
      "violations": {...},
      "summary": {...}
    }
  ],
  "comparison": {...}
}
```

---

## 📊 UI Features

### Sidebar Controls

- **Load Type Selector**: Filter by training/inference/mixed
- **Architecture Selector**: Filter by AC/48V DC/800V DC
- **BESS Selector**: Filter by with/without BESS
- **Scenario List**: Select specific scenario to view

### Main View Charts

1. **Bus Voltage vs Time**: Voltage at data center main bus (p.u.)
2. **Transformer Loading vs Time**: Loading percentage with 100% limit
3. **Grid Power Exchange**: Active (MW) and reactive (MVAr) from grid
4. **Load Profile**: Data center aggregated load P and Q
5. **BESS Dispatch**: Charge/discharge power over time
6. **BESS State of Charge**: SOC percentage with min/max limits

### Violation Highlighting

- Undervoltage violations (< 0.95 p.u.)
- Overvoltage violations (> 1.05 p.u.)
- Transformer overload (> 100%)

---

## 🔬 Methodology

### Power Flow Algorithm

- **Type**: AC power flow (Newton-Raphson)
- **Model**: Positive-sequence equivalent
- **Assumption**: Balanced three-phase system

### Load Modeling

Loads are modeled as constant P/Q at the data center main bus. The aggregated load represents:

```
Total Load = IT Load × Efficiency_Architecture + Cooling Load

Where:
- IT Load: Based on workload type (training/inference)
- Efficiency_Architecture: Accounts for distribution losses
- Cooling Load: PUE-based fraction of IT load
```

### BESS Modeling

BESS is modeled as a static generator (sgen) with:

- Positive P = discharge (generation)
- Negative P = charge (consumption)
- Q = 0 (unity power factor assumed)

Dispatch strategy: Peak shaving at 75th percentile of load

### Constraint Checking

After each power flow solution, we check:

1. **Voltage**: 0.95 ≤ V ≤ 1.05 p.u.
2. **Transformer Loading**: ≤ 100%

---

## 🛡️ Limitations

### Technical Limitations

1. **Positive-sequence only**: No unbalanced analysis
2. **Steady-state only**: No transient dynamics
3. **No harmonics**: No power quality assessment
4. **Simplified BESS**: No inverter control dynamics
5. **No protection**: No relay coordination study

### Modeling Limitations

1. **Aggregated loads**: Individual equipment not modeled
2. **Constant P/Q**: No voltage-dependent load behavior
3. **No contingencies**: N-1 analysis not included
4. **No reactive optimization**: Fixed power factors

### Use Case Limitations

- **DO NOT** use for operational dispatch
- **DO NOT** use for interconnection studies
- **DO NOT** use for protection settings
- **DO NOT** use for warranty or guarantee claims

---

## 🤝 Contributing

Contributions are welcome for:

- Additional scenario types
- Improved load profiles
- Additional constraint checks
- UI enhancements
- Documentation improvements

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 📚 References

- [pandapower Documentation](https://pandapower.readthedocs.io/)
- [IEEE Std 3002.2 - Recommended Practice for Power System Studies](https://standards.ieee.org/)
- [Data Center Power Distribution Architectures](https://datacenters.lbl.gov/)

---

## ⚡ Quick Start

```bash
# Clone repository
git clone <repo-url>
cd data-center-powerflow

# Run backend simulation
cd backend
pip install -r requirements.txt
python main.py

# View results in browser
cd ../frontend
npm install
npm start
```

Then open http://localhost:3000 to view the planning study results.

---

**Remember**: This is a **PLANNING-LEVEL** tool. All results require professional engineering review before use in any decision-making process.
