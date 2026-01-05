import React, { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';

/**
 * Data Center Power Flow Study - Static UI
 * 
 * This is a PLANNING-LEVEL visualization tool.
 * It loads precomputed power flow results and displays them.
 * There is NO live simulation or calculation in the browser.
 * 
 * Purpose: Educational and explanatory
 * NOT for: Operational decisions or interconnection approval
 */

// Constants for scenario filtering
const LOAD_TYPES = [
  { id: 'all', label: 'All Load Types' },
  { id: 'training', label: 'Training' },
  { id: 'inference', label: 'Inference' },
  { id: 'mixed', label: 'Mixed' }
];

const ARCHITECTURES = [
  { id: 'all', label: 'All Architectures' },
  { id: 'ac_traditional', label: 'AC Traditional' },
  { id: 'dc_48v', label: '48V DC' },
  { id: 'dc_800v', label: '800V DC' }
];

const BESS_OPTIONS = [
  { id: 'all', label: 'All BESS Options' },
  { id: 'no_bess', label: 'No BESS' },
  { id: 'with_bess', label: 'With BESS' }
];

// Chart colors
const COLORS = {
  voltage: '#3182ce',
  transformer: '#805ad5',
  gridMW: '#38a169',
  gridMVAr: '#319795',
  load: '#d69e2e',
  bess: '#e53e3e',
  soc: '#dd6b20'
};

// Plotly layout defaults
const defaultLayout = {
  autosize: true,
  margin: { l: 60, r: 30, t: 40, b: 50 },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'white',
  font: { family: 'Inter, sans-serif', size: 12 },
  xaxis: {
    title: 'Time (hours)',
    gridcolor: '#e2e8f0',
    linecolor: '#cbd5e0'
  },
  yaxis: {
    gridcolor: '#e2e8f0',
    linecolor: '#cbd5e0'
  },
  legend: {
    orientation: 'h',
    yanchor: 'bottom',
    y: 1.02,
    xanchor: 'right',
    x: 1
  }
};

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Filters
  const [selectedLoadType, setSelectedLoadType] = useState('all');
  const [selectedArchitecture, setSelectedArchitecture] = useState('all');
  const [selectedBESS, setSelectedBESS] = useState('all');
  const [selectedScenario, setSelectedScenario] = useState(null);

  // Load data on mount
  useEffect(() => {
    async function loadData() {
      try {
        // Try to load the simulation results
        const response = await fetch('./data/simulation_results.json');
        if (!response.ok) {
          throw new Error('Failed to load simulation results');
        }
        const jsonData = await response.json();
        setData(jsonData);
        
        // Select first scenario by default
        if (jsonData.scenarios && jsonData.scenarios.length > 0) {
          setSelectedScenario(jsonData.scenarios[0].name);
        }
      } catch (err) {
        console.error('Error loading data:', err);
        setError(err.message);
        
        // Load sample data for development
        try {
          const sampleResponse = await fetch('./data/sample_results.json');
          if (sampleResponse.ok) {
            const sampleData = await sampleResponse.json();
            setData(sampleData);
            if (sampleData.scenarios && sampleData.scenarios.length > 0) {
              setSelectedScenario(sampleData.scenarios[0].name);
            }
            setError(null);
          }
        } catch (sampleErr) {
          // Keep original error
        }
      } finally {
        setLoading(false);
      }
    }
    
    loadData();
  }, []);

  // Filter scenarios
  const filteredScenarios = useMemo(() => {
    if (!data || !data.scenarios) return [];
    
    return data.scenarios.filter(scenario => {
      if (selectedLoadType !== 'all' && scenario.load_type !== selectedLoadType) {
        return false;
      }
      if (selectedArchitecture !== 'all' && scenario.architecture !== selectedArchitecture) {
        return false;
      }
      if (selectedBESS !== 'all') {
        const hasBESS = scenario.bess_mw && Math.max(...scenario.bess_mw.map(Math.abs)) > 0.1;
        if (selectedBESS === 'no_bess' && hasBESS) return false;
        if (selectedBESS === 'with_bess' && !hasBESS) return false;
      }
      return true;
    });
  }, [data, selectedLoadType, selectedArchitecture, selectedBESS]);

  // Get current scenario
  const currentScenario = useMemo(() => {
    if (!data || !data.scenarios) return null;
    return data.scenarios.find(s => s.name === selectedScenario);
  }, [data, selectedScenario]);

  // Render loading state
  if (loading) {
    return (
      <div className="app-container">
        <Sidebar
          metadata={null}
          loadTypes={LOAD_TYPES}
          architectures={ARCHITECTURES}
          bessOptions={BESS_OPTIONS}
          selectedLoadType={selectedLoadType}
          selectedArchitecture={selectedArchitecture}
          selectedBESS={selectedBESS}
          onLoadTypeChange={setSelectedLoadType}
          onArchitectureChange={setSelectedArchitecture}
          onBESSChange={setSelectedBESS}
          scenarios={[]}
          selectedScenario={null}
          onScenarioChange={() => {}}
        />
        <main className="main-content">
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading simulation results...</p>
          </div>
        </main>
      </div>
    );
  }

  // Render error state
  if (error && !data) {
    return (
      <div className="app-container">
        <Sidebar
          metadata={null}
          loadTypes={LOAD_TYPES}
          architectures={ARCHITECTURES}
          bessOptions={BESS_OPTIONS}
          selectedLoadType={selectedLoadType}
          selectedArchitecture={selectedArchitecture}
          selectedBESS={selectedBESS}
          onLoadTypeChange={setSelectedLoadType}
          onArchitectureChange={setSelectedArchitecture}
          onBESSChange={setSelectedBESS}
          scenarios={[]}
          selectedScenario={null}
          onScenarioChange={() => {}}
        />
        <main className="main-content">
          <div className="error-container">
            <h2>Error Loading Data</h2>
            <p>{error}</p>
            <p>Please ensure simulation_results.json is in the public/data/ directory.</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Sidebar
        metadata={data?.metadata}
        loadTypes={LOAD_TYPES}
        architectures={ARCHITECTURES}
        bessOptions={BESS_OPTIONS}
        selectedLoadType={selectedLoadType}
        selectedArchitecture={selectedArchitecture}
        selectedBESS={selectedBESS}
        onLoadTypeChange={setSelectedLoadType}
        onArchitectureChange={setSelectedArchitecture}
        onBESSChange={setSelectedBESS}
        scenarios={filteredScenarios}
        selectedScenario={selectedScenario}
        onScenarioChange={setSelectedScenario}
      />
      <main className="main-content">
        <PageHeader metadata={data?.metadata} />
        
        {currentScenario ? (
          <>
            <ScenarioInfo scenario={currentScenario} />
            <SummaryCards scenario={currentScenario} />
            <ChartSection scenario={currentScenario} />
            <ViolationsSection scenario={currentScenario} />
          </>
        ) : (
          <div className="card">
            <p>Select a scenario from the sidebar to view results.</p>
          </div>
        )}
      </main>
    </div>
  );
}

// Sidebar Component
function Sidebar({
  metadata,
  loadTypes,
  architectures,
  bessOptions,
  selectedLoadType,
  selectedArchitecture,
  selectedBESS,
  onLoadTypeChange,
  onArchitectureChange,
  onBESSChange,
  scenarios,
  selectedScenario,
  onScenarioChange
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">⚡ Power Flow Study</h1>
        <p className="sidebar-subtitle">Planning-Level Analysis</p>
      </div>

      <div className="sidebar-section">
        <h2 className="sidebar-section-title">Load Type</h2>
        <div className="selector-group">
          {loadTypes.map(lt => (
            <button
              key={lt.id}
              className={`selector-button ${selectedLoadType === lt.id ? 'active' : ''}`}
              onClick={() => onLoadTypeChange(lt.id)}
            >
              {lt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <h2 className="sidebar-section-title">Architecture</h2>
        <div className="selector-group">
          {architectures.map(arch => (
            <button
              key={arch.id}
              className={`selector-button ${selectedArchitecture === arch.id ? 'active' : ''}`}
              onClick={() => onArchitectureChange(arch.id)}
            >
              {arch.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <h2 className="sidebar-section-title">BESS</h2>
        <div className="selector-group">
          {bessOptions.map(opt => (
            <button
              key={opt.id}
              className={`selector-button ${selectedBESS === opt.id ? 'active' : ''}`}
              onClick={() => onBESSChange(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="sidebar-section">
        <h2 className="sidebar-section-title">Scenarios ({scenarios.length})</h2>
        <div className="selector-group" style={{ maxHeight: '200px', overflowY: 'auto' }}>
          {scenarios.map(scenario => (
            <button
              key={scenario.name}
              className={`selector-button ${selectedScenario === scenario.name ? 'active' : ''}`}
              onClick={() => onScenarioChange(scenario.name)}
              style={{ fontSize: '0.75rem' }}
            >
              {formatScenarioName(scenario.name)}
            </button>
          ))}
        </div>
      </div>

      <div className="disclaimer-box">
        <p className="disclaimer-title">⚠️ Planning Only</p>
        <p>
          This is a planning-level study using positive-sequence, steady-state 
          power flow. Results are NOT suitable for operational decisions.
        </p>
      </div>
    </aside>
  );
}

// Page Header
function PageHeader({ metadata }) {
  return (
    <header className="page-header">
      <h1 className="page-title">
        {metadata?.name || 'Data Center Power Flow Study'}
      </h1>
      <p className="page-description">
        {metadata?.description || 'Planning-level electrical simulation results'}
      </p>
    </header>
  );
}

// Scenario Info
function ScenarioInfo({ scenario }) {
  return (
    <div className="scenario-info">
      <div className="scenario-info-item">
        <span className="scenario-info-label">Load Type</span>
        <span className="scenario-info-value">{formatLabel(scenario.load_type)}</span>
      </div>
      <div className="scenario-info-item">
        <span className="scenario-info-label">Architecture</span>
        <span className="scenario-info-value">{formatLabel(scenario.architecture)}</span>
      </div>
      <div className="scenario-info-item">
        <span className="scenario-info-label">Power Factor</span>
        <span className="scenario-info-value">{scenario.power_factor?.toFixed(2)}</span>
      </div>
      <div className="scenario-info-item">
        <span className="scenario-info-label">Intervals</span>
        <span className="scenario-info-value">{scenario.n_intervals} × {scenario.interval_minutes} min</span>
      </div>
    </div>
  );
}

// Summary Cards
function SummaryCards({ scenario }) {
  const summary = scenario.summary;
  const violations = scenario.violations?.count || 0;
  
  return (
    <div className="summary-grid">
      <div className={`summary-card ${getVoltageStatus(summary?.voltage?.min_pu)}`}>
        <p className="summary-label">Voltage Range</p>
        <p className="summary-value">
          {summary?.voltage?.min_pu?.toFixed(3)} - {summary?.voltage?.max_pu?.toFixed(3)}
          <span className="summary-unit">p.u.</span>
        </p>
      </div>
      
      <div className={`summary-card ${getLoadingStatus(summary?.transformer?.max_loading_pct)}`}>
        <p className="summary-label">Max Transformer Loading</p>
        <p className="summary-value">
          {summary?.transformer?.max_loading_pct?.toFixed(1)}
          <span className="summary-unit">%</span>
        </p>
      </div>
      
      <div className="summary-card">
        <p className="summary-label">Peak Grid Import</p>
        <p className="summary-value">
          {summary?.grid?.max_import_mw?.toFixed(1)}
          <span className="summary-unit">MW</span>
        </p>
      </div>
      
      <div className="summary-card">
        <p className="summary-label">Total Energy</p>
        <p className="summary-value">
          {summary?.load?.total_energy_mwh?.toFixed(0)}
          <span className="summary-unit">MWh</span>
        </p>
      </div>
      
      <div className="summary-card">
        <p className="summary-label">BESS Max Discharge</p>
        <p className="summary-value">
          {summary?.bess?.max_discharge_mw?.toFixed(1) || '0.0'}
          <span className="summary-unit">MW</span>
        </p>
      </div>
      
      <div className={`summary-card ${violations > 0 ? 'error' : 'success'}`}>
        <p className="summary-label">Violations</p>
        <p className="summary-value">
          {violations}
          <span className="summary-unit">intervals</span>
        </p>
      </div>
    </div>
  );
}

// Chart Section
function ChartSection({ scenario }) {
  const time = scenario.time;
  
  return (
    <>
      <div className="chart-row">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Bus Voltage</h3>
            <span className="card-subtitle">Data Center Main Bus</span>
          </div>
          <div className="chart-container">
            <Plot
              data={[
                {
                  x: time,
                  y: scenario.voltage_pu,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Voltage',
                  line: { color: COLORS.voltage, width: 2 }
                },
                {
                  x: [time[0], time[time.length - 1]],
                  y: [0.95, 0.95],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Min Limit',
                  line: { color: COLORS.bess, width: 1, dash: 'dash' }
                },
                {
                  x: [time[0], time[time.length - 1]],
                  y: [1.05, 1.05],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Max Limit',
                  line: { color: COLORS.bess, width: 1, dash: 'dash' }
                }
              ]}
              layout={{
                ...defaultLayout,
                yaxis: { ...defaultLayout.yaxis, title: 'Voltage (p.u.)', range: [0.92, 1.08] }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Transformer Loading</h3>
            <span className="card-subtitle">Main Transformer</span>
          </div>
          <div className="chart-container">
            <Plot
              data={[
                {
                  x: time,
                  y: scenario.transformer_loading_pct,
                  type: 'scatter',
                  mode: 'lines',
                  fill: 'tozeroy',
                  name: 'Loading',
                  line: { color: COLORS.transformer, width: 2 },
                  fillcolor: 'rgba(128, 90, 213, 0.2)'
                },
                {
                  x: [time[0], time[time.length - 1]],
                  y: [100, 100],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Rating',
                  line: { color: COLORS.bess, width: 2, dash: 'dash' }
                }
              ]}
              layout={{
                ...defaultLayout,
                yaxis: { ...defaultLayout.yaxis, title: 'Loading (%)', range: [0, 120] }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>
      </div>

      <div className="chart-row">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Grid Power Exchange</h3>
            <span className="card-subtitle">POI Active & Reactive Power</span>
          </div>
          <div className="chart-container">
            <Plot
              data={[
                {
                  x: time,
                  y: scenario.grid_mw,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'P (MW)',
                  line: { color: COLORS.gridMW, width: 2 }
                },
                {
                  x: time,
                  y: scenario.grid_mvar,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Q (MVAr)',
                  line: { color: COLORS.gridMVAr, width: 2 }
                }
              ]}
              layout={{
                ...defaultLayout,
                yaxis: { ...defaultLayout.yaxis, title: 'Power (MW / MVAr)' }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Load Profile</h3>
            <span className="card-subtitle">Data Center Aggregated Load</span>
          </div>
          <div className="chart-container">
            <Plot
              data={[
                {
                  x: time,
                  y: scenario.load_p_mw,
                  type: 'scatter',
                  mode: 'lines',
                  fill: 'tozeroy',
                  name: 'P (MW)',
                  line: { color: COLORS.load, width: 2 },
                  fillcolor: 'rgba(214, 158, 46, 0.2)'
                },
                {
                  x: time,
                  y: scenario.load_q_mvar,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Q (MVAr)',
                  line: { color: COLORS.soc, width: 2 }
                }
              ]}
              layout={{
                ...defaultLayout,
                yaxis: { ...defaultLayout.yaxis, title: 'Power (MW / MVAr)' }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>
      </div>

      <div className="chart-row">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">BESS Dispatch</h3>
            <span className="card-subtitle">Positive = Discharge, Negative = Charge</span>
          </div>
          <div className="chart-container">
            <Plot
              data={[
                {
                  x: time,
                  y: scenario.bess_mw,
                  type: 'bar',
                  name: 'BESS Power',
                  marker: {
                    color: scenario.bess_mw.map(v => v >= 0 ? COLORS.gridMW : COLORS.voltage)
                  }
                }
              ]}
              layout={{
                ...defaultLayout,
                yaxis: { ...defaultLayout.yaxis, title: 'Power (MW)' },
                bargap: 0.1
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">BESS State of Charge</h3>
            <span className="card-subtitle">SOC Profile</span>
          </div>
          <div className="chart-container">
            <Plot
              data={[
                {
                  x: time,
                  y: scenario.bess_soc.map(s => s * 100),
                  type: 'scatter',
                  mode: 'lines',
                  fill: 'tozeroy',
                  name: 'SOC',
                  line: { color: COLORS.soc, width: 2 },
                  fillcolor: 'rgba(221, 107, 32, 0.2)'
                },
                {
                  x: [time[0], time[time.length - 1]],
                  y: [10, 10],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Min SOC',
                  line: { color: COLORS.bess, width: 1, dash: 'dash' }
                },
                {
                  x: [time[0], time[time.length - 1]],
                  y: [90, 90],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Max SOC',
                  line: { color: COLORS.bess, width: 1, dash: 'dash' }
                }
              ]}
              layout={{
                ...defaultLayout,
                yaxis: { ...defaultLayout.yaxis, title: 'SOC (%)', range: [0, 100] }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        </div>
      </div>
    </>
  );
}

// Violations Section
function ViolationsSection({ scenario }) {
  const violations = scenario.violations;
  
  if (!violations || violations.count === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Constraint Violations</h3>
        </div>
        <p style={{ color: 'var(--color-success)' }}>
          ✓ No constraint violations detected in this scenario.
        </p>
      </div>
    );
  }
  
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Constraint Violations</h3>
        <span className="card-subtitle">{violations.count} violations detected</span>
      </div>
      
      <table className="violations-table">
        <thead>
          <tr>
            <th>Time (h)</th>
            <th>Type</th>
            <th>Location</th>
            <th>Limit</th>
            <th>Actual</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
          {violations.details?.slice(0, 10).map((v, i) => (
            <tr key={i}>
              <td>{v.time_hours?.toFixed(2)}</td>
              <td>{formatLabel(v.type)}</td>
              <td>{v.location}</td>
              <td>{v.limit_pu?.toFixed(3) || v.limit_pct?.toFixed(1)}</td>
              <td>{v.actual_pu?.toFixed(3) || v.actual_pct?.toFixed(1)}</td>
              <td>
                <span className={`violation-badge ${v.severity}`}>
                  {v.severity}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {violations.details?.length > 10 && (
        <p style={{ marginTop: 'var(--spacing-md)', color: 'var(--color-gray-500)' }}>
          Showing first 10 of {violations.details.length} violations.
        </p>
      )}
    </div>
  );
}

// Helper functions
function formatScenarioName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace('Ac ', 'AC ')
    .replace('Dc ', 'DC ')
    .replace('Bess', 'BESS')
    .replace('Mw', 'MW');
}

function formatLabel(label) {
  if (!label) return '';
  return label
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace('Ac ', 'AC ')
    .replace('Dc ', 'DC ');
}

function getVoltageStatus(minVoltage) {
  if (!minVoltage) return '';
  if (minVoltage < 0.95) return 'error';
  if (minVoltage < 0.97) return 'warning';
  return 'success';
}

function getLoadingStatus(maxLoading) {
  if (!maxLoading) return '';
  if (maxLoading > 100) return 'error';
  if (maxLoading > 80) return 'warning';
  return 'success';
}

export default App;
