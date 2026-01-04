"""
Microgrid Network Topology
==========================

Assembles POI, transformers, buses, and resources into a
complete electrical topology for the data center.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math

from .poi import PointOfInterconnection
from .transformer import Transformer, TransformerBank
from .bus import Bus, BusType, PCS


@dataclass
class MicrogridTopology:
    """
    Complete microgrid topology model.
    
    Represents the electrical structure from POI to loads:
    POI → Transformers → Main Bus → [Critical Bus, Generator Bus, BESS Bus]
    
    Attributes:
        name: Topology identifier
        poi: Point of interconnection
        transformers: List of transformers
        buses: List of buses
        pcs_systems: Dict of PCS systems by BESS name
    """
    name: str
    poi: PointOfInterconnection
    transformers: List[Transformer] = field(default_factory=list)
    buses: List[Bus] = field(default_factory=list)
    pcs_systems: Dict[str, PCS] = field(default_factory=dict)
    
    def add_transformer(self, transformer: Transformer) -> None:
        """Add a transformer to the topology."""
        self.transformers.append(transformer)
    
    def add_bus(self, bus: Bus) -> None:
        """Add a bus to the topology."""
        self.buses.append(bus)
    
    def add_pcs(self, bess_name: str, pcs: PCS) -> None:
        """Associate a PCS with a BESS."""
        self.pcs_systems[bess_name] = pcs
    
    def get_bus(self, name: str) -> Optional[Bus]:
        """Get a bus by name."""
        for bus in self.buses:
            if bus.name == name:
                return bus
        return None
    
    def get_critical_bus(self) -> Optional[Bus]:
        """Get the critical load bus."""
        for bus in self.buses:
            if bus.bus_type == BusType.CRITICAL:
                return bus
        return None
    
    @property
    def total_transformer_mva(self) -> float:
        """Total transformer capacity."""
        return sum(t.mva_rating for t in self.transformers)
    
    @property
    def firm_transformer_mva(self) -> float:
        """Firm transformer capacity (after N-1)."""
        if not self.transformers:
            return 0.0
        # Assume N+1 redundancy - lose largest unit
        largest = max(t.mva_rating for t in self.transformers)
        return self.total_transformer_mva - largest
    
    def check_electrical_feasibility(
        self,
        grid_p_mw: float,
        grid_q_mvar: float,
        bess_dispatch: Dict[str, Dict[str, float]] = None
    ) -> dict:
        """
        Check if an operating point is electrically feasible.
        
        Validates:
        - POI MW/MVA/PF limits
        - Transformer loading
        - PCS MVA limits
        
        Args:
            grid_p_mw: Power from grid (positive = import)
            grid_q_mvar: Reactive power from grid
            bess_dispatch: Dict of {bess_name: {'p_mw': x, 'q_mvar': y}}
            
        Returns:
            Dict with feasibility status and any violations
        """
        all_violations = []
        
        # 1. Check POI limits
        poi_result = self.poi.check_operating_point(grid_p_mw, grid_q_mvar)
        if not poi_result["feasible"]:
            for v in poi_result["violations"]:
                v["location"] = "POI"
                all_violations.append(v)
        
        # 2. Check transformer loading
        for xfmr in self.transformers:
            # Assume all power flows through each transformer equally
            n_xfmr = len(self.transformers) if self.transformers else 1
            p_per_xfmr = grid_p_mw / n_xfmr
            q_per_xfmr = grid_q_mvar / n_xfmr
            
            xfmr_result = xfmr.get_loading(p_per_xfmr, q_per_xfmr)
            if not xfmr_result["feasible"]:
                all_violations.append({
                    "location": f"Transformer {xfmr.name}",
                    "constraint": "mva_rating",
                    "limit": xfmr.mva_rating,
                    "actual": xfmr_result["s_mva"],
                    "loading_pct": xfmr_result["loading_pct"]
                })
        
        # 3. Check PCS limits
        if bess_dispatch:
            for bess_name, dispatch in bess_dispatch.items():
                if bess_name in self.pcs_systems:
                    pcs = self.pcs_systems[bess_name]
                    p = dispatch.get("p_mw", 0)
                    q = dispatch.get("q_mvar", 0)
                    
                    pcs_result = pcs.check_operating_point(p, q)
                    if not pcs_result["feasible"]:
                        for v in pcs_result["violations"]:
                            v["location"] = f"PCS {pcs.name}"
                            all_violations.append(v)
        
        return {
            "feasible": len(all_violations) == 0,
            "violations": all_violations,
            "poi_status": poi_result,
            "grid_p_mw": grid_p_mw,
            "grid_q_mvar": grid_q_mvar
        }
    
    def get_topology_summary(self) -> dict:
        """
        Generate a summary of the topology for display.
        
        Returns:
            Dict with topology parameters
        """
        return {
            "name": self.name,
            "poi": {
                "max_import_mw": self.poi.max_import_mw,
                "max_export_mw": self.poi.max_export_mw,
                "mva_rating": self.poi.mva_rating,
                "min_pf": self.poi.min_power_factor,
                "voltage_kv": self.poi.voltage_kv
            },
            "transformers": {
                "count": len(self.transformers),
                "total_mva": self.total_transformer_mva,
                "firm_mva": self.firm_transformer_mva,
                "units": [
                    {"name": t.name, "mva": t.mva_rating, "config": t.redundancy_config}
                    for t in self.transformers
                ]
            },
            "buses": {
                "count": len(self.buses),
                "types": [{"name": b.name, "type": b.bus_type.value} for b in self.buses]
            },
            "pcs": {
                "count": len(self.pcs_systems),
                "total_mva": sum(p.mva_rating for p in self.pcs_systems.values()),
                "units": [
                    {"name": p.name, "mva": p.mva_rating, "bess": bess}
                    for bess, p in self.pcs_systems.items()
                ]
            }
        }
    
    def calculate_ride_through_capability(
        self,
        critical_load_mw: float,
        bess_soc_mwh: float,
        bess_power_mw: float,
        gen_start_time_min: float
    ) -> dict:
        """
        Calculate ride-through capability for critical loads.
        
        Determines how long critical load can be served during
        a grid outage, considering:
        - BESS energy and power
        - Generator start time
        
        Args:
            critical_load_mw: Critical load to serve
            bess_soc_mwh: Available BESS energy (above min SOC)
            bess_power_mw: BESS discharge power capability
            gen_start_time_min: Time for generator to come online
            
        Returns:
            Dict with ride-through duration and status
        """
        # BESS must bridge until generator starts
        # Limited by either power or energy
        
        if bess_power_mw < critical_load_mw:
            return {
                "capable": False,
                "reason": "BESS power insufficient for critical load",
                "bess_power_mw": bess_power_mw,
                "critical_load_mw": critical_load_mw,
                "ride_through_min": 0
            }
        
        # Energy limited duration
        energy_duration_min = (bess_soc_mwh / critical_load_mw) * 60
        
        # Can we bridge until gen starts?
        can_bridge = energy_duration_min >= gen_start_time_min
        
        return {
            "capable": can_bridge,
            "ride_through_min": min(energy_duration_min, 9999),
            "gen_start_time_min": gen_start_time_min,
            "margin_min": energy_duration_min - gen_start_time_min,
            "critical_load_mw": critical_load_mw,
            "bess_power_mw": bess_power_mw,
            "bess_energy_mwh": bess_soc_mwh
        }


def create_standard_topology(
    poi_import_mw: float,
    poi_export_mw: float,
    poi_mva: float,
    transformer_mva: float,
    n_transformers: int = 2,
    name: str = "DataCenter"
) -> MicrogridTopology:
    """
    Create a standard data center topology.
    
    Creates:
    - POI with specified limits
    - N transformers in N+1 configuration
    - Main, Critical, Generator, and BESS buses
    
    Args:
        poi_import_mw: Maximum import from grid
        poi_export_mw: Maximum export to grid
        poi_mva: POI apparent power rating
        transformer_mva: Rating per transformer
        n_transformers: Number of transformers
        name: Topology name
        
    Returns:
        Configured MicrogridTopology
    """
    # Create POI
    poi = PointOfInterconnection(
        name=f"{name}_POI",
        max_import_mw=poi_import_mw,
        max_export_mw=poi_export_mw,
        mva_rating=poi_mva,
        min_power_factor=0.95
    )
    
    # Create transformers
    transformers = [
        Transformer(
            name=f"{name}_XFMR_{i+1}",
            mva_rating=transformer_mva,
            primary_kv=138.0,
            secondary_kv=13.8,
            redundancy_config="N+1"
        )
        for i in range(n_transformers)
    ]
    
    # Create buses
    buses = [
        Bus(name=f"{name}_MAIN", bus_type=BusType.MAIN, voltage_kv=13.8),
        Bus(name=f"{name}_CRITICAL", bus_type=BusType.CRITICAL, voltage_kv=13.8),
        Bus(name=f"{name}_GEN", bus_type=BusType.GENERATOR, voltage_kv=13.8),
        Bus(name=f"{name}_BESS", bus_type=BusType.BESS, voltage_kv=13.8),
    ]
    
    return MicrogridTopology(
        name=name,
        poi=poi,
        transformers=transformers,
        buses=buses
    )
