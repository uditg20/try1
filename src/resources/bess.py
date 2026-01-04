"""
Battery Energy Storage System (BESS)
====================================

Models grid-scale battery storage with:
- SOC dynamics and limits
- Power/energy constraints
- Degradation modeling
- Reserve requirements
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class BESS:
    """
    Battery Energy Storage System model.
    
    Attributes:
        name: BESS identifier
        power_mw: Maximum charge/discharge power (MW)
        energy_mwh: Total energy capacity (MWh)
        pcs_mva: Power conversion system MVA rating
        efficiency_charge: Charging efficiency (0-1)
        efficiency_discharge: Discharging efficiency (0-1)
        soc_min: Minimum state of charge (fraction)
        soc_max: Maximum state of charge (fraction)
        soc_reserve: SOC reserved for ride-through (fraction)
        initial_soc: Initial state of charge (fraction)
        ramp_rate_mw_per_min: Maximum ramp rate
        degradation_cost_per_mwh: Cycle degradation cost ($/MWh throughput)
        aux_load_mw: Auxiliary load (cooling, BMS, etc.)
    """
    name: str
    power_mw: float
    energy_mwh: float
    pcs_mva: float
    efficiency_charge: float = 0.94
    efficiency_discharge: float = 0.94
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_reserve: float = 0.20  # Reserved for critical load ride-through
    initial_soc: float = 0.50
    ramp_rate_mw_per_min: float = 100.0  # Effectively unlimited for BESS
    degradation_cost_per_mwh: float = 5.0  # $/MWh throughput
    aux_load_mw: float = 0.0
    
    def __post_init__(self):
        """Validate BESS parameters."""
        if self.power_mw <= 0:
            raise ValueError("power_mw must be positive")
        if self.energy_mwh <= 0:
            raise ValueError("energy_mwh must be positive")
        if self.pcs_mva <= 0:
            raise ValueError("pcs_mva must be positive")
        if not (0 < self.efficiency_charge <= 1):
            raise ValueError("efficiency_charge must be between 0 and 1")
        if not (0 < self.efficiency_discharge <= 1):
            raise ValueError("efficiency_discharge must be between 0 and 1")
        if not (0 <= self.soc_min < self.soc_max <= 1):
            raise ValueError("SOC limits must be 0 <= soc_min < soc_max <= 1")
        if not (0 <= self.soc_reserve <= 1):
            raise ValueError("soc_reserve must be between 0 and 1")
    
    @property
    def duration_hours(self) -> float:
        """Energy/power duration in hours."""
        return self.energy_mwh / self.power_mw
    
    @property
    def roundtrip_efficiency(self) -> float:
        """Round-trip efficiency."""
        return self.efficiency_charge * self.efficiency_discharge
    
    @property
    def usable_energy_mwh(self) -> float:
        """Usable energy between SOC limits."""
        return self.energy_mwh * (self.soc_max - self.soc_min)
    
    @property
    def dispatchable_soc_min(self) -> float:
        """
        Minimum SOC for market dispatch.
        
        Market dispatch cannot use the ride-through reserve.
        """
        return max(self.soc_min, self.soc_reserve)
    
    def get_soc_energy(self, soc_fraction: float) -> float:
        """Convert SOC fraction to MWh."""
        return soc_fraction * self.energy_mwh
    
    def get_energy_soc(self, energy_mwh: float) -> float:
        """Convert MWh to SOC fraction."""
        return energy_mwh / self.energy_mwh
    
    def get_available_charge_power(
        self,
        current_soc: float,
        interval_hours: float
    ) -> float:
        """
        Get maximum charge power considering SOC headroom.
        
        Args:
            current_soc: Current state of charge
            interval_hours: Dispatch interval duration
            
        Returns:
            Maximum charge power (MW, positive value)
        """
        # Energy headroom to soc_max
        headroom_mwh = (self.soc_max - current_soc) * self.energy_mwh
        
        # Power limited by headroom (considering efficiency)
        power_from_headroom = headroom_mwh / (interval_hours * self.efficiency_charge)
        
        return min(self.power_mw, power_from_headroom)
    
    def get_available_discharge_power(
        self,
        current_soc: float,
        interval_hours: float,
        respect_reserve: bool = True
    ) -> float:
        """
        Get maximum discharge power considering SOC floor.
        
        Args:
            current_soc: Current state of charge
            interval_hours: Dispatch interval duration
            respect_reserve: If True, cannot discharge below soc_reserve
            
        Returns:
            Maximum discharge power (MW, positive value)
        """
        # SOC floor depends on whether we respect reserve
        soc_floor = self.dispatchable_soc_min if respect_reserve else self.soc_min
        
        # Available energy above floor
        available_mwh = (current_soc - soc_floor) * self.energy_mwh
        
        if available_mwh <= 0:
            return 0.0
        
        # Power limited by available energy
        power_from_energy = available_mwh * self.efficiency_discharge / interval_hours
        
        return min(self.power_mw, power_from_energy)
    
    def calculate_soc_transition(
        self,
        current_soc: float,
        charge_mw: float,
        discharge_mw: float,
        interval_hours: float
    ) -> dict:
        """
        Calculate SOC after charge/discharge.
        
        Args:
            current_soc: Current SOC fraction
            charge_mw: Charging power (positive)
            discharge_mw: Discharging power (positive)
            interval_hours: Duration in hours
            
        Returns:
            Dict with new_soc, energy flows, and validity
        """
        # Net energy flow (positive = charging)
        charge_energy = charge_mw * interval_hours * self.efficiency_charge
        discharge_energy = discharge_mw * interval_hours / self.efficiency_discharge
        
        # SOC change
        delta_soc = (charge_energy - discharge_energy) / self.energy_mwh
        new_soc = current_soc + delta_soc
        
        # Check validity
        valid = self.soc_min <= new_soc <= self.soc_max
        
        return {
            "new_soc": new_soc,
            "delta_soc": delta_soc,
            "charge_energy_mwh": charge_energy,
            "discharge_energy_mwh": discharge_energy,
            "valid": valid,
            "violation": None if valid else (
                "soc_min" if new_soc < self.soc_min else "soc_max"
            )
        }
    
    def get_degradation_cost(
        self,
        discharge_mw: float,
        interval_hours: float
    ) -> float:
        """
        Calculate degradation cost for discharge.
        
        Uses simple throughput-based model.
        
        Args:
            discharge_mw: Discharge power
            interval_hours: Interval duration
            
        Returns:
            Degradation cost ($)
        """
        throughput_mwh = discharge_mw * interval_hours
        return throughput_mwh * self.degradation_cost_per_mwh
    
    def get_reserve_capability(
        self,
        current_soc: float,
        reserve_duration_hours: float,
        direction: str = "up"
    ) -> dict:
        """
        Calculate reserve provision capability.
        
        Args:
            current_soc: Current SOC
            reserve_duration_hours: Required duration to sustain reserve
            direction: 'up' (discharge) or 'down' (charge)
            
        Returns:
            Dict with max_mw and energy_backing
        """
        if direction == "up":
            # Discharge reserve - limited by energy above reserve floor
            available_soc = current_soc - self.dispatchable_soc_min
            available_mwh = available_soc * self.energy_mwh
            max_mw_from_energy = available_mwh / reserve_duration_hours
            max_mw = min(self.power_mw, max_mw_from_energy)
        else:
            # Charge reserve - limited by headroom to soc_max
            headroom_soc = self.soc_max - current_soc
            headroom_mwh = headroom_soc * self.energy_mwh
            max_mw_from_energy = headroom_mwh / reserve_duration_hours
            max_mw = min(self.power_mw, max_mw_from_energy)
        
        return {
            "max_mw": max(0, max_mw),
            "energy_backing_mwh": max(0, max_mw) * reserve_duration_hours,
            "duration_hours": reserve_duration_hours
        }
    
    def get_parameters(self) -> dict:
        """Get BESS parameters as dictionary."""
        return {
            "name": self.name,
            "power_mw": self.power_mw,
            "energy_mwh": self.energy_mwh,
            "pcs_mva": self.pcs_mva,
            "duration_hours": self.duration_hours,
            "efficiency_charge": self.efficiency_charge,
            "efficiency_discharge": self.efficiency_discharge,
            "roundtrip_efficiency": self.roundtrip_efficiency,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "soc_reserve": self.soc_reserve,
            "dispatchable_soc_min": self.dispatchable_soc_min,
            "usable_energy_mwh": self.usable_energy_mwh,
            "degradation_cost_per_mwh": self.degradation_cost_per_mwh
        }
