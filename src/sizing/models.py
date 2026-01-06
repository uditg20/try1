from __future__ import annotations

from typing import Literal, Optional, Tuple, Dict, Any, List

from pydantic import BaseModel, Field, PositiveFloat, confloat, model_validator


class GridSpec(BaseModel):
    max_import_mw: PositiveFloat = Field(..., description="Maximum grid import at POI (MW).")
    max_export_mw: confloat(ge=0) = Field(0.0, description="Maximum grid export at POI (MW).")
    poi_voltage_kv: PositiveFloat = Field(..., description="POI voltage (kV).")
    poi_min_power_factor: confloat(gt=0, le=1) = Field(
        0.95, description="Minimum operating PF requirement at POI."
    )
    poi_mva_rating: Optional[PositiveFloat] = Field(
        None,
        description="Optional POI MVA rating. If omitted, the tool will size it.",
    )


class TransformerSpec(BaseModel):
    primary_kv: PositiveFloat = Field(138.0, description="Transformer primary kV (typically POI kV).")
    secondary_kv: PositiveFloat = Field(13.8, description="Transformer secondary kV.")
    n_transformers: int = Field(2, ge=1, description="Number of transformers in the bank.")
    redundancy: Literal["N", "N+1"] = Field("N+1", description="Redundancy assumption.")
    per_unit_mva: Optional[PositiveFloat] = Field(
        None, description="Per-transformer MVA rating. If omitted, the tool will size it."
    )
    design_loading_pu: confloat(gt=0, le=1.5) = Field(
        0.9, description="Design target loading (pu) for normal operation."
    )


class LoadSpec(BaseModel):
    load_mw: PositiveFloat = Field(..., description="Data center load (MW). Assumed flat unless profile provided.")
    power_factor: confloat(gt=0, le=1) = Field(0.98, description="Site load PF used for MVA screening.")


class ProfileSpec(BaseModel):
    horizon_hours: int = Field(8760, ge=24, description="Simulation horizon for CFE screening (hours).")
    timestep_hours: int = Field(1, ge=1, description="Timestep resolution (hours).")
    pv_capacity_factor: confloat(gt=0, lt=1) = Field(0.25, description="Annual PV capacity factor (fraction).")
    wind_capacity_factor: confloat(gt=0, lt=1) = Field(0.40, description="Annual wind capacity factor (fraction).")
    seed: int = Field(42, description="Random seed (wind variability).")


class RenewableMixSpec(BaseModel):
    pv_energy_fraction: confloat(ge=0, le=1) = Field(0.5, description="Fraction of renewable energy from PV.")
    wind_energy_fraction: confloat(ge=0, le=1) = Field(0.5, description="Fraction of renewable energy from wind.")

    @model_validator(mode="after")
    def _sum_to_one(self) -> "RenewableMixSpec":
        s = float(self.pv_energy_fraction) + float(self.wind_energy_fraction)
        if abs(s - 1.0) > 1e-6:
            raise ValueError("pv_energy_fraction + wind_energy_fraction must equal 1.0")
        return self


class BessSpec(BaseModel):
    allow_grid_charging: bool = Field(
        False,
        description="If False, BESS only charges from excess renewables (keeps CFE accounting clean).",
    )
    inverter_min_pf: confloat(gt=0, le=1) = Field(
        0.95, description="Minimum inverter PF used to translate MW to MVA."
    )
    roundtrip_efficiency: confloat(gt=0, le=1) = Field(0.88, description="Round-trip efficiency.")
    soc_min: confloat(ge=0, le=1) = Field(0.10, description="Minimum SOC fraction.")
    soc_max: confloat(ge=0, le=1) = Field(0.90, description="Maximum SOC fraction.")
    initial_soc: confloat(ge=0, le=1) = Field(0.50, description="Initial SOC fraction.")
    reserve_soc: confloat(ge=0, le=1) = Field(
        0.0, description="Optional SOC reserve (fraction) held out of normal dispatch."
    )
    # Search bounds
    duration_hours_bounds: Tuple[float, float] = Field(
        (0.0, 12.0), description="Search range for BESS duration (hours)."
    )
    power_fraction_bounds: Tuple[float, float] = Field(
        (0.25, 2.0), description="Search range for BESS power as fraction of load."
    )


class SizingInputs(BaseModel):
    name: str = Field("DataCenterConfiguration", description="Configuration name.")

    # Required user inputs (core)
    cfe_target: confloat(ge=0, le=1) = Field(..., description="Target CFE fraction (0-1).")
    load: LoadSpec
    grid: GridSpec

    # Optional / advanced
    transformer: TransformerSpec = Field(default_factory=TransformerSpec)
    profiles: ProfileSpec = Field(default_factory=ProfileSpec)
    renewable_mix: RenewableMixSpec = Field(default_factory=RenewableMixSpec)
    bess: BessSpec = Field(default_factory=BessSpec)

    # Guardrails
    allow_unserved_load: bool = Field(
        False, description="If False, sizing must meet load within grid limit (else report infeasible)."
    )
    design_margin_pu: confloat(gt=0, le=1.5) = Field(
        1.10, description="Sizing margin applied to MVA/MW ratings."
    )


class ElectricalReport(BaseModel):
    feasible: bool
    warnings: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)

    # Screening metrics
    max_grid_import_mw: float
    grid_import_limit_mw: float
    max_grid_export_mw: float
    grid_export_limit_mw: float
    required_poi_mva: float
    required_transformer_mva_total: float
    firm_transformer_mva: float

    achieved_cfe: float
    annual_load_mwh: float
    annual_grid_import_mwh: float
    annual_renewable_used_mwh: float
    annual_renewable_curtailed_mwh: float
    annual_bess_throughput_mwh: float


class SizingOutputs(BaseModel):
    inputs: SizingInputs

    # Recommended asset sizing
    pv_ac_mw: float
    wind_mw: float
    bess_power_mw: float
    bess_energy_mwh: float
    bess_pcs_mva: float

    # Grid / topology
    poi_mva_rating: float
    transformers: List[Dict[str, Any]]

    # Derived case JSON (compatible with this repo's conventions; renewables live under resources.renewables)
    case_json: Dict[str, Any]
    report: ElectricalReport

