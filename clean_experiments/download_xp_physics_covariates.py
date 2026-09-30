#!/usr/bin/env python3
"""EXPLORATORY (not frozen) data for the physics probes -> data/xp_physics/.

Register: docs/EXPLORATION_PHYSICS_2026-09-29.md (test T7).
ERA5 via CDS, global 0.5 deg. Resumable: existing targets are skipped.

  static   sub-grid orography statistics                      (~5 MB)
  diurnal  2023 monthly means by hour of day (8 hours):
           boundary-layer height, sensible heat flux          (~100 MB)
  monthly  2021-2024 monthly means of surface / boundary-layer
           / parametrised-drag fields                         (~350 MB)
  plev     2021-2024 monthly temperature on 5 levels          (~120 MB)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cdsapi

YEARS = ["2021", "2022", "2023", "2024"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
GRID = [0.5, 0.5]
HOURS8 = [f"{h:02d}:00" for h in range(0, 24, 3)]

STATIC_VARS = ["standard_deviation_of_orography",
               "standard_deviation_of_filtered_subgrid_orography",
               "slope_of_sub_gridscale_orography",
               "anisotropy_of_sub_gridscale_orography"]
DIURNAL_VARS = ["boundary_layer_height", "mean_surface_sensible_heat_flux"]
MONTHLY_VARS = ["surface_pressure", "boundary_layer_height",
                "convective_inhibition", "low_cloud_cover",
                "mean_surface_sensible_heat_flux",
                "mean_surface_latent_heat_flux",
                "10m_wind_speed", "forecast_surface_roughness",
                "mean_boundary_layer_dissipation",
                "mean_gravity_wave_dissipation",
                "mean_eastward_gravity_wave_surface_stress",
                "mean_northward_gravity_wave_surface_stress",
                "mean_eastward_turbulent_surface_stress",
                "mean_northward_turbulent_surface_stress",
                "k_index", "zero_degree_level"]


def fetch(c, dataset, request, target: Path) -> None:
    if target.exists():
        print(f"skip {target.name}")
        return
    c.retrieve(dataset, request, str(target))
    print(f"done {target.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/xp_physics"))
    ap.add_argument("--parts", nargs="+",
                    default=["static", "diurnal", "monthly", "plev"])
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    if "static" in args.parts:
        fetch(c, "reanalysis-era5-single-levels-monthly-means",
              {"product_type": "monthly_averaged_reanalysis",
               "variable": STATIC_VARS, "year": "2023", "month": "01",
               "time": "00:00", "grid": GRID, "format": "netcdf"},
              out / "era5_static_subgrid_orography.nc")
    if "diurnal" in args.parts:
        fetch(c, "reanalysis-era5-single-levels-monthly-means",
              {"product_type": "monthly_averaged_reanalysis_by_hour_of_day",
               "variable": DIURNAL_VARS, "year": "2023", "month": MONTHS,
               "time": HOURS8, "grid": GRID, "format": "netcdf"},
              out / "era5_2023_by_hour_blh_sshf.nc")
    if "monthly" in args.parts:
        fetch(c, "reanalysis-era5-single-levels-monthly-means",
              {"product_type": "monthly_averaged_reanalysis",
               "variable": MONTHLY_VARS, "year": YEARS, "month": MONTHS,
               "time": "00:00", "grid": GRID, "format": "netcdf"},
              out / "era5_monthly_surface_pbl.nc")
    if "plev" in args.parts:
        fetch(c, "reanalysis-era5-pressure-levels-monthly-means",
              {"product_type": "monthly_averaged_reanalysis",
               "variable": ["temperature"],
               "pressure_level": ["1000", "925", "850", "700", "500"],
               "year": YEARS, "month": MONTHS, "time": "00:00",
               "grid": GRID, "format": "netcdf"},
              out / "era5_monthly_temperature_5lev.nc")
    print("xp_physics complete")


if __name__ == "__main__":
    main()
