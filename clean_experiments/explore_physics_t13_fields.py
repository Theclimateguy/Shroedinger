#!/usr/bin/env python3
"""EXPLORATORY (not frozen) test T13: surface and planetary fields against
the corrected coupling map P_c.

Register with candidates, signs and bars fixed before reading the fields:
docs/EXPLORATION_PHYSICS_2026-09-29.md, section 9.
Inputs: data/xp_physics (ERA5 monthly fields, downloaded 2026-09-30),
data/b25hydro (monthly wind 850/500), results/explore_physics (iso tiles).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import explore_physics_analysis as A  # noqa: E402
from experiment_B20_armB_global_map import loso_r2  # noqa: E402
from experiment_B25_residual_candidates import tile_mean  # noqa: E402

XPD = Path("data/xp_physics")
G0, RD, CP, OMEGA = 9.80665, 287.0, 1004.0, 7.292e-5
Z850 = 1457.0


def clim(path, name):
    ds = xr.open_dataset(path)
    a = ds[name].mean("valid_time").squeeze().values
    la, lo = ds["latitude"].values, ds["longitude"].values
    ds.close()
    return a, la, lo


def fields() -> dict:
    sub = XPD / "_x_era5_monthly_surface_pbl"
    hr = XPD / "_x_era5_2023_by_hour_blh_sshf"
    out = {}
    # S1: afternoon-maximum BLH (max over the 8 hour-of-day composites,
    # averaged over the 12 months) minus height of 850 hPa above ground
    ds = xr.open_dataset(hr / "data_stream-mnth_stepType-avgua.nc")
    la, lo = ds["latitude"].values, ds["longitude"].values
    blh = ds["blh"].values.reshape(12, 8, len(la), len(lo))
    ds.close()
    blh_max = blh.max(axis=1).mean(axis=0)
    inv = xr.open_dataset("data/b4cov/era5_invariants_global.nc")
    orog = (inv["z"].squeeze().values / G0)[::2, ::2]
    inv.close()
    out["S1_immersion"] = blh_max - np.maximum(Z850 - np.maximum(orog, 0.0), 0.0)
    out["S2_blh_mean"] = clim(sub / "data_stream-moda_stepType-avgua.nc", "blh")[0]
    f_ad = sub / "data_stream-moda_stepType-avgad.nc"
    out["S3_abs_sshf"] = np.abs(clim(f_ad, "avg_ishf")[0])
    st = xr.open_dataset(XPD / "era5_static_subgrid_orography.nc")
    out["S4_sdor"] = st["sdor"].squeeze().values
    out["S5_slor"] = st["slor"].squeeze().values
    st.close()
    out["S6_log_roughness"] = np.log10(np.maximum(
        clim(sub / "data_stream-moda_stepType-avgua.nc", "fsr")[0], 1e-6))
    out["S7_turb_stress"] = np.hypot(clim(f_ad, "avg_iews")[0], clim(f_ad, "avg_inss")[0])
    out["S8_gw_stress"] = np.hypot(clim(f_ad, "avg_iegwss")[0], clim(f_ad, "avg_ingwss")[0])
    out["S9_bl_dissipation"] = clim(f_ad, "avg_ibld")[0]
    out["S10_gw_dissipation"] = clim(f_ad, "avg_igwd")[0]

    dt = xr.open_dataset(XPD / "era5_monthly_temperature_5lev.nc")
    T = {int(p): dt["t"].sel(pressure_level=p).mean("valid_time").values
         for p in dt["pressure_level"].values}
    dt.close()
    th = {p: T[p] * (1000.0 / p) ** (RD / CP) for p in T}
    out["P1_LTS"] = th[700] - th[1000]
    dz = RD * 0.5 * (T[850] + T[500]) / G0 * np.log(850.0 / 500.0)
    n2 = G0 / (0.5 * (th[850] + th[500])) * (th[500] - th[850]) / dz
    N = np.sqrt(np.maximum(n2, 1e-6))
    out["P2_N_850_500"] = N
    dw = xr.open_dataset("data/b25hydro/era5_monthly_wind_850_500.nc")
    lev = dw["pressure_level"].values
    i8, i5 = int(np.argmin(np.abs(lev - 850))), int(np.argmin(np.abs(lev - 500)))
    shear = np.hypot(dw["u"].isel(pressure_level=i8) - dw["u"].isel(pressure_level=i5),
                     dw["v"].isel(pressure_level=i8) - dw["v"].isel(pressure_level=i5)
                     ).mean("valid_time").values
    dw.close()
    f = 2 * OMEGA * np.abs(np.sin(np.deg2rad(la)))[:, None]
    out["P3_eady"] = 0.31 * f * shear / (N * dz)
    out["P4_scale_over_LR"] = f * 200e3 / (N * 4000.0)
    dy = 6371e3 * np.deg2rad(0.5)
    dx = dy * np.maximum(np.cos(np.deg2rad(la)), 0.05)[:, None]

    def grad(a):
        gy, gx = np.gradient(a)
        return np.hypot(gy / dy, gx / dx) * 1e5      # K per 100 km
    out["P5_gradT850"] = grad(T[850])
    out["P6_gradT1000"] = grad(T[1000])
    return {k: (v, la, lo) for k, v in out.items()}


def main() -> None:
    fz = A.Frozen()
    iso = A.corrected(fz)
    Pc = iso["Pc"]
    fld = fields()
    C = {k: np.array([tile_mean(v, la, lo, t) for t in fz.tiles])
         for k, (v, la, lo) in fld.items()}
    res = {"n": fz.n, "base_loso_r2_8cov": float(loso_r2(fz.X_cov, Pc, fz.sector)),
           "single": {}}
    for k, c in C.items():
        rho, p = fz.spearman_rot(c, Pc, n_rot=499)
        res["single"][k] = {
            "rho": round(rho, 3), "p_rot": p,
            "rho_nonzonal": round(float(spearmanr(A.nonzonal(fz, c), A.nonzonal(fz, Pc)).statistic), 3),
            "rho_land": round(float(spearmanr(c[fz.is_land], Pc[fz.is_land]).statistic), 3),
            "rho_ocean": round(float(spearmanr(c[fz.is_ocean], Pc[fz.is_ocean]).statistic), 3),
            "rho_land_frac": round(float(spearmanr(c, fz.land).statistic), 3),
            "gain_over_8cov": fz.loso_gain(c, Pc, n_rot=199)}
    blocks = {"S": [k for k in C if k.startswith("S")],
              "Pi": [k for k in C if k.startswith("P")]}
    res["blocks"] = {}
    for b, ks in blocks.items():
        X = np.column_stack([C[k] for k in ks])
        res["blocks"][b] = {
            "over_8cov": fz.loso_gain(X, Pc, n_rot=499),
            "alone_loso_r2": round(float(loso_r2(np.column_stack(
                [A.z(C[k]) for k in ks]), Pc, fz.sector)), 4)}
    Xs = np.column_stack([A.z(C[k]) for k in blocks["S"]])
    Xp = np.column_stack([A.z(C[k]) for k in blocks["Pi"]])
    res["blocks"]["S_over_8cov_plus_Pi"] = fz.loso_gain(
        Xs, Pc, base=np.column_stack([fz.X_cov, Xp]), n_rot=199)
    res["blocks"]["Pi_over_8cov_plus_S"] = fz.loso_gain(
        Xp, Pc, base=np.column_stack([fz.X_cov, Xs]), n_rot=199)
    res["blocks"]["all_loso_r2"] = round(float(loso_r2(
        np.column_stack([fz.X_cov, Xs, Xp]), Pc, fz.sector)), 4)
    # does the ocean-land contrast survive the surface block?
    lo_, oc_ = fz.is_land, fz.is_ocean

    def contrast_after(X):
        r = Pc - A.ols_predict(X, Pc, X)
        return float(r[oc_].mean() - r[lo_].mean())
    noland = np.column_stack([fz.Xz[n] for n in A.ALL_COV if n != "land_frac"])
    res["ocean_minus_land"] = {
        "raw": float(Pc[oc_].mean() - Pc[lo_].mean()),
        "after_7cov_no_land": contrast_after(noland),
        "after_S_block_only": contrast_after(Xs),
        "after_Pi_block_only": contrast_after(Xp),
        "after_7cov_plus_S": contrast_after(np.column_stack([noland, Xs]))}
    (A.XP / "analysis_t13.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(A.XP / "t13_tile_fields.npz", tid=np.array(fz.keep), **C)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
