#!/usr/bin/env python3
"""Phase-8: cross-reanalysis replication of the transfer-asymmetry invariant.

Protocol: docs/PROTOCOL_PHASE8_CROSS_REANALYSIS.md (frozen 2026-08-13).
MERRA-2 grid (0.5 x 0.625 deg) resolves nothing below ~200 km, so the ladder
starts at 200 km (resolved subset, matching the ERA5 resolved-scales
controls): curvature bands [200-400, 400-800, 800-1600, 1600-3200] km,
envelope ladder [200, 400, 800, 1600] km.

H8a: within/between clustering of MERRA-2 Fnorm profiles, p < 0.05.
H8b: Spearman across 12 regions between region-median log fine resolved
curvature (bands 200-800 km) in ERA5 (Phase-3 E2, same windows) and MERRA-2:
rho > 0, p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import clean_experiments.experiment_B2_scale_irreversibility as b2
    import clean_experiments.experiment_B4_curvature_invariant as b4
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
    )
    from clean_experiments.experiment_B2b_heldout_invariants import (
        _perm_test,
        _zscore_cols,
    )
except ImportError:
    import experiment_B2_scale_irreversibility as b2  # type: ignore
    import experiment_B4_curvature_invariant as b4  # type: ignore
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2b_heldout_invariants import _perm_test, _zscore_cols  # type: ignore

SEED = 20260811
MERRA_ENV_LADDER = [200.0, 400.0, 800.0, 1600.0]
MERRA_CURV_EDGES = [200.0, 400.0, 800.0, 1600.0, 3200.0]
OUT = _HERE / "results" / "experiment_B8_cross_reanalysis"
B3_DIR = _HERE / "results" / "experiment_B3_scattering_benchmark"


def load_merra(path: Path):
    ds = xr.open_dataset(path)
    tdim = next(d for d in ds["u"].dims if "time" in d)
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    u = np.nan_to_num(np.asarray(ds["u"].transpose(tdim, "lat", "lon").values, dtype=float))
    v = np.nan_to_num(np.asarray(ds["v"].transpose(tdim, "lat", "lon").values, dtype=float))
    ds.close()
    return lat, lon, u, v


def process(path: Path) -> dict:
    tag = path.stem.replace("merra2_wind850_", "")
    cached = OUT / f"{tag}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    print(f"[{tag}] computing", flush=True)
    lat, lon, u, v = load_merra(path)

    # Patch module ladders for the MERRA-2 resolved subset.
    b2_ells_orig, b2_nsteps_orig = b2.ELLS_KM, b2.N_STEPS
    b4_edges_orig = None
    try:
        b2.ELLS_KM = MERRA_ENV_LADDER
        b2.N_STEPS = len(MERRA_ENV_LADDER) - 1
        import clean_experiments.experiment_B2_scale_irreversibility as _b2mod  # noqa: F401

        dx_km, dy_km = _grid_geometry(lat, lon)
        mask = _interior_mask(u.shape[1], u.shape[2], MERRA_ENV_LADDER[-1], dx_km, dy_km)
        omega = b2.compute_vorticity(u, v, lat, lon)
        rho = b2.envelope_rho_profile(omega, dx_km, dy_km, mask)
        p_prof = [float(x) for x in np.median(rho, axis=0)]

        import clean_experiments.experiment_B4_curvature_invariant as _b4mod
        b4_edges_orig = b4.SCALE_EDGES_KM
        b4.SCALE_EDGES_KM = MERRA_CURV_EDGES
        curv = b4.curvature_profiles(u, v, lat, lon)
    finally:
        b2.ELLS_KM, b2.N_STEPS = b2_ells_orig, b2_nsteps_orig
        if b4_edges_orig is not None:
            b4.SCALE_EDGES_KM = b4_edges_orig

    res = {
        "tag": tag, "region": tag.split("__")[0], "window": tag.split("__")[1],
        "P_merra": p_prof,
        "Fnorm_merra": curv["Fnorm_profile"],
        "n_time": int(u.shape[0]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b8merra2"))
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("merra2_wind850_*.nc"))
    print(f"{len(files)} MERRA-2 region-windows", flush=True)
    results = [process(f) for f in files]
    if len(results) < 12:
        raise SystemExit("not enough region-windows yet")

    labels = np.asarray([r["region"] for r in results])
    fn = np.log(np.maximum(np.asarray([r["Fnorm_merra"] for r in results]), 1e-12))
    p_mat = np.asarray([r["P_merra"] for r in results])

    rng = np.random.default_rng(SEED)
    h8a = _perm_test(_zscore_cols(fn), labels, rng)
    p_sig = _perm_test(_zscore_cols(p_mat), labels, rng)

    # H8b: cross-reanalysis consistency of fine resolved curvature (200-800 km)
    merra_fine = {}
    for reg in sorted(set(labels)):
        m = labels == reg
        merra_fine[reg] = float(np.median(np.mean(fn[m][:, :2], axis=1)))
    era_fine = {}
    for f in sorted(B3_DIR.glob("*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "E2_curvature" in r:
            era_fine.setdefault(r["region"], []).append(
                float(np.log(max(np.mean(r["E2_curvature"]["Fnorm_profile"][2:4]), 1e-12))))
    regions = sorted(set(merra_fine) & set(era_fine))
    x = [np.median(era_fine[r]) for r in regions]
    y = [merra_fine[r] for r in regions]
    rs = spearmanr(x, y)
    h8b = {"spearman": float(rs.statistic), "p": float(rs.pvalue),
           "pass": bool(rs.statistic > 0 and rs.pvalue < 0.05), "n_regions": len(regions)}

    verdict = "REPLICATED" if (h8a["pass"] and h8b["pass"]) else "NOT_REPLICATED"
    summary = {
        "n_region_windows": len(results),
        "H8a_signature_merra2": h8a,
        "P_signature_merra2_descriptive": p_sig,
        "H8b_cross_reanalysis": h8b,
        "PHASE8_VERDICT": verdict,
        "era_fine_by_region": {r: float(np.median(era_fine[r])) for r in regions},
        "merra_fine_by_region": merra_fine,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
