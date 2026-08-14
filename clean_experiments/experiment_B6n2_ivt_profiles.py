#!/usr/bin/env python3
"""Phase-6 N2 (exploratory, declared): envelope profiles on moisture transport.

Profiles computed on the curl of vertically-integrated water-vapour flux
(IVT), same octave machinery as the wind-vorticity profile. Question: does
the moisture-transport normalization reduce the regional signature
(collapse toward a universal curve) relative to the wind profile on the
same region-windows?
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        compute_vorticity,
        envelope_rho_profile,
    )
except ImportError:
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        compute_vorticity,
        envelope_rho_profile,
    )

OUT = _HERE / "results" / "experiment_B6_source_sink"
B3_DIR = _HERE / "results" / "experiment_B3_scattering_benchmark"


def load_ivt(path: Path):
    ds = xr.open_dataset(path)
    names = list(ds.data_vars)
    ue = next((n for n in names if "east" in n.lower() or n in ("viwve", "p71.162")), names[0])
    vn = next((n for n in names if "north" in n.lower() or n in ("viwvn", "p72.162")), names[-1])
    da_u = ds[ue].squeeze()
    tdim = next(d for d in da_u.dims if "time" in d or d == "valid_time")
    lat = np.asarray(da_u["latitude"].values, dtype=float)
    lon = np.asarray(da_u["longitude"].values, dtype=float)
    u = np.nan_to_num(np.asarray(da_u.transpose(tdim, ...).values, dtype=float))
    v = np.nan_to_num(np.asarray(ds[vn].squeeze().transpose(tdim, ...).values, dtype=float))
    ds.close()
    return lat, lon, u, v


def clustering_diff(profiles: np.ndarray, labels: np.ndarray) -> float:
    z = (profiles - profiles.mean(0)) / np.where(profiles.std(0) < 1e-12, 1, profiles.std(0))
    w, b = [], []
    for i, j in itertools.combinations(range(len(labels)), 2):
        d = float(np.linalg.norm(z[i] - z[j]))
        (w if labels[i] == labels[j] else b).append(d)
    return float(np.median(b) - np.median(w))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ivt-dir", type=Path, default=Path("data/b6ivt"))
    args = parser.parse_args()

    rows = []
    for f in sorted(args.ivt_dir.glob("era5_ivt_*.nc")):
        tag = f.stem.replace("era5_ivt_", "")
        cached = OUT / f"b6n2_{tag}.json"
        if cached.exists():
            rows.append(json.loads(cached.read_text(encoding="utf-8")))
            continue
        print(f"[{tag}] computing", flush=True)
        lat, lon, iu, iv = load_ivt(f)
        # IVT in kg m-1 s-1; scale to wind-like magnitude for the shared
        # machinery (units cancel in rank correlations).
        scale = max(float(np.std(iu)), 1e-9) / 10.0
        omega = compute_vorticity(iu / scale, iv / scale, lat, lon)
        dx, dy = _grid_geometry(lat, lon)
        mask = _interior_mask(omega.shape[1], omega.shape[2], 1600.0, dx, dy)
        rho = envelope_rho_profile(omega, dx, dy, mask)
        r = {"tag": tag, "region": tag.split("__")[0],
             "P_ivt": [float(x) for x in np.median(rho, axis=0)]}
        OUT.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(r, indent=2), encoding="utf-8")
        rows.append(r)

    labels = np.asarray([r["region"] for r in rows])
    p_ivt = np.asarray([r["P_ivt"] for r in rows])
    diff_ivt = clustering_diff(p_ivt, labels)

    tags = {r["tag"] for r in rows}
    wind_rows = []
    for f in sorted(B3_DIR.glob("*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("tag") in tags and "P_real" in r:
            wind_rows.append((r["region"], r["P_real"]))
    diff_wind = np.nan
    if wind_rows:
        diff_wind = clustering_diff(np.asarray([p for _, p in wind_rows]),
                                    np.asarray([l for l, _ in wind_rows]))

    out = {
        "n": len(rows),
        "clustering_diff_ivt_profiles": diff_ivt,
        "clustering_diff_wind_profiles_same_windows": float(diff_wind),
        "mean_P_ivt": [float(x) for x in p_ivt.mean(0)],
    }
    (OUT / "b6n2_ivt_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
