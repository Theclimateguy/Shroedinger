#!/usr/bin/env python3
"""Phase-6B: driver of rho_5 (synoptic decoupling).

Protocol: docs/PROTOCOL_PHASE6_SOURCE_SINK_DYNAMICS.md (frozen 2026-08-12).
Primary: rho_5 ~ [abs center latitude, bulk shear |V850-V500|],
LOO R^2 > 0 with permutation p < 0.05.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import clean_experiments.download_b1_era5_wind as dl_b1
    from clean_experiments.download_b2_era5_wind import REGIONS_B2
    from clean_experiments.download_b2b_era5_wind import REGIONS_B2B
    from clean_experiments.download_b5_wind500 import COMBOS
    from clean_experiments.experiment_B4_curvature_invariant import region_covariates
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    import download_b1_era5_wind as dl_b1  # type: ignore
    from download_b2_era5_wind import REGIONS_B2  # type: ignore
    from download_b2b_era5_wind import REGIONS_B2B  # type: ignore
    from download_b5_wind500 import COMBOS  # type: ignore
    from experiment_B4_curvature_invariant import region_covariates  # type: ignore
    from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # type: ignore

SEED = 20260811
REGIONS_ALL = {**dl_b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}
OUT = _HERE / "results" / "experiment_B6_source_sink"

RESULT_DIRS = [
    _HERE / "results" / "experiment_B4_curvature_invariant",
    _HERE / "results" / "experiment_B3_scattering_benchmark",
    _HERE / "results" / "experiment_B2b_heldout_invariants",
]


def load_wind(path: Path, prefix: str):
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None)
    return u, v


def main() -> None:
    # rho_5 medians per region, pooled over all result dirs with P_real
    rho5: dict[str, list[float]] = {}
    for d in RESULT_DIRS:
        for f in sorted(d.glob("*__*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if "P_real" in r:
                rho5.setdefault(r["region"], []).append(float(r["P_real"][4]))
    regions = sorted(rho5.keys())
    y = np.asarray([np.median(rho5[r]) for r in regions])

    # bulk shear per region from on-disk wind500/wind850 pairs
    shear = {}
    for region, window in COMBOS:
        w500 = Path("data/b5w500") / f"era5_wind500_{region}__{window}.nc"
        w850_dir = "data/b1" if window.startswith(("W1", "W2", "W3", "W4")) else "data/b2b"
        w850 = Path(w850_dir) / f"era5_wind850_{region}__{window}.nc"
        if not (w500.exists() and w850.exists()):
            continue
        u5, v5 = load_wind(w500, "wind500")
        u8, v8 = load_wind(w850, "wind850")
        nt = min(u5.shape[0], u8.shape[0])
        s = np.sqrt((u5[:nt] - u8[:nt]) ** 2 + (v5[:nt] - v8[:nt]) ** 2)
        shear.setdefault(region, []).append(float(np.nanmedian(s)))
        print(f"shear {region} {window}: {shear[region][-1]:.2f} m/s", flush=True)

    cov = region_covariates(Path("data/b4cov"))
    abs_lat = {r: abs((REGIONS_ALL[r][0] + REGIONS_ALL[r][2]) / 2) for r in regions}
    sh = {r: float(np.median(shear.get(r, [np.nan]))) for r in regions}

    x_primary = np.asarray([[abs_lat[r], sh[r]] for r in regions])
    xz = (x_primary - x_primary.mean(0)) / x_primary.std(0)

    def loo_r2(xm, yv):
        pred = np.zeros(len(yv))
        for k in range(len(yv)):
            tr = np.arange(len(yv)) != k
            a = np.hstack([np.ones((tr.sum(), 1)), xm[tr]])
            c, *_ = np.linalg.lstsq(a, yv[tr], rcond=None)
            pred[k] = np.hstack([[1.0], xm[k]]) @ c
        return 1.0 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)

    r2 = loo_r2(xz, y)
    rng = np.random.default_rng(SEED)
    perm = [loo_r2(xz, y[rng.permutation(len(y))]) for _ in range(999)]
    p = float((1 + sum(q >= r2 for q in perm)) / 1000)
    c6b = bool(r2 > 0 and p < 0.05)

    ext = {
        "abs_lat": [abs_lat[r] for r in regions],
        "shear": [sh[r] for r in regions],
        "cape_mean": [cov[r]["cape_mean"] for r in regions],
        "land_frac": [cov[r]["land_frac"] for r in regions],
        "orog_std": [cov[r]["orog_std"] for r in regions],
    }
    singles = {k: {"spearman": float(spearmanr(v, y).statistic),
                   "p": float(spearmanr(v, y).pvalue)} for k, v in ext.items()}

    out = {
        "regions": regions,
        "rho5_median": [float(v) for v in y],
        "primary_loo_r2": r2, "primary_p": p, "C6b_pass": c6b,
        "shear_median": sh, "abs_lat": abs_lat,
        "single_predictor_spearman": singles,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b6b_rho5_driver.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("primary_loo_r2", "primary_p", "C6b_pass",
                       "single_predictor_spearman")}, indent=2))


if __name__ == "__main__":
    main()
