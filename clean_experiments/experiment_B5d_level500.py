#!/usr/bin/env python3
"""Phase-5d: observer invariance — curvature structure at 500 hPa.

Protocol: docs/PROTOCOL_PHASE5_GAUGE_STRUCTURE.md (frozen 2026-08-12).
C5d: (i) within/between clustering of 500 hPa Fnorm profiles diff>0 p<0.05;
(ii) Spearman across 12 regions of region-median fine-band log Fnorm
850 vs 500 > 0, p<0.05.
"""

from __future__ import annotations

import argparse
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
    from clean_experiments.experiment_B2b_heldout_invariants import (
        _perm_test,
        _zscore_cols,
    )
    from clean_experiments.experiment_B4_curvature_invariant import (
        curvature_profiles,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    from experiment_B2b_heldout_invariants import _perm_test, _zscore_cols  # type: ignore
    from experiment_B4_curvature_invariant import curvature_profiles  # type: ignore
    from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # type: ignore

SEED = 20260811
B4_DIR = _HERE / "results" / "experiment_B4_curvature_invariant"
OUT_DIR = _HERE / "results" / "experiment_B5_gauge_structure"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b5w500"))
    args = parser.parse_args()

    results = []
    for f in sorted(args.data_dir.glob("era5_wind500_*.nc")):
        tag = f.stem.replace("era5_wind500_", "")
        cached = OUT_DIR / f"b5d_{tag}.json"
        if cached.exists():
            results.append(json.loads(cached.read_text(encoding="utf-8")))
            continue
        print(f"[{tag}] computing", flush=True)
        _, lat, lon, u, v, _, _ = _load_vector_fields(
            input_path=f, field_set="wind", u_var=None, v_var=None,
            time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
            max_time=None, crop_ny=None, crop_nx=None,
        )
        r = {"tag": tag, "region": tag.split("__")[0],
             **curvature_profiles(u, v, lat, lon)}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(r, indent=2), encoding="utf-8")
        results.append(r)

    labels = np.asarray([r["region"] for r in results])
    fn500 = np.asarray([r["Fnorm_profile"] for r in results])
    rng = np.random.default_rng(SEED)
    sig = _perm_test(_zscore_cols(fn500), labels, rng)

    fine500 = {}
    for reg in sorted(set(labels)):
        m = labels == reg
        fine500[reg] = float(np.median(np.log(
            np.maximum(np.mean(fn500[m][:, :2], axis=1), 1e-12))))
    fine850 = {}
    for f in sorted(B4_DIR.glob("*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        fine850.setdefault(r["region"], []).append(
            float(np.log(max(np.mean(r["Fnorm_profile"][:2]), 1e-12))))
    regions = sorted(fine500.keys())
    x = [np.median(fine850[r]) for r in regions]
    y = [fine500[r] for r in regions]
    rs = spearmanr(x, y)
    cross = {"spearman": float(rs.statistic), "p": float(rs.pvalue)}

    c5d = bool(sig["pass"] and rs.statistic > 0 and rs.pvalue < 0.05)
    out = {
        "n_region_windows": len(results),
        "signature_500": sig,
        "cross_level": cross,
        "fine_median_log_fnorm_500": fine500,
        "C5d_pass": c5d,
    }
    (OUT_DIR / "b5d_level500.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
