#!/usr/bin/env python3
"""B5c RETEST with a temporal-coherence-preserving null (audit remediation).

The original 5c null (per-snapshot spatial phase randomization) destroyed
temporal coherence; an independent-band AR(1) null reproduces the original
margin, refuting the 'integrability' reading. This retest uses per-mode
TIME-domain phase randomization of the band coefficient series: each mode's
temporal power spectrum (hence autocorrelation) is preserved exactly, while
cross-mode and cross-band phase alignment is destroyed. If the real path
discrepancy still falls below the 5th percentile of this null (>=3/5 band
pairs, >=32/48 region-windows), path consistency reflects genuine
cross-scale structure; otherwise claim 3 is retracted.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B4_curvature_invariant import (
        confirmation_files,
    )
    from clean_experiments.experiment_B5c_integrability import (
        _coeffs,
        path_discrepancy,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    from experiment_B4_curvature_invariant import confirmation_files  # type: ignore
    from experiment_B5c_integrability import _coeffs, path_discrepancy  # type: ignore
    from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # type: ignore

SEED = 20260811
N_SURR = 99
OUT = _HERE / "results" / "experiment_B5_gauge_structure"

_G: dict[str, object] = {}


def temporal_phase_randomize(coeffs: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for arr in coeffs:
        a = np.asarray(arr, dtype=np.complex128)
        nt = a.shape[0]
        fh = np.fft.fft(a, axis=0)
        phases = np.exp(2j * np.pi * rng.random((nt, a.shape[1])))
        out.append(np.fft.ifft(fh * phases, axis=0))
    return out


def _init(coeffs) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _G["coeffs"] = coeffs


def _worker(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sur = temporal_phase_randomize(_G["coeffs"], rng)  # type: ignore[arg-type]
    return [float(x) for x in path_discrepancy(sur)]


def run_rw(nc_path: Path, workers: int) -> dict:
    tag = nc_path.stem.replace("era5_wind850_", "")
    cached = OUT / f"b5c_retest_{tag}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    print(f"[{tag}] computing", flush=True)
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    coeffs = _coeffs(u, v, lat, lon)
    d_real = path_discrepancy(coeffs)
    seeds = [int(s) for s in np.random.SeedSequence(
        SEED + zlib.crc32(("retest" + tag).encode()) % 100000).generate_state(N_SURR)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, _init, (coeffs,)) as pool:
        sur = np.asarray(pool.map(_worker, seeds))
    p05 = np.percentile(sur, 5, axis=0)
    pairs_below = int(np.sum(d_real < p05))
    res = {
        "tag": tag, "region": tag.split("__")[0],
        "d_real": [float(x) for x in d_real],
        "d_null_p05": [float(x) for x in p05],
        "d_null_median": [float(x) for x in np.median(sur, axis=0)],
        "pairs_below_p05": pairs_below,
        "pass_rw": bool(pairs_below >= 3),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{tag}] pairs_below={pairs_below}/5", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_HERE.parent)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    files = confirmation_files(args.repo_root)
    print(f"{len(files)} region-windows", flush=True)
    results = [run_rw(f, args.workers) for f in files]
    n_pass = sum(1 for r in results if r["pass_rw"])
    summary = {
        "null": "per-mode temporal phase randomization (spectrum-preserving)",
        "n_region_windows": len(results),
        "rw_pass_count": n_pass,
        "C5c_retest_pass": bool(n_pass >= 32),
        "median_d_real": [float(x) for x in np.median(
            np.asarray([r["d_real"] for r in results]), axis=0)],
        "median_d_null": [float(x) for x in np.median(
            np.asarray([r["d_null_median"] for r in results]), axis=0)],
    }
    (OUT / "b5c_retest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
