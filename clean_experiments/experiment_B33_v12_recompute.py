#!/usr/bin/env python3
"""Phase 32, part 2: full recomputation of article-1 quantities under definition V12.

Protocol: docs/PROTOCOL_PHASE32_REVIEW_CHECKS.md, section "Part 2" (2026-10-08).

Definition V12 (replaces the Phase-2 pipeline everywhere a profile P is computed):
  * relative vorticity on the sphere: omega = dv/dx - du/dy + u tan(phi)/a
  * grid points where the ERA5 surface height exceeds 1457 m (850 hPa in the
    standard atmosphere) are treated as missing: set to NaN before every
    Gaussian smoothing, which becomes a normalised convolution, and excluded
    from the spatial rank correlation (interior mask AND above-surface).
  * MERRA-2: its own missing values (below-surface fill) define the mask; no
    zero-filling.

The original experiment scripts are reused unchanged by monkey-patching
`compute_vorticity` / `envelope_rho_profile` (and B18's `_ells` variant) in
every module namespace that holds them; outputs go to `results/<exp>_v12/`.
The entropy profile of Phase 2 (not used by article 1) is stubbed to NaN to
halve the surrogate cost.

Stages: degree (96 ERA5 windows, 199 surrogates) | km (Phase 18, 80 windows) |
        model (Phase 13: free run + ERA5 lattice) | merra | consolidate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_HERE.parent)                      # scripts use paths relative to the repo root

import experiment_B2_scale_irreversibility as b2  # noqa: E402
import experiment_B32_vorticity_mask as vm  # noqa: E402

SUFFIX = "_v12"
DEG_SETS = [("data/b1", "experiment_B2_scale_irreversibility"),
            ("data/b2b", "experiment_B2b_heldout_invariants"),
            ("data/b3", "experiment_B3_scattering_benchmark")]

_ORIG_VORT = b2.compute_vorticity
_MASK_CACHE: dict = {}
_CUR: dict = {}


def _valid_for(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    key = (len(lat), len(lon), round(float(lat[0]), 4), round(float(lon[0]), 4),
           round(float(lat[-1]), 4), round(float(lon[-1]), 4))
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = ~vm.below_surface_mask(lat, lon)
    return _MASK_CACHE[key]


def v12_compute_vorticity(u, v, lat, lon):
    om = _ORIG_VORT(u, v, lat, lon)
    term = (np.nan_to_num(u) * np.tan(np.deg2rad(np.asarray(lat, float)))[None, :, None] / vm.EARTH_M).astype(np.float32)
    _CUR["valid"] = _valid_for(np.asarray(lat, float), np.asarray(lon, float))
    return om + term


def v12_envelope_rho_profile(omega, dx_km, dy_km, mask, ells_km=None):
    ells = list(ells_km) if ells_km is not None else list(b2.ELLS_KM)
    valid = _CUR.get("valid")
    if valid is None or valid.shape != omega.shape[1:]:
        raise RuntimeError("V12 mask not set for this grid: compute_vorticity must precede envelope_rho_profile")
    return vm.rho_profile(omega, dx_km, dy_km, mask & valid, valid, ells)


def v12_entropy_stub(u, v, lat, lon):
    return np.full((u.shape[0], 6), np.nan, dtype=np.float32)


def _patch_everywhere(mod_suffix: str, attr: str, value) -> int:
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is not None and (name == mod_suffix or name.endswith("." + mod_suffix)) and hasattr(mod, attr):
            setattr(mod, attr, value)
            n += 1
    return n


def apply_patches(verbose: bool = True) -> None:
    import experiment_B9_predictability  # noqa: F401
    import experiment_B13_free_running  # noqa: F401
    import experiment_B18_equal_km_regions  # noqa: F401
    counts = {
        "b2.compute_vorticity": _patch_everywhere("experiment_B2_scale_irreversibility", "compute_vorticity", v12_compute_vorticity),
        "b2.envelope_rho_profile": _patch_everywhere("experiment_B2_scale_irreversibility", "envelope_rho_profile", v12_envelope_rho_profile),
        "b2.entropy_profile": _patch_everywhere("experiment_B2_scale_irreversibility", "entropy_profile", v12_entropy_stub),
        "b9.compute_vorticity": _patch_everywhere("experiment_B9_predictability", "compute_vorticity", v12_compute_vorticity),
        "b13.envelope_rho_profile": _patch_everywhere("experiment_B13_free_running", "envelope_rho_profile", v12_envelope_rho_profile),
        "b18.compute_vorticity": _patch_everywhere("experiment_B18_equal_km_regions", "compute_vorticity", v12_compute_vorticity),
        "b18.envelope_rho_profile_ells": _patch_everywhere("experiment_B18_equal_km_regions", "envelope_rho_profile_ells", v12_envelope_rho_profile),
        "b3.compute_vorticity": _patch_everywhere("experiment_B3_scattering_benchmark", "compute_vorticity", v12_compute_vorticity),
    }
    if verbose:
        print("patched:", counts, flush=True)
    assert all(v >= 1 for k, v in counts.items() if not k.startswith("b3")), counts


# ---------------------------------------------------------------------------

def stage_degree(workers: int, n_sur: int, only: str | None) -> None:
    for data_dir, exp in DEG_SETS:
        out_dir = _HERE / "results" / (exp + SUFFIX)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(data_dir).glob("era5_wind850_*.nc"))
        if only:
            files = [f for f in files if only in f.name]
        for f in files:
            tag = f.stem.replace("era5_wind850_", "")
            if (out_dir / f"{tag}.json").exists():
                continue
            t0 = time.time()
            b2.run_region_window(f, out_dir, n_sur, workers)
            print(f"[{tag}] {exp}{SUFFIX} done in {time.time() - t0:.0f}s", flush=True)


def stage_km(workers: int, only: str | None) -> None:
    import experiment_B18_equal_km_regions as b18
    out_dir = _HERE / "results" / ("experiment_B18_equal_km_regions" + SUFFIX)
    out_dir.mkdir(parents=True, exist_ok=True)
    s_factors, ld_km, ld_ref = b18.deformation_scale_factors()
    table = {}
    for r in b18.REGIONS:
        lat_c, lon_c = b18.region_center(r)
        hla, hlo = b18.km_half_extents_deg(lat_c, 1.0)
        table[r] = {"center": [lat_c, lon_c], "deg_box_NWSE": b18.REGIONS_ALL[r],
                    "km_half_lat_deg": hla, "km_half_lon_deg": hlo, "Ld_km": ld_km[r], "dyn_scale": s_factors[r]}
    table["_Ld_ref_km"] = ld_ref
    table["_nominal_km"] = list(b18.NOMINAL_KM)
    (out_dir / "region_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    primary = [Path("data/b3") / f"era5_wind850_{r}__{w}.nc" for r in b18.REGIONS for w in b18.PRIMARY_WINDOWS]
    heldout = [Path("data/b2b") / f"era5_wind850_{r}__{w}.nc" for r in b18.HELDOUT_REGIONS for w in b18.HELDOUT_WINDOWS]
    if only:
        primary = [f for f in primary if only in f.name]
        heldout = [f for f in heldout if only in f.name]
    results = [b18.run_region_window(f, out_dir, workers, True, s_factors) for f in primary]
    held = [b18.run_region_window(f, out_dir, workers, False, s_factors) for f in heldout]
    if len(results) == 48 and len(held) == 32:
        summary = b18.consolidate(results, held, out_dir,
                                  _HERE / "results" / ("experiment_B3_scattering_benchmark" + SUFFIX),
                                  _HERE / "results" / "experiment_B4_curvature_invariant")
        print(json.dumps({k: v for k, v in summary.items() if not k.startswith("H18b_secondary")}, indent=1, default=str)[:3000])


def stage_model(which: str) -> None:
    import experiment_B13_free_running as b13
    b13.OUT = _HERE / "results" / ("experiment_B13_free_running" + SUFFIX)
    b13.OUT.mkdir(parents=True, exist_ok=True)
    if which in ("model", "all"):
        b13.stage_model()
    if which in ("era5", "all"):
        b13.stage_era5()
    if which in ("tests", "all"):
        res = b13.stage_tests()
        (b13.OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "domain_values"}, indent=1, default=str)[:3000])


def stage_merra() -> None:
    out = _HERE / "results" / "experiment_B32_review_checks"
    for f in (out / "merra").glob("*.json"):
        f.unlink()                          # always recompute: cheap, and avoids stale variants
    vm.stage_merra(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["degree", "km", "model", "merra", "selftest"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-surrogates", type=int, default=199)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--model-which", choices=["model", "era5", "tests", "all"], default="all")
    args = ap.parse_args()
    if args.stage == "merra":
        # experiment_B32_vorticity_mask has its own explicit variants; it must run UNPATCHED
        stage_merra()
        return
    apply_patches()
    if args.stage == "selftest":
        # one window, few surrogates, scratch output; compare with experiment_B32_vorticity_mask V12
        tag = "R4_CASIA__W9_2023JFM"
        out_dir = Path(os.environ.get("B33_SCRATCH", "/tmp")) / "b33_selftest"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = b2.run_region_window(Path("data/b3") / f"era5_wind850_{tag}.nc", out_dir, 3, 3)
        ref = json.loads((_HERE / "results" / "experiment_B32_review_checks" / "era5" / f"{tag}.json").read_text())
        print("P_real v12 (B33):", [round(x, 4) for x in r["P_real"]])
        print("P_real V12 (B32):", [round(x, 4) for x in ref["V12"]])
        print("max abs diff:", max(abs(a - b) for a, b in zip(r["P_real"], ref["V12"])))
        return
    if args.stage == "degree":
        stage_degree(args.workers, args.n_surrogates, args.only)
    elif args.stage == "km":
        stage_km(args.workers, args.only)
    elif args.stage == "model":
        stage_model(args.model_which)


if __name__ == "__main__":
    main()
