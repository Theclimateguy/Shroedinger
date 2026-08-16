#!/usr/bin/env python3
"""Phase-13: do the regional geographies survive without data assimilation?

Preregistered protocol: docs/PROTOCOL_PHASE13_FREE_RUNNING.md
(frozen 2026-08-16).

  --stage model : descriptors from the free-running HighResMIP integration
  --stage era5  : the same descriptors from ERA5 on the same 0.5 deg grid
  --stage tests : H13a-H13d and the verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        envelope_rho_profile, spectral_features,
    )
    from clean_experiments.experiment_B4_curvature_invariant import curvature_profiles
    from clean_experiments.experiment_B9_predictability import (
        BANDS_RESOLVED, band_setup, bandpass, coarsen_era5, perm_spearman, vort,
    )
    from clean_experiments.experiment_B12_estimator_validity import (
        PAIRS_RESOLVED, irrecoverability,
    )
    from clean_experiments.download_b11_wb2 import DOMAINS
except ImportError:  # pragma: no cover
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        envelope_rho_profile, spectral_features,
    )
    from experiment_B4_curvature_invariant import curvature_profiles  # type: ignore
    from experiment_B9_predictability import (  # type: ignore
        BANDS_RESOLVED, band_setup, bandpass, coarsen_era5, perm_spearman, vort,
    )
    from experiment_B12_estimator_validity import PAIRS_RESOLVED, irrecoverability  # type: ignore
    from download_b11_wb2 import DOMAINS  # type: ignore

SEED = 20260816
OUT = Path("clean_experiments/results/experiment_B13_free_running")
MODEL_DIR = Path("data/b13hrmip/ECMWF-IFS-HR")
MIN_TIMES = 300
WINDOWS = {"W15_2014JFM": ("201401", "201402", "201403"),
           "W16_2014JAS": ("201407", "201408", "201409")}
LEVEL_PA = 85000.0
ERA5_SETS = [(Path("data/b3"), "2023_24"), (Path("data/b10era5"), "2024"),
             (Path("data/b2b"), "2021_22")]


def _persistence(omega: np.ndarray, masks, interior) -> float:
    ac = []
    for b in BANDS_RESOLVED:
        env = np.abs(bandpass(omega, masks[b]))
        env = gaussian_filter(env, (0, 4, 4), mode="nearest")[:, interior]
        env = env - env.mean(0, keepdims=True)
        num = (env[:-1] * env[1:]).sum(0)
        den = np.sqrt((env[:-1] ** 2).sum(0) * (env[1:] ** 2).sum(0))
        ac.append(float(np.mean(num / np.where(den < 1e-20, 1.0, den))))
    return float(np.mean(ac))


def _descriptors(u: np.ndarray, v: np.ndarray, lat: np.ndarray,
                 lon: np.ndarray) -> dict:
    ny, nx = u.shape[-2:]
    dx_km, dy_km = _grid_geometry(lat, lon)
    interior = _interior_mask(ny, nx, 1600.0, dx_km, dy_km)
    masks = band_setup(lat, lon, ny, nx)
    om = vort(u, v, lat, lon)
    cur = curvature_profiles(u, v, lat, lon)
    fn = np.asarray(cur["Fnorm_profile"], dtype=float)
    rho = envelope_rho_profile(om, dx_km, dy_km, interior)
    prof = np.median(rho, axis=0)
    uu = irrecoverability(om, dx_km, dy_km, interior)
    spec = spectral_features(om, dx_km, dy_km, interior)
    var_b = [float(np.mean(bandpass(om, masks[b])[:, interior] ** 2))
             for b in BANDS_RESOLVED]
    nb = len(fn)
    return {
        "n_times": int(u.shape[0]),
        "A": float(np.mean([np.log10(max(fn[b], 1e-12))
                            for b in BANDS_RESOLVED if b < nb])),
        "P": float(np.mean([prof[i] for i in PAIRS_RESOLVED if i < len(prof)])),
        "P_profile": [float(x) for x in prof],
        "u": float(np.median(np.mean(uu[:, list(PAIRS_RESOLVED)], axis=1))),
        "persist": _persistence(om, masks, interior),
        "band_var": var_b,
        "slope": float(spec.get("slope", np.nan)),
    }


# --------------------------------------------------------------------------

def stage_model() -> None:
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    for window, months in WINDOWS.items():
        cubes = {}
        for var in ("ua", "va"):
            parts = []
            for m in months:
                hits = sorted(MODEL_DIR.glob(f"{var}_*_{m}01*.nc"))
                if not hits:
                    print(f"MISSING {var} {m}", flush=True)
                    continue
                ds = xr.open_dataset(hits[0])
                sel = ds[var].sel(plev=LEVEL_PA)
                parts.append((np.asarray(sel.values, dtype=np.float32),
                              np.asarray(ds["lat"].values, dtype=float),
                              np.asarray(ds["lon"].values, dtype=float)))
                ds.close()
            if not parts:
                continue
            cubes[var] = (np.concatenate([p[0] for p in parts], axis=0),
                          parts[0][1], parts[0][2])
        if len(cubes) != 2:
            continue
        U, glat, glon = cubes["ua"]
        V = cubes["va"][0]
        step = float(abs(glat[1] - glat[0]))
        for name, (n, w, s, e) in DOMAINS.items():
            cache = OUT / f"model_{name}__{window}.json"
            if cache.exists():
                continue
            want_lat = np.arange(n, s - 1e-9, -step)
            li = np.array([int(np.argmin(np.abs(glat - x))) for x in want_lat])
            want_lon = (np.arange(w, e + 1e-9, step) + 360.0) % 360.0
            lo = np.array([int(np.argmin(np.abs(((glon + 360) % 360) - x)))
                           for x in want_lon])
            lat = glat[li]
            lon = np.arange(w, e + 1e-9, step)
            u = U[:, li[:, None], lo[None, :]]
            v = V[:, li[:, None], lo[None, :]]
            rec = {"domain": name, "window": window, "source": "IFS-HR-freerun",
                   **_descriptors(u, v, lat, lon)}
            cache.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"[model {name} {window}] A={rec['A']:.2f} P={rec['P']:.3f} "
                  f"u={rec['u']:.3f} pers={rec['persist']:.3f}", flush=True)


def stage_era5() -> None:
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    for root, label in ERA5_SETS:
        for path in sorted(root.glob("era5_wind850_*.nc")):
            tag = path.stem.replace("era5_wind850_", "")
            domain = tag.split("__")[0]
            if domain not in DOMAINS:
                continue
            cache = OUT / f"era5_{tag}.json"
            if cache.exists():
                continue
            ds = xr.open_dataset(path)
            lat = np.asarray(ds["latitude"].values, dtype=float)[::2]
            lon = np.asarray(ds["longitude"].values, dtype=float)[::2]
            u = coarsen_era5(np.asarray(ds["u"].squeeze().values, dtype=np.float32))
            v = coarsen_era5(np.asarray(ds["v"].squeeze().values, dtype=np.float32))
            ds.close()
            rec = {"domain": domain, "window": tag.split("__")[1],
                   "source": f"ERA5_{label}", **_descriptors(u, v, lat, lon)}
            cache.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"[era5 {tag}] A={rec['A']:.2f} P={rec['P']:.3f} "
                  f"u={rec['u']:.3f} pers={rec['persist']:.3f}", flush=True)


# --------------------------------------------------------------------------

def _collect(prefix: str) -> dict[str, dict]:
    acc: dict[str, dict] = {}
    for f in sorted(OUT.glob(f"{prefix}_*.json")):
        d = json.loads(f.read_text())
        rec = acc.setdefault(d["domain"], {k: [] for k in
                                           ("A", "P", "u", "persist", "band_var",
                                            "n_times", "source")})
        for k in ("A", "P", "u", "persist", "n_times"):
            rec[k].append(d[k])
        rec["band_var"].append(float(np.mean(d["band_var"])))
        rec["source"].append(d["source"])
    return acc


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED, "protocol": "docs/PROTOCOL_PHASE13_FREE_RUNNING.md"}
    mod, era = _collect("model"), _collect("era5")
    names = sorted(set(mod) & set(era))
    res["n_domains"] = len(names)
    if len(names) < 8:
        res["PHASE13_VERDICT"] = "INSUFFICIENT_DATA"
        return res

    def vec(src: dict, key: str) -> np.ndarray:
        return np.array([float(np.median(src[n][key])) for n in names])

    res["C13_1_sanity"] = {
        "min_times_model": int(min(min(mod[n]["n_times"]) for n in names)),
        "min_times_era5": int(min(min(era[n]["n_times"]) for n in names)),
        "median_band_var_ratio": float(np.median(vec(mod, "band_var")
                                                 / vec(era, "band_var"))),
    }
    ratio = res["C13_1_sanity"]["median_band_var_ratio"]
    res["C13_1_sanity"]["pass"] = bool(
        res["C13_1_sanity"]["min_times_model"] >= MIN_TIMES
        and (1 / 3.0) <= ratio <= 3.0)

    tests = {}
    for key, hyp in (("P", "H13a"), ("A", "H13b"), ("u", "H13c"),
                     ("persist", "H13d")):
        x, y = vec(era, key), vec(mod, key)
        tests[hyp] = {"quantity": key, **perm_spearman(x, y, rng)}
    res.update(tests)
    res["H13a_pass"] = bool(res["H13a"]["rho"] > 0 and res["H13a"]["p_perm"] < 0.05)

    # ceiling: ERA5 against itself across periods, where two periods exist
    ceil = {}
    for key in ("P", "A", "u", "persist"):
        pairs = []
        for n in names:
            srcs = era[n]["source"]
            vals = era[n][key]
            groups: dict[str, list] = {}
            for s, v in zip(srcs, vals):
                groups.setdefault(s, []).append(v)
            if len(groups) >= 2:
                gk = sorted(groups)
                pairs.append((float(np.median(groups[gk[0]])),
                              float(np.median(groups[gk[1]]))))
        if len(pairs) >= 8:
            a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
            ceil[key] = {"n": len(pairs),
                         "rho": float(spearmanr(a, b).statistic)}
    res["era5_across_period_ceiling"] = ceil
    res["domain_values"] = {
        n: {f"{k}_{src}": float(np.median(s[n][k]))
            for k in ("A", "P", "u", "persist")
            for src, s in (("model", mod), ("era5", era))}
        for n in names}

    if not res["C13_1_sanity"]["pass"]:
        verdict = "PIPELINE_FAULT"
    elif res["H13a_pass"]:
        verdict = "ATMOSPHERIC"
    elif ceil.get("P", {}).get("rho", 0) > 0.5:
        verdict = "OBSERVING_SYSTEM_ARTEFACT"
    else:
        verdict = "INCONCLUSIVE"
    res["PHASE13_VERDICT"] = verdict
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["model", "era5", "tests"], default="model")
    args = ap.parse_args()
    if args.stage == "model":
        stage_model()
    elif args.stage == "era5":
        stage_era5()
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "domain_values"},
                         indent=2)[:3500])


if __name__ == "__main__":
    main()
