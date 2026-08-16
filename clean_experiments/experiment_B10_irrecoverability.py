#!/usr/bin/env python3
"""Phase-10: does A predict analysis-time mesoscale irrecoverability?

Preregistered protocol: docs/PROTOCOL_PHASE10_ANALYSIS_IRRECOVERABILITY.md
(frozen 2026-08-15).

Stages:
  --stage descriptors : A, P and spectral features for the 27 new domains
  --stage metrics     : D (inter-analysis disagreement) and r12 (12 h error)
  --stage tests       : the frozen battery H10a-H10g
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.download_b9_gefs import N_MEMBERS
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        envelope_rho_profile, spectral_features,
    )
    from clean_experiments.experiment_B3_scattering_benchmark import loo_pca_residuals
    from clean_experiments.experiment_B4_curvature_invariant import curvature_profiles
    from clean_experiments.experiment_B9_predictability import (
        BANDS_REPORT, BANDS_RESOLVED, ELL_INTERIOR_KM, band_setup, bandpass,
        coarsen_era5, perm_spearman, vort,
    )
    from clean_experiments.regions_b10 import REGIONS_B10, WINDOWS_B10
    from clean_experiments.download_b9_gefs import REGIONS_ALL
except ImportError:  # pragma: no cover
    from download_b9_gefs import N_MEMBERS, REGIONS_ALL  # type: ignore
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        envelope_rho_profile, spectral_features,
    )
    from experiment_B3_scattering_benchmark import loo_pca_residuals  # type: ignore
    from experiment_B4_curvature_invariant import curvature_profiles  # type: ignore
    from experiment_B9_predictability import (  # type: ignore
        BANDS_REPORT, BANDS_RESOLVED, ELL_INTERIOR_KM, band_setup, bandpass,
        coarsen_era5, perm_spearman, vort,
    )
    from regions_b10 import REGIONS_B10, WINDOWS_B10  # type: ignore

SEED = 20260815
MIN_ANALYSIS_TIMES = 150
LEAD_SHORT = 12
INIT_STRIDE = 5

OUT = Path("clean_experiments/results/experiment_B10_irrecoverability")
ERA5_B10 = Path("data/b10era5")
EDA_B10 = Path("data/b10eda")
GEFS_B10 = Path("data/b9gefs/b10")
GEFS_PROG = Path("data/b9gefs_prog/b10")
ERA5_PROG = Path("data/b3")
B9_METRICS = Path("clean_experiments/results/experiment_B9_predictability")
B3_RESULTS = Path("clean_experiments/results/experiment_B3_scattering_benchmark")

WINDOW_SPAN = {"W11_2024JFM": ("2024-01-01", "2024-03-31"),
               "W12_2024JAS": ("2024-07-01", "2024-09-30")}


# --------------------------------------------------------------------------
# stage: descriptors for the new domains
# --------------------------------------------------------------------------

def stage_descriptors() -> None:
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    for region in REGIONS_B10:
        for window in WINDOWS_B10:
            tag = f"{region}__{window}"
            cache = OUT / f"desc_{tag}.json"
            if cache.exists():
                print(f"[{tag}] cached", flush=True)
                continue
            nc = ERA5_B10 / f"era5_wind850_{tag}.nc"
            if not nc.exists():
                print(f"[{tag}] MISSING {nc}", flush=True)
                continue
            ds = xr.open_dataset(nc)
            lat = np.asarray(ds["latitude"].values, dtype=float)
            lon = np.asarray(ds["longitude"].values, dtype=float)
            u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
            v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
            ds.close()
            dx_km, dy_km = _grid_geometry(lat, lon)
            mask = _interior_mask(u.shape[1], u.shape[2], 1600.0, dx_km, dy_km)
            omega = vort(u, v, lat, lon)
            rec = {
                "tag": tag, "region": region, "window": window,
                **curvature_profiles(u, v, lat, lon),
                "P_real": [float(x) for x in
                           np.median(envelope_rho_profile(omega, dx_km, dy_km, mask), axis=0)],
                "F_spec": spectral_features(omega, dx_km, dy_km, mask),
            }
            cache.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            fn = np.asarray(rec["Fnorm_profile"])
            print(f"[{tag}] A={np.mean([np.log10(fn[b]) for b in BANDS_RESOLVED]):.2f}",
                  flush=True)


# --------------------------------------------------------------------------
# stage: D and r12
# --------------------------------------------------------------------------

def _window_dates(window: str) -> list[dt.date]:
    d0, d1 = (dt.date.fromisoformat(s) for s in WINDOW_SPAN[window])
    out, d = [], d0
    while d <= d1:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def _inits(window: str) -> list[dt.date]:
    d0, d1 = (dt.date.fromisoformat(s) for s in WINDOW_SPAN[window])
    out, d = [], d0
    while d <= d1:
        out.append(d)
        d += dt.timedelta(days=INIT_STRIDE)
    return out


def _metrics_one(region: str, window: str, era_dir: Path, gefs_dir: Path,
                 era_prefix: str) -> dict | None:
    import xarray as xr
    tag = f"{region}__{window}"
    nc = era_dir / f"{era_prefix}_{tag}.nc"
    if not nc.exists():
        print(f"[{tag}] MISSING {nc}", flush=True)
        return None
    ds = xr.open_dataset(nc)
    lat = np.asarray(ds["latitude"].values)[::2]
    lon = np.asarray(ds["longitude"].values)[::2]
    times = np.asarray(ds["valid_time"].values)
    u5 = coarsen_era5(np.asarray(ds["u"].squeeze().values, dtype=np.float32))
    v5 = coarsen_era5(np.asarray(ds["v"].squeeze().values, dtype=np.float32))
    ds.close()

    ny, nx = u5.shape[-2:]
    masks = band_setup(lat, lon, ny, nx)
    dx_km, dy_km = _grid_geometry(lat, lon)
    interior = _interior_mask(ny, nx, ELL_INTERIOR_KM, dx_km, dy_km)
    npts = int(interior.sum())
    w_era = vort(u5, v5, lat, lon)
    t_index = {np.datetime64(t, "h").astype(object): i for i, t in enumerate(times)}

    nb = len(BANDS_REPORT)
    era_bp, sigma = {}, np.zeros(nb)
    for j, b in enumerate(BANDS_REPORT):
        bp = bandpass(w_era, masks[b])
        era_bp[b] = bp
        sigma[j] = float(np.sqrt(np.mean(bp[:, interior] ** 2)))

    # ---- D: inter-analysis disagreement -------------------------------
    sse_d = np.zeros(nb)
    sig_gds_acc = np.zeros(nb)
    n_d = 0
    for d in _window_dates(window):
        for cycle in ("00", "12"):
            f = gefs_dir / f"anl_{d:%Y%m%d}_t{cycle}z.npz"
            valid = dt.datetime(d.year, d.month, d.day, int(cycle))
            if not f.exists() or valid not in t_index:
                continue
            with np.load(f) as z:
                if region not in z:
                    continue
                a = z[region]
            w_a = vort(a[None, 0], a[None, 1], lat, lon)[0]
            ti = t_index[valid]
            for j, b in enumerate(BANDS_REPORT):
                bpa = bandpass(w_a, masks[b])
                sse_d[j] += float(np.sum((bpa - era_bp[b][ti])[interior] ** 2))
                sig_gds_acc[j] += float(np.mean(bpa[interior] ** 2))
            n_d += 1
    if n_d:
        sigma_gds = np.sqrt(sig_gds_acc / n_d)
        # symmetric saturation scale: geometric mean of the two analyses'
        # band amplitudes, so that neither system's own smoothing sets the norm
        sat_d = np.sqrt(2.0) * np.sqrt(sigma * sigma_gds)
        D = np.sqrt(sse_d / n_d / npts) / sat_d
    else:
        sigma_gds = np.full(nb, np.nan)
        D = np.full(nb, np.nan)

    # ---- r12: 12 h relative error, self-verified ----------------------
    sse_r = np.zeros(nb)
    sig_a = np.zeros(nb)
    n_r = 0
    for d0 in _inits(window):
        npz = gefs_dir / f"gefs_{d0:%Y%m%d}_f{LEAD_SHORT:03d}.npz"
        valid = dt.datetime(d0.year, d0.month, d0.day) + dt.timedelta(hours=LEAD_SHORT)
        anl = gefs_dir / f"anl_{valid:%Y%m%d}_t{valid:%H}z.npz"
        if not npz.exists() or not anl.exists():
            continue
        with np.load(npz) as z:
            if region not in z:
                continue
            stack = z[region][:N_MEMBERS]
        with np.load(anl) as z:
            a = z[region]
        wmean = vort(stack[:, 0], stack[:, 1], lat, lon).mean(axis=0)
        w_a = vort(a[None, 0], a[None, 1], lat, lon)[0]
        for j, b in enumerate(BANDS_REPORT):
            bpa = bandpass(w_a, masks[b])
            sse_r[j] += float(np.sum((bandpass(wmean, masks[b]) - bpa)[interior] ** 2))
            sig_a[j] += float(np.mean(bpa[interior] ** 2))
        n_r += 1
    if n_r:
        sigma_a = np.sqrt(sig_a / n_r)
        r12 = np.sqrt(sse_r / n_r / npts) / (np.sqrt(2.0) * sigma_a)
    else:
        r12 = np.full(nb, np.nan)

    return {"tag": tag, "region": region, "window": window,
            "bands": list(BANDS_REPORT),
            "n_analysis_times": n_d, "n_inits": n_r,
            "sigma_band_era5": [float(x) for x in sigma],
            "sigma_band_gdas": [float(x) for x in sigma_gds],
            "D": [float(x) for x in D], "r12": [float(x) for x in r12]}


def stage_metrics(which: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if which == "b10":
        pairs = [(r, w, ERA5_B10, GEFS_B10, "era5_wind850") for r in REGIONS_B10
                 for w in WINDOWS_B10]
    else:
        pairs = [(r, w, ERA5_PROG, GEFS_PROG, "era5_wind850") for r in REGIONS_ALL
                 for w in WINDOWS_B10]
    for region, window, era_dir, gefs_dir, prefix in pairs:
        cache = OUT / f"metrics_{region}__{window}.json"
        if cache.exists():
            print(f"[{region}__{window}] cached", flush=True)
            continue
        rec = _metrics_one(region, window, era_dir, gefs_dir, prefix)
        if rec is None:
            continue
        rec["set"] = which
        cache.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[{rec['tag']}] nD={rec['n_analysis_times']} nI={rec['n_inits']} "
              f"D={np.round(rec['D'], 3)} r12={np.round(rec['r12'], 3)}", flush=True)


# --------------------------------------------------------------------------
# stage: tests
# --------------------------------------------------------------------------

def _A_from(rec: dict) -> tuple[float, list[float]]:
    fn = np.asarray(rec["Fnorm_profile"], dtype=float)
    per_band = [float(np.log10(max(fn[b], 1e-12))) for b in BANDS_REPORT]
    head = float(np.mean([np.log10(max(fn[b], 1e-12)) for b in BANDS_RESOLVED]))
    return head, per_band


def _collect(which: str) -> dict[str, dict]:
    """domain -> {A, P5, logvar, slope, D, r12} as medians over its windows."""
    out: dict[str, dict] = {}
    for f in sorted(OUT.glob("metrics_*.json")):
        m = json.loads(f.read_text())
        if m.get("set") != which:
            continue
        region, window = m["region"], m["window"]
        if which == "b10":
            d = OUT / f"desc_{region}__{window}.json"
            if not d.exists():
                continue
            desc = json.loads(d.read_text())
        else:
            d = B3_RESULTS / f"{region}__{window}.json"
            if not d.exists():
                continue
            raw = json.loads(d.read_text())
            desc = {**raw["E2_curvature"], "P_real": raw["P_real"],
                    "F_spec": raw["F_spec"]}
        A, A_band = _A_from(desc)
        bpos = [BANDS_REPORT.index(b) for b in BANDS_RESOLVED]
        rec = out.setdefault(region, {k: [] for k in
                                      ("A", "P5", "logvar", "slope", "D", "r12")})
        rec["A"].append(A)
        rec["P5"].append(desc["P_real"][4] if len(desc["P_real"]) > 4 else np.nan)
        rec["logvar"].append(float(np.mean([desc["F_spec"].get(f"logvar_{b}", np.nan)
                                            for b in BANDS_RESOLVED])))
        rec["slope"].append(float(desc["F_spec"].get("slope", np.nan)))
        rec["D"].append(float(np.mean([m["D"][j] for j in bpos])))
        rec["r12"].append(float(np.mean([m["r12"][j] for j in bpos])))
    return {k: {kk: float(np.nanmedian(vv)) for kk, vv in v.items()}
            for k, v in out.items()}


def _within(which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Per-domain window anomalies for the within-domain arm."""
    per: dict[str, list] = {}
    for f in sorted(OUT.glob("metrics_*.json")):
        m = json.loads(f.read_text())
        if which != "all" and m.get("set") != which:
            continue
        region, window = m["region"], m["window"]
        if m["set"] == "b10":
            d = OUT / f"desc_{region}__{window}.json"
            if not d.exists():
                continue
            desc = json.loads(d.read_text())
        else:
            d = B3_RESULTS / f"{region}__{window}.json"
            if not d.exists():
                continue
            desc = json.loads(d.read_text())["E2_curvature"]
        A, _ = _A_from(desc)
        bpos = [BANDS_REPORT.index(b) for b in BANDS_RESOLVED]
        per.setdefault(region, []).append(
            (A, float(np.mean([m["D"][j] for j in bpos])),
             float(np.mean([m["r12"][j] for j in bpos]))))
    xs, yd, yr, groups = [], [], [], []
    for region, rows in per.items():
        if len(rows) < 2:
            continue
        a = np.asarray([r[0] for r in rows]); d = np.asarray([r[1] for r in rows])
        r = np.asarray([r[2] for r in rows])
        xs.append(a - a.mean()); yd.append(d - d.mean()); yr.append(r - r.mean())
        groups.append(region)
    return (np.concatenate(xs) if xs else np.array([]),
            np.concatenate(yd) if yd else np.array([]),
            np.concatenate(yr) if yr else np.array([]), xs)


def eda_control() -> dict[str, float]:
    """Region-mean resolved-band ERA5 EDA spread (observing-density proxy)."""
    import xarray as xr
    out: dict[str, list] = {}
    for f in sorted(EDA_B10.glob("era5_edaspread850_*.nc")):
        tag = f.stem.replace("era5_edaspread850_", "")
        region = tag.split("__")[0]
        ds = xr.open_dataset(f)
        u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
        v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
        ds.close()
        out.setdefault(region, []).append(
            float(np.sqrt(np.nanmean(u ** 2 + v ** 2))))
    return {k: float(np.nanmedian(v)) for k, v in out.items()}


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE10_ANALYSIS_IRRECOVERABILITY.md"}

    new = _collect("b10")
    names = sorted(new)
    res["n_new_domains"] = len(names)
    res["domains"] = names
    if len(names) < 3:
        res["PHASE10_VERDICT"] = "INSUFFICIENT_DATA"
        return res

    A = np.array([new[n]["A"] for n in names])
    D = np.array([new[n]["D"] for n in names])
    R = np.array([new[n]["r12"] for n in names])
    logvar = np.array([new[n]["logvar"] for n in names])
    slope = np.array([new[n]["slope"] for n in names])
    res["domain_values"] = {n: new[n] for n in names}

    # ---- C10-1 sanity -----------------------------------------------------
    n_times = [json.loads(f.read_text())["n_analysis_times"]
               for f in OUT.glob("metrics_*.json")
               if json.loads(f.read_text()).get("set") == "b10"]
    res["C10_1_sanity"] = {
        "min_analysis_times": int(np.min(n_times)) if n_times else 0,
        "D_finite_positive": bool(np.all(np.isfinite(D) & (D > 0))),
        "r12_finite_positive": bool(np.all(np.isfinite(R) & (R > 0))),
        "D_max": float(np.max(D)), "D_min": float(np.min(D)),
        "r12_max": float(np.max(R)), "r12_min": float(np.min(R)),
        "pass": bool(n_times and np.min(n_times) >= MIN_ANALYSIS_TIMES
                     and np.all(np.isfinite(D) & (D > 0))
                     and np.all(np.isfinite(R) & (R > 0))),
    }

    # ---- H10a / H10b ------------------------------------------------------
    res["H10a_A_vs_D"] = perm_spearman(A, D, rng)
    res["H10b_A_vs_r12"] = perm_spearman(A, R, rng)

    # ---- controls ---------------------------------------------------------
    latc = np.array([abs(0.5 * (REGIONS_B10[n][0] + REGIONS_B10[n][2])) for n in names])
    try:
        from clean_experiments.experiment_B4_curvature_invariant import region_covariates
    except ImportError:  # pragma: no cover
        from experiment_B4_curvature_invariant import region_covariates  # type: ignore
    land = np.array([_land_fraction(REGIONS_B10[n]) for n in names])
    ctrl = np.column_stack([latc, land, logvar, slope])
    res["control_matrix_columns"] = ["abs_lat_centre", "land_frac",
                                     "log_band_var", "spectral_slope"]
    res["H10c"] = {}
    for key, y in (("D", D), ("r12", R)):
        ra = loo_pca_residuals(ctrl, A[:, None], 2)[:, 0]
        ry = loo_pca_residuals(ctrl, y[:, None], 2)[:, 0]
        res["H10c"][key] = perm_spearman(ra, ry, rng)
    res["H10c_pass"] = bool(all(res["H10c"][k]["rho"] > 0
                                and res["H10c"][k]["p_perm"] < 0.05
                                for k in ("D", "r12")))

    # ---- H10d observing-density control -----------------------------------
    eda = eda_control()
    if all(n in eda for n in names):
        e = np.array([eda[n] for n in names])
        ctrl2 = np.column_stack([ctrl, e])
        res["H10d"] = {"eda_vs_A": float(spearmanr(A, e).statistic),
                       "eda_vs_D": float(spearmanr(D, e).statistic),
                       "eda_vs_r12": float(spearmanr(R, e).statistic)}
        for key, y in (("D", D), ("r12", R)):
            ra = loo_pca_residuals(ctrl2, A[:, None], 2)[:, 0]
            ry = loo_pca_residuals(ctrl2, y[:, None], 2)[:, 0]
            res["H10d"][key] = perm_spearman(ra, ry, rng)
        res["H10d_survives"] = bool(all(res["H10d"][k]["rho"] > 0
                                        and res["H10d"][k]["p_perm"] < 0.05
                                        for k in ("D", "r12")))
    else:
        res["H10d"] = None
        res["H10d_survives"] = None

    # ---- H10e within-domain ----------------------------------------------
    xw, ydw, yrw, blocks = _within("all")
    res["H10e_within_domain"] = {}
    if len(xw) >= 8:
        for nm, yy in (("D", ydw), ("r12", yrw)):
            slope_w = float(np.polyfit(xw, yy, 1)[0])
            null = [np.polyfit(np.concatenate([rng.permutation(b) for b in blocks]),
                               yy, 1)[0] for _ in range(999)]
            p = float((1 + np.sum(np.asarray(null) >= slope_w)) / 1000)
            res["H10e_within_domain"][nm] = {
                "slope": slope_w, "p_perm": p, "n_pairs": int(len(xw)),
                "rho": float(spearmanr(xw, yy).statistic),
                "pass": bool(slope_w > 0 and p < 0.05)}

    # ---- H10f placebo -----------------------------------------------------
    res["H10f_placebo"] = {
        "logvar_vs_D": perm_spearman(logvar, D, rng, alternative="two-sided"),
        "logvar_vs_r12": perm_spearman(logvar, R, rng, alternative="two-sided"),
        "slope_vs_D": perm_spearman(slope, D, rng, alternative="two-sided"),
        "slope_vs_r12": perm_spearman(slope, R, rng, alternative="two-sided"),
    }
    beats = (abs(res["H10a_A_vs_D"]["rho"]) > max(
        abs(res["H10f_placebo"]["logvar_vs_D"]["rho"]),
        abs(res["H10f_placebo"]["slope_vs_D"]["rho"]))
        and abs(res["H10b_A_vs_r12"]["rho"]) > max(
        abs(res["H10f_placebo"]["logvar_vs_r12"]["rho"]),
        abs(res["H10f_placebo"]["slope_vs_r12"]["rho"])))
    res["H10f_A_beats_placebos"] = bool(beats)

    # ---- H10g pooled ------------------------------------------------------
    prog = _collect("program")
    if prog:
        pooled = {**new, **prog}
        pn = sorted(pooled)
        Ap = np.array([pooled[n]["A"] for n in pn])
        Dp = np.array([pooled[n]["D"] for n in pn])
        Rp = np.array([pooled[n]["r12"] for n in pn])
        res["H10g_pooled"] = {
            "n": len(pn),
            "A_vs_D": perm_spearman(Ap, Dp, rng),
            "A_vs_r12": perm_spearman(Ap, Rp, rng),
        }
    else:
        res["H10g_pooled"] = None

    # ---- verdict ----------------------------------------------------------
    a_ok = res["H10a_A_vs_D"]["rho"] > 0 and res["H10a_A_vs_D"]["p_perm"] < 0.05
    b_ok = res["H10b_A_vs_r12"]["rho"] > 0 and res["H10b_A_vs_r12"]["p_perm"] < 0.05
    res["H10a_pass"], res["H10b_pass"] = bool(a_ok), bool(b_ok)
    if not res["C10_1_sanity"]["pass"]:
        verdict = "PIPELINE_FAULT"
    elif not (a_ok or b_ok):
        verdict = "NEGATIVE"
    elif res["H10d_survives"] is False:
        verdict = "NEGATIVE"
    elif not beats:
        verdict = "NEGATIVE"
    elif a_ok and b_ok and res["H10c_pass"]:
        verdict = "CONFIRMED"
    else:
        verdict = "PARTIAL"
    res["PHASE10_VERDICT"] = verdict
    return res


_LSM_CACHE: dict = {}


def _land_fraction(area: list[float]) -> float:
    import xarray as xr
    if "lsm" not in _LSM_CACHE:
        inv = xr.open_dataset(Path("data/b4cov") / "era5_invariants_global.nc")
        name = "lsm" if "lsm" in inv else list(inv.data_vars)[-1]
        _LSM_CACHE["lsm"] = inv[name].squeeze().load()
        inv.close()
    lsm = _LSM_CACHE["lsm"]
    lat_name = "latitude" if "latitude" in lsm.dims else "lat"
    lon_name = "longitude" if "longitude" in lsm.dims else "lon"
    n, w, s, e = area
    sub = lsm.sel({lat_name: slice(n, s)})
    lon_vals = (np.asarray(sub[lon_name].values) + 360) % 360
    w360, e360 = (w + 360) % 360, (e + 360) % 360
    m = ((lon_vals >= w360) & (lon_vals <= e360) if w360 <= e360
         else (lon_vals >= w360) | (lon_vals <= e360))
    return float(sub.isel({lon_name: np.where(m)[0]}).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["descriptors", "metrics", "tests"],
                    default="descriptors")
    ap.add_argument("--set", choices=["b10", "program"], default="b10")
    args = ap.parse_args()
    if args.stage == "descriptors":
        stage_descriptors()
    elif args.stage == "metrics":
        stage_metrics(args.set)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items()
                          if k.startswith(("H10", "C10", "PHASE", "n_"))}, indent=2))


if __name__ == "__main__":
    main()
