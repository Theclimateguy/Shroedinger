#!/usr/bin/env python3
"""Phase-17: P^eq over the long record (1979-2024, daily 00Z ERA5).

Preregistered protocol: docs/PROTOCOL_PHASE17_PENV_LONG_RECORD.md
(frozen 2026-08-17).

  --stage series : daily composite P and amplitude V per region-year npz
  --stage tests  : monthly assembly, C17-0, H17a/b, C17-1/2, verdict
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
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
        ELLS_KM, EPS, N_STEPS, compute_vorticity, envelope_rho_profile,
        gaussian_bar_grouped,
    )
except ImportError:  # pragma: no cover
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM, EPS, N_STEPS, compute_vorticity, envelope_rho_profile,
        gaussian_bar_grouped,
    )

SEED = 20260817
OUT = _HERE / "results" / "experiment_B17_penv_long_record"
DATA = Path("data/b17daily")
ONI_FILE = Path("data/oni_monthly_psl.txt")
N_DRAW = 999
BANDS = (3, 4)

CONF_YEARS = range(1979, 2017)      # confirmatory 1979-2016
CHECK_YEARS = range(2017, 2025)     # known-era check
ERA_SPLIT = 1998                    # C17-2: 1979-1997 vs 1998-2016
MIN_DAYS = 20


# --------------------------------------------------------------------------
# series
# --------------------------------------------------------------------------

def _series_one(path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    tag = path.stem.replace("era5_wind850daily_", "")
    cached = OUT / f"daily_{tag}.npz"
    if cached.exists():
        return tag
    ds = xr.open_dataset(path)
    tname = "valid_time" if "valid_time" in ds else "time"
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    months = np.asarray(ds[tname].dt.month.values, dtype=int)
    ds.close()
    if u.shape[0] < 360:
        return f"SHORT {tag} nt={u.shape[0]}"
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)
    om = compute_vorticity(u, v, lat, lon)
    rho = envelope_rho_profile(om, dx_km, dy_km, mask)
    P = rho[:, list(BANDS)].mean(axis=1)
    # amplitude: log interior-mean D_i^2 for bands 3,4
    w = np.nan_to_num(om).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km)
            for ell in ELLS_KM]
    vs = []
    for i in BANDS:
        d = bars[i] - bars[i + 1]
        vs.append(np.log(np.mean(d[:, mask] ** 2, axis=1) + EPS))
    V = np.mean(vs, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, P=P.astype(np.float32),
                        V=V.astype(np.float32), month=months[: len(P)])
    return f"DONE {tag}"


def stage_series(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("era5_wind850daily_*.nc"))
    todo = [f for f in files
            if not (OUT / f"daily_{f.stem.replace('era5_wind850daily_', '')}.npz"
                    ).exists()]
    print(f"{len(files)} files on disk, {len(todo)} to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(_series_one, todo)):
            if not msg.startswith("DONE") or (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(todo)}] {msg}", flush=True)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def load_oni() -> dict[tuple[int, int], float]:
    out = {}
    for line in ONI_FILE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 13 and parts[0].isdigit():
            y = int(parts[0])
            for m, val in enumerate(parts[1:], start=1):
                x = float(val)
                if x > -90:
                    out[(y, m)] = x
    return out


def monthly_table() -> dict[str, dict[tuple[int, int], dict]]:
    """region -> {(year, month): {P, V}} from cached daily npz."""
    table: dict[str, dict] = {}
    for f in sorted(OUT.glob("daily_*.npz")):
        tag = f.stem.replace("daily_", "")
        region, year = tag.rsplit("_", 1)
        year = int(year)
        z = np.load(f)
        P, V, mo = z["P"], z["V"], z["month"]
        for m in range(1, 13):
            sel = mo == m
            if sel.sum() >= MIN_DAYS:
                table.setdefault(region, {})[(year, m)] = {
                    "P": float(np.median(P[sel])),
                    "V": float(np.median(V[sel]))}
    return table


def anomalies(series: dict[tuple[int, int], dict], key: str
              ) -> dict[tuple[int, int], float]:
    """Deseasonalize with month-of-year climatology from CONF_YEARS only."""
    clim = {}
    for m in range(1, 13):
        vals = [v[key] for (y, mm), v in series.items()
                if mm == m and y in CONF_YEARS]
        clim[m] = float(np.mean(vals)) if vals else np.nan
    return {(y, m): v[key] - clim[m] for (y, m), v in series.items()
            if np.isfinite(clim[m])}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def region_rho(anom: dict, oni: dict, years: range,
               shift: int = 0) -> float:
    """Spearman(P~, ONI) over months of `years`; ONI circularly shifted
    by `shift` months within the full 1979-2024 index."""
    keys = sorted(k for k in anom if k[0] in years)
    if shift:
        all_keys = sorted(k for k in anom)
        idx = {k: i for i, k in enumerate(all_keys)}
        n = len(all_keys)
        x = np.array([oni.get(all_keys[(idx[k] + shift) % n], np.nan)
                      for k in keys])
    else:
        x = np.array([oni.get(k, np.nan) for k in keys])
    y = np.array([anom[k] for k in keys])
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 24:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def pooled_S(anoms: dict[str, dict], oni: dict, years: range,
             shift: int = 0) -> float:
    vals = [region_rho(a, oni, years, shift) for a in anoms.values()]
    return float(np.nanmean(vals))


def c17_0(anom: dict, rng: np.random.Generator) -> dict:
    """Interannual variance ratio vs month-across-year shuffle null."""
    years = sorted({y for (y, m) in anom if y in CONF_YEARS})
    mat = np.full((len(years), 12), np.nan)
    for i, y in enumerate(years):
        for m in range(1, 13):
            if (y, m) in anom:
                mat[i, m - 1] = anom[(y, m)]
    valid = np.isfinite(mat)

    def fratio(a: np.ndarray) -> float:
        ann = np.nanmean(a, axis=1)
        return float(np.var(ann, ddof=1)
                     / (np.nanvar(a, ddof=1) / 12.0))

    f_obs = fratio(mat)
    null = np.empty(N_DRAW)
    for i in range(N_DRAW):
        sh = mat.copy()
        for m in range(12):
            col = sh[:, m]
            fin = np.isfinite(col)
            col[fin] = rng.permutation(col[fin])
            sh[:, m] = col
        null[i] = fratio(sh)
    q95 = float(np.quantile(null, 0.95))
    return {"F": f_obs, "null_q95": q95, "excess": bool(f_obs > q95)}


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE17_PENV_LONG_RECORD.md"}
    oni = load_oni()
    table = monthly_table()
    regions = sorted(table)
    res["n_regions"] = len(regions)
    res["months_per_region"] = {r: len(table[r]) for r in regions}
    if len(regions) < 12:
        res["PHASE17_VERDICT"] = "INCOMPLETE_DATA"
        return res

    anomP = {r: anomalies(table[r], "P") for r in regions}
    anomV = {r: anomalies(table[r], "V") for r in regions}

    # ---- C17-0 -----------------------------------------------------------
    c0 = {r: c17_0(anomP[r], rng) for r in regions}
    n_var = sum(1 for v in c0.values() if v["excess"])
    res["C17_0"] = {"per_region": c0, "n_regions_excess": n_var}

    # ---- H17a ------------------------------------------------------------
    s_obs = pooled_S(anomP, oni, CONF_YEARS)
    shifts = rng.integers(60, 397, size=N_DRAW) * rng.choice([-1, 1],
                                                             size=N_DRAW)
    null = np.array([pooled_S(anomP, oni, CONF_YEARS, int(k))
                     for k in shifts])
    p2 = float((np.sum(np.abs(null) >= abs(s_obs)) + 1) / (N_DRAW + 1))
    res["H17a"] = {"S": s_obs, "p_two_sided": p2,
                   "null_sd": float(np.nanstd(null)),
                   "per_region_rho": {r: region_rho(anomP[r], oni, CONF_YEARS)
                                      for r in regions}}
    res["H17a_pass"] = bool(p2 < 0.05)

    # ---- C17-1 placebo ---------------------------------------------------
    s_v = pooled_S(anomV, oni, CONF_YEARS)
    res["C17_1"] = {"S_placebo": s_v}
    res["C17_1_pass"] = bool(abs(s_obs) > abs(s_v))

    # ---- C17-2 era split -------------------------------------------------
    s_e1 = pooled_S(anomP, oni, range(1979, ERA_SPLIT))
    s_e2 = pooled_S(anomP, oni, range(ERA_SPLIT, 2017))
    res["C17_2"] = {"S_1979_1997": s_e1, "S_1998_2016": s_e2}
    res["C17_2_pass"] = bool(np.sign(s_e1) == np.sign(s_e2))

    # ---- H17b check era --------------------------------------------------
    s_chk = pooled_S(anomP, oni, CHECK_YEARS)
    res["H17b"] = {"S_2017_2024": s_chk,
                   "same_sign_as_H17a": bool(np.sign(s_chk) == np.sign(s_obs))}

    # ---- descriptives ----------------------------------------------------
    desc: dict = {}
    trends = {}
    for r in regions:
        keys = sorted(k for k in anomP[r] if k[0] in CONF_YEARS)
        t = np.array([y + (m - 0.5) / 12.0 for (y, m) in keys])
        y_ = np.array([anomP[r][k] for k in keys])
        t = t - t.mean()
        trends[r] = float((t @ y_) / (t @ t) * 10.0)
    desc["trend_per_decade"] = trends
    # periodogram of the 12-region mean anomaly series
    keys = sorted(set.intersection(*(set(anomP[r]) for r in regions)))
    keys = [k for k in keys if k[0] in CONF_YEARS]
    mean_series = np.array([np.mean([anomP[r][k] for r in regions])
                            for k in keys])
    f = np.fft.rfftfreq(len(mean_series), d=1.0 / 12.0)   # cycles/year
    pw = np.abs(np.fft.rfft(mean_series - mean_series.mean())) ** 2
    desc["periodogram_cpy"] = [[float(a), float(b)] for a, b in
                               zip(f[1:], pw[1:])]
    # lagged pooled cross-correlation
    lags = {}
    for lag in range(-24, 25, 3):
        lags[lag] = pooled_S(anomP, oni, CONF_YEARS, lag)
    desc["lagged_S"] = lags
    res["descriptive"] = desc

    # ---- verdict ---------------------------------------------------------
    if res["H17a_pass"] and res["C17_1_pass"] and res["C17_2_pass"]:
        v = "ENSO_CONFIRMED"
    elif res["H17a_pass"]:
        v = "ENSO_PARTIAL"
    elif n_var >= 3:
        v = "VARIANCE_WITHOUT_ENSO"
    else:
        v = "STATIC_CONFIRMED"
    res["PHASE17_VERDICT"] = v
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["series", "tests"], default="series")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    if args.stage == "series":
        stage_series(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {k: res.get(k) for k in
                 ("n_regions", "C17_0", "H17a", "C17_1", "C17_2", "H17b",
                  "PHASE17_VERDICT")}
        if isinstance(brief.get("C17_0"), dict):
            brief["C17_0"] = {"n_regions_excess":
                              brief["C17_0"]["n_regions_excess"]}
        print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
