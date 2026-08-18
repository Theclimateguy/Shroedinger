#!/usr/bin/env python3
"""Phase-19 Arm C: the long-record P^eq questions on the km carrier.

Frozen spec: docs/PROTOCOL_PHASE19_KM_DYNAMICS.md, "Arm C execution spec"
(2026-08-18). Consumes data/b17daily (552 region-years, daily 00Z),
km-cropped with the Phase-18 region table; reruns the B17 ENSO statistic
and the interannual variance F-ratio, plus the declared lead-lag
P vs E_syn descriptive.

Stages: --stage series | tests
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
    import clean_experiments.experiment_B17_penv_long_record as b17
    from clean_experiments.experiment_B18_equal_km_regions import km_crop
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM,
        EPS,
        compute_vorticity,
        envelope_rho_profile,
        gaussian_bar_grouped,
    )
except ImportError:
    import experiment_B17_penv_long_record as b17  # type: ignore
    from experiment_B18_equal_km_regions import km_crop  # type: ignore
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM,
        EPS,
        compute_vorticity,
        envelope_rho_profile,
        gaussian_bar_grouped,
    )

SEED = 20260818
OUT = _HERE / "results" / "experiment_B19_armC_km_longrecord"
DATA = Path("data/b17daily")
BANDS = b17.BANDS                      # (3, 4)
CONF_YEARS = b17.CONF_YEARS
MIN_DAYS = b17.MIN_DAYS
N_DRAW = 999
LAGS = range(-6, 7)
EXCESS_REGIONS = ("R1_WPWP", "R3_AMAZ", "R5_SPCZ")


# --------------------------------------------------------------------------
# series
# --------------------------------------------------------------------------

def _series_one(path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    tag = path.stem.replace("era5_wind850daily_", "")
    cached = OUT / f"daily_{tag}.npz"
    if cached.exists():
        return f"SKIP {tag}"
    region = tag.rsplit("_", 1)[0]
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
    lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)
    om = compute_vorticity(u, v, lat, lon)
    rho = envelope_rho_profile(om, dx_km, dy_km, mask)
    P = rho[:, list(BANDS)].mean(axis=1)
    w = np.nan_to_num(om).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km)
            for ell in ELLS_KM]
    vs = []
    for i in BANDS:
        d = bars[i] - bars[i + 1]
        vs.append(np.log(np.mean(d[:, mask] ** 2, axis=1) + EPS))
    V = np.mean(vs, axis=0)
    # monthly synoptic proxy: interior mean of across-days wind variance
    mo_ids, eke = [], []
    for m in range(1, 13):
        sel = months[: u.shape[0]] == m
        if sel.sum() >= MIN_DAYS:
            e = (np.var(u[sel], axis=0) + np.var(v[sel], axis=0))[mask].mean()
            mo_ids.append(m)
            eke.append(float(e))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, P=P.astype(np.float32),
                        V=V.astype(np.float32), month=months[: len(P)],
                        eke_month=np.asarray(mo_ids, dtype=int),
                        eke=np.asarray(eke, dtype=np.float32))
    return f"DONE {tag}"


def stage_series(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("era5_wind850daily_*.nc"))
    todo = [f for f in files
            if not (OUT / f"daily_{f.stem.replace('era5_wind850daily_', '')}.npz"
                    ).exists()]
    print(f"{len(files)} files, {len(todo)} to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(_series_one, todo)):
            if not msg.startswith("DONE") or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(todo)}] {msg}", flush=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def monthly_table_km() -> dict[str, dict[tuple[int, int], dict]]:
    table: dict[str, dict] = {}
    for f in sorted(OUT.glob("daily_*.npz")):
        tag = f.stem.replace("daily_", "")
        region, year = tag.rsplit("_", 1)
        year = int(year)
        z = np.load(f)
        P, V, mo = z["P"], z["V"], z["month"]
        emo, eke = z["eke_month"], z["eke"]
        emap = {int(m): float(e) for m, e in zip(emo, eke)}
        for m in range(1, 13):
            sel = mo == m
            if sel.sum() >= MIN_DAYS:
                table.setdefault(region, {})[(year, m)] = {
                    "P": float(np.median(P[sel])),
                    "V": float(np.median(V[sel])),
                    "E": emap.get(m, np.nan)}
    return table


def lead_lag(anomP: dict[str, dict], anomE: dict[str, dict],
             rng: np.random.Generator) -> dict:
    """Pooled Spearman S(lag) P~(t) vs E~(t+lag); circular-shift null band."""
    def pooled(lag: int, shift: int = 0) -> float:
        vals = []
        for r in anomP:
            keys = sorted(k for k in anomP[r] if k[0] in CONF_YEARS)
            ek_all = sorted(anomE[r].keys())
            idx = {k: i for i, k in enumerate(ek_all)}
            n = len(ek_all)
            x, y = [], []
            for k in keys:
                if k not in idx:
                    continue
                j = (idx[k] + lag + shift) % n
                x.append(anomE[r][ek_all[j]])
                y.append(anomP[r][k])
            x, y = np.asarray(x), np.asarray(y)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 24:
                vals.append(spearmanr(x[m], y[m]).statistic)
        return float(np.nanmean(vals))

    curve = {int(l): pooled(l) for l in LAGS}
    null_max = np.empty(N_DRAW)
    for i in range(N_DRAW):
        s = int(rng.integers(20, 71)) * int(rng.choice([-1, 1]))
        null_max[i] = max(abs(pooled(l, s)) for l in (-6, -3, 0, 3, 6))
    return {"S_by_lag": curve,
            "null_q95_maxabs": float(np.quantile(null_max, 0.95)),
            "max_abs_obs": float(max(abs(v) for v in curve.values())),
            "argmax_lag": int(max(curve, key=lambda l: abs(curve[l])))}


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    oni = b17.load_oni()
    table = monthly_table_km()
    res: dict = {"seed": SEED, "n_regions": len(table),
                 "protocol": "docs/PROTOCOL_PHASE19_KM_DYNAMICS.md (Arm C spec)"}

    anomP = {r: b17.anomalies(s, "P") for r, s in table.items()}
    anomV = {r: b17.anomalies(s, "V") for r, s in table.items()}
    anomE = {r: b17.anomalies(s, "E") for r, s in table.items()}

    # ---- H-C1: ENSO on km ------------------------------------------------
    s_obs = b17.pooled_S(anomP, oni, CONF_YEARS)
    shifts = [int(k) for k in rng.integers(20, 71, N_DRAW)
              * rng.choice([-1, 1], N_DRAW)]
    null = np.array([b17.pooled_S(anomP, oni, CONF_YEARS, k) for k in shifts])
    p_two = float((1 + np.sum(np.abs(null) >= abs(s_obs))) / (N_DRAW + 1))
    res["H_C1_enso_km"] = {
        "S_pooled": s_obs, "p_two_sided": p_two,
        "null_q95_abs": float(np.quantile(np.abs(null), 0.95)),
        "per_region_rho": {r: b17.region_rho(anomP[r], oni, CONF_YEARS)
                           for r in sorted(anomP)},
        "significant": bool(p_two < 0.05),
        "S_pooled_V_control": b17.pooled_S(anomV, oni, CONF_YEARS),
    }

    # ---- H-C2: variance excess on km ------------------------------------
    per = {r: b17.c17_0(anomP[r], rng) for r in sorted(anomP)}
    res["H_C2_variance_km"] = {"per_region": per}
    retained = sum(1 for r in EXCESS_REGIONS if per[r]["excess"])
    res["H_C2_variance_km"]["excess_regions_deg"] = list(EXCESS_REGIONS)
    res["H_C2_variance_km"]["n_retained_of_3"] = retained
    res["H_C2_variance_km"]["carrier_robust"] = bool(retained >= 2)
    res["H_C2_variance_km"]["new_excess_km"] = [
        r for r in per if per[r]["excess"] and r not in EXCESS_REGIONS]

    # ---- secondary: lead-lag P vs E_syn ---------------------------------
    res["leadlag_P_vs_Esyn"] = lead_lag(anomP, anomE, rng)

    # ---- verdict ---------------------------------------------------------
    enso_leg = "b_DILUTION_FOUND" if res["H_C1_enso_km"]["significant"] \
        else "a_ENSO_STAYS_NULL"
    var_leg = ("a_EXCESS_CARRIER_ROBUST"
               if res["H_C2_variance_km"]["carrier_robust"]
               else "b_EXCESS_GEOMETRY_ARTEFACT")
    res["ARM_C_VERDICT"] = [enso_leg, var_leg]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["series", "tests"], default="series")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    if args.stage == "series":
        stage_series(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {"H_C1": {k: res["H_C1_enso_km"][k] for k in
                          ("S_pooled", "p_two_sided", "significant")},
                 "H_C2_retained": res["H_C2_variance_km"]["n_retained_of_3"],
                 "H_C2_new_excess": res["H_C2_variance_km"]["new_excess_km"],
                 "leadlag": {k: res["leadlag_P_vs_Esyn"][k] for k in
                             ("max_abs_obs", "argmax_lag", "null_q95_maxabs")},
                 "ARM_C_VERDICT": res["ARM_C_VERDICT"]}
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
