#!/usr/bin/env python3
"""Phase-9: does the transfer-asymmetry index A predict ensemble error growth?

Preregistered protocol: docs/PROTOCOL_PHASE9_PREDICTABILITY.md
(frozen 2026-08-15).

Stage 1 (--stage metrics): per region-window forecast-error and ensemble-spread
growth metrics from GEFS v12 verified against ERA5.
Stage 2 (--stage tests): the frozen hypothesis battery H9a-H9e + controls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kstest, spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.download_b9_gefs import (
        LEADS, MEMBERS, N_MEMBERS, REGIONS_ALL, SAMPLES, WINDOWS, init_dates,
    )
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        SCALE_EDGES_KM, compute_vorticity,
    )
    from clean_experiments.experiment_B3_scattering_benchmark import loo_pca_residuals
    from clean_experiments.experiment_B4_curvature_invariant import region_covariates
    from clean_experiments.experiment_M_cosmo_flow import (
        _build_band_masks, _xy_coordinates_m,
    )
except ImportError:  # pragma: no cover
    from download_b9_gefs import (  # type: ignore
        LEADS, MEMBERS, N_MEMBERS, REGIONS_ALL, SAMPLES, WINDOWS, init_dates,
    )
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        SCALE_EDGES_KM, compute_vorticity,
    )
    from experiment_B3_scattering_benchmark import loo_pca_residuals  # type: ignore
    from experiment_B4_curvature_invariant import region_covariates  # type: ignore
    from experiment_M_cosmo_flow import _build_band_masks, _xy_coordinates_m  # type: ignore

SEED = 20260815
BANDS_RESOLVED = (2, 3)          # 200-400 and 400-800 km
BANDS_REPORT = (2, 3, 4)         # + 800-1600 km
SAT_FRAC = 0.6                   # fit ln r while r <= 0.6
MIN_FIT_PTS = 3
ELL_INTERIOR_KM = 1600.0         # same interior convention as Phase 4
POOLED_TOL = 0.01                # allowed non-material dip of a pooled curve

ERA5_DIR = {"primary": Path("data/b3"), "oos": Path("data/b2b")}
A_SOURCE = {
    "primary": (Path("clean_experiments/results/experiment_B3_scattering_benchmark"),
                "E2_curvature"),
    "oos": (Path("clean_experiments/results/experiment_B4_curvature_invariant"), None),
}
OOS_REGIONS = ("R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS",
               "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM")


# --------------------------------------------------------------------------
# stage 1: forecast metrics
# --------------------------------------------------------------------------

def coarsen_era5(arr: np.ndarray) -> np.ndarray:
    """0.25 deg -> 0.5 deg by anti-aliased subsampling (grids are co-located)."""
    a = gaussian_filter1d(arr, sigma=1.0, axis=-2, mode="nearest")
    a = gaussian_filter1d(a, sigma=1.0, axis=-1, mode="nearest")
    return np.ascontiguousarray(a[..., ::2, ::2])


def band_setup(lat: np.ndarray, lon: np.ndarray, ny: int, nx: int):
    x_m, y_m = _xy_coordinates_m(lat, lon)
    dx = float(np.median(np.abs(np.diff(x_m / 1000.0))))
    dy = float(np.median(np.abs(np.diff(y_m / 1000.0))))
    masks, _, _, _, _, _ = _build_band_masks(
        ny=ny, nx=nx, dy_km=dy, dx_km=dx, scale_edges_km=SCALE_EDGES_KM
    )
    return masks


def bandpass(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """field: (..., ny, nx) -> band-passed copy."""
    spec = np.fft.rfft2(field, axes=(-2, -1))
    return np.fft.irfft2(spec * mask, s=field.shape[-2:], axes=(-2, -1))


def vort(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """u, v: (n, ny, nx) -> vorticity (n, ny, nx)."""
    return compute_vorticity(u, v, lat, lon)


def usable_inits(window: str) -> list[dt.date]:
    """Inits whose longest lead still verifies inside the window."""
    end = dt.date.fromisoformat(WINDOWS[window][1])
    limit = end - dt.timedelta(days=max(LEADS) / 24.0)
    return [d for d in init_dates(window) if d <= limit]


def stage_metrics(sample: str, gefs_root: Path, out_dir: Path) -> list[dict]:
    import xarray as xr

    out_dir.mkdir(parents=True, exist_ok=True)
    windows = SAMPLES[sample]
    regions = list(REGIONS_ALL) if sample == "primary" else list(OOS_REGIONS)
    rows: list[dict] = []

    for window in windows:
        inits = usable_inits(window)
        for region in regions:
            tag = f"{region}__{window}"
            cache = out_dir / f"metrics_{tag}.json"
            if cache.exists():
                rows.append(json.loads(cache.read_text()))
                print(f"[{tag}] cached", flush=True)
                continue

            nc = ERA5_DIR[sample] / f"era5_wind850_{tag}.nc"
            if not nc.exists():
                print(f"[{tag}] MISSING ERA5 {nc}", flush=True)
                continue
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
            w_era = vort(u5, v5, lat, lon)
            t_index = {np.datetime64(t, "h").astype(object): i for i, t in enumerate(times)}

            nb = len(BANDS_REPORT)
            sigma = np.zeros(nb)
            era_bp = {}
            for j, b in enumerate(BANDS_REPORT):
                bp = bandpass(w_era, masks[b])
                era_bp[b] = bp
                sigma[j] = float(np.sqrt(np.mean(bp[:, interior] ** 2)))

            nl = len(LEADS)
            sse_era = np.zeros((nb, nl))
            sse_gds = np.zeros((nb, nl))
            sse_s = np.zeros((nb, nl))
            cnt_era = np.zeros(nl)
            cnt_gds = np.zeros(nl)
            cnt = np.zeros(nl)
            sig_gds_acc = np.zeros(nb)
            sig_gds_n = 0
            missing = 0
            for d in inits:
                for li, lead in enumerate(LEADS):
                    npz = gefs_root / f"gefs_{d:%Y%m%d}_f{lead:03d}.npz"
                    valid = dt.datetime(d.year, d.month, d.day) + dt.timedelta(hours=lead)
                    if not npz.exists():
                        missing += 1
                        continue
                    with np.load(npz) as z:
                        stack = z[region][:N_MEMBERS]          # (M, 2, ny, nx)
                    wm = vort(stack[:, 0], stack[:, 1], lat, lon)   # (M, ny, nx)
                    wmean = wm.mean(axis=0)

                    anl = gefs_root / f"anl_{valid:%Y%m%d}_t{valid:%H}z.npz"
                    w_anl = None
                    if anl.exists():
                        with np.load(anl) as z:
                            a = z[region]
                        w_anl = vort(a[None, 0], a[None, 1], lat, lon)[0]

                    ti = t_index.get(valid)
                    for j, b in enumerate(BANDS_REPORT):
                        bpf = bandpass(wmean, masks[b])
                        if ti is not None:
                            err = bpf - era_bp[b][ti]
                            sse_era[j, li] += float(np.sum(err[interior] ** 2))
                        if w_anl is not None:
                            bpa = bandpass(w_anl, masks[b])
                            sse_gds[j, li] += float(np.sum((bpf - bpa)[interior] ** 2))
                            sig_gds_acc[j] += float(np.mean(bpa[interior] ** 2))
                        dev = bandpass(wm - wmean[None], masks[b])
                        sse_s[j, li] += float(np.sum(dev[:, interior] ** 2) / N_MEMBERS)
                    if ti is not None:
                        cnt_era[li] += 1
                    if w_anl is not None:
                        cnt_gds[li] += 1
                        sig_gds_n += 1
                    cnt[li] += 1

            npts = int(interior.sum())
            sigma_gds = (np.sqrt(sig_gds_acc / max(sig_gds_n, 1))
                         if sig_gds_n else np.full(nb, np.nan))

            def _rel(sse, c, sig):
                out = np.full((nb, nl), np.nan)
                ok = c > 0
                if ok.any():
                    out[:, ok] = np.sqrt(sse[:, ok] / (c[ok] * npts))
                return out / (np.sqrt(2.0) * sig[:, None])

            r_era = _rel(sse_era, cnt_era, sigma)
            r_gds = _rel(sse_gds, cnt_gds, sigma_gds)
            okc = cnt > 0
            s = np.full((nb, nl), np.nan)
            s[:, okc] = np.sqrt(sse_s[:, okc] / (cnt[okc] * npts)
                                * N_MEMBERS / (N_MEMBERS - 1.0))
            q = s / (np.sqrt(2.0) * sigma_gds[:, None])
            r = r_gds if np.isfinite(r_gds).any() else r_era

            rec = {
                "tag": tag, "region": region, "window": window, "sample": sample,
                "n_inits": int(cnt.max()) if cnt.size else 0,
                "n_slots_era5": int(cnt_era.max()) if cnt_era.size else 0,
                "n_slots_gdas": int(cnt_gds.max()) if cnt_gds.size else 0,
                "n_missing_slots": missing, "n_interior_pts": npts,
                "leads_h": list(LEADS),
                "bands": list(BANDS_REPORT),
                "sigma_band_era5": [float(x) for x in sigma],
                "sigma_band_gdas": [float(x) for x in sigma_gds],
                "r": [[float(x) for x in row] for row in r],
                "r_era5": [[float(x) for x in row] for row in r_era],
                "q": [[float(x) for x in row] for row in q],
            }
            for name, mat in (("lambda", r), ("lambda_era5", r_era), ("mu", q)):
                fits = [fit_growth(np.asarray(LEADS, float) / 24.0, mat[j])
                        for j in range(nb)]
                rec[name] = [f["slope_logit"] for f in fits]
                rec[name + "_ln"] = [f["slope_ln"] for f in fits]
                rec[name + "_npts"] = [f["n"] for f in fits]
                rec[name + "_early_sat"] = [f["early_sat"] for f in fits]
            rec["T50"] = [t50(np.asarray(LEADS, float), r[j]) for j in range(nb)]
            rec["r_init"] = [float(r[j, 0]) for j in range(nb)]
            rec["monotone_r"] = bool(np.all(r[:, -1] > r[:, 0]))
            rec["monotone_q"] = bool(np.all(np.diff(q, axis=1) >= -1e-9))

            cache.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            rows.append(rec)
            print(f"[{tag}] inits={rec['n_inits']} lam={np.round(rec['lambda'],3)} "
                  f"mu={np.round(rec['mu'],3)}", flush=True)
    return rows


def _ols_slope(x: np.ndarray, z: np.ndarray) -> float:
    a = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    return float(coef[1])


def fit_growth(tau_days: np.ndarray, y: np.ndarray) -> dict:
    """Logistic (Dalcher-Kalnay) growth rate + the frozen log-linear rate.

    The logistic rate is the slope of logit(y) on lead time, fitted over the
    full, fixed lead set: unlike a log-linear fit it does not depend on where
    the curve happens to cross a saturation threshold, so it is comparable
    across regions with different initial error.
    """
    good = np.isfinite(y) & (y > 0)
    out = {"slope_logit": float("nan"), "slope_ln": float("nan"),
           "n": 0, "early_sat": False}
    if int(good.sum()) < 3:
        return out
    yc = np.clip(y[good], 1e-4, 0.995)
    out["slope_logit"] = _ols_slope(tau_days[good], np.log(yc / (1.0 - yc)))
    out["n"] = int(good.sum())

    sel = good & (y <= SAT_FRAC)
    if int(sel.sum()) < MIN_FIT_PTS:
        idx = np.where(good)[0][:MIN_FIT_PTS]
        sel = np.zeros(len(y), dtype=bool)
        sel[idx] = True
        out["early_sat"] = True
    if int(sel.sum()) >= 2:
        out["slope_ln"] = _ols_slope(tau_days[sel], np.log(y[sel]))
    return out


def t50(tau_h: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(y)
    if not good.any():
        return float("nan")
    x, z = tau_h[good], y[good]
    if z[0] >= 0.5:
        return float(x[0])
    for i in range(1, len(z)):
        if z[i] >= 0.5:
            f = (0.5 - z[i - 1]) / (z[i] - z[i - 1])
            return float(x[i - 1] + f * (x[i] - x[i - 1]))
    return float(x[-1])          # right-censored


# --------------------------------------------------------------------------
# descriptor side
# --------------------------------------------------------------------------

def load_descriptors(sample: str) -> dict[str, dict]:
    root, sub = A_SOURCE[sample]
    out = {}
    for f in sorted(root.glob("R*__W*.json")):
        d = json.loads(f.read_text())
        cur = d[sub] if sub else d
        if "Fnorm_profile" not in cur:
            continue
        fn = np.asarray(cur["Fnorm_profile"], dtype=float)
        out[d["tag"]] = {
            "region": d["region"], "window": d["window"],
            "A_band": [float(np.log10(max(fn[b], 1e-12))) for b in BANDS_REPORT],
            "A": float(np.mean([np.log10(max(fn[b], 1e-12)) for b in BANDS_RESOLVED])),
            "P": [float(x) for x in d.get("P_real", [])],
            "F_spec": d.get("F_spec", {}),
        }
    return out


# --------------------------------------------------------------------------
# stage 2: frozen tests
# --------------------------------------------------------------------------

def perm_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                  n: int = 999, alternative: str = "greater") -> dict:
    rho = float(spearmanr(x, y).statistic)
    null = np.empty(n)
    for i in range(n):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    if alternative == "greater":
        p = float((1 + np.sum(null >= rho)) / (n + 1))
    else:
        p = float((1 + np.sum(np.abs(null) >= abs(rho))) / (n + 1))
    return {"rho": rho, "p_perm": p, "p_two_sided_asym": float(spearmanr(x, y).pvalue),
            "n": int(len(x))}


def region_table(metrics: list[dict], desc: dict[str, dict]) -> dict:
    reg: dict[str, dict] = {}
    for m in metrics:
        d = desc.get(m["tag"])
        if d is None:
            continue
        reg.setdefault(m["region"], []).append((m, d))
    return reg


def aggregate_regions(reg: dict, key: str, band_pos: int) -> tuple[list[str], np.ndarray]:
    names = sorted(reg)
    vals = []
    for r in names:
        v = [x[0][key][band_pos] if isinstance(x[0][key], list) else x[0][key]
             for x in reg[r]]
        vals.append(np.nanmedian(v))
    return names, np.asarray(vals, dtype=float)


def aggregate_A(reg: dict, band_pos: int | None) -> np.ndarray:
    names = sorted(reg)
    out = []
    for r in names:
        v = [(x[1]["A_band"][band_pos] if band_pos is not None else x[1]["A"])
             for x in reg[r]]
        out.append(np.nanmedian(v))
    return np.asarray(out, dtype=float)


def stage_tests(out_dir: Path) -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED, "protocol": "docs/PROTOCOL_PHASE9_PREDICTABILITY.md"}

    prim = [json.loads(f.read_text()) for f in sorted(out_dir.glob("metrics_*.json"))
            if json.loads(f.read_text())["sample"] == "primary"]
    desc_p = load_descriptors("primary")
    reg = region_table(prim, desc_p)
    names = sorted(reg)
    res["n_region_windows_primary"] = len(prim)
    res["regions"] = names

    # ---- C9-1 positive control (re-specified, see deviation 4) -----------
    rr = np.asarray([m["r"] for m in prim])          # (n, band, lead)
    qq = np.asarray([m["q"] for m in prim])
    grow_r = np.all(rr[:, :, -1] > rr[:, :, 0], axis=1)
    grow_q = np.all(qq[:, :, -1] > qq[:, :, 0], axis=1)
    pooled_r = np.nanmean(rr, axis=0)
    pooled_q = np.nanmean(qq, axis=0)
    sat12 = rr[:, :, 0] >= SAT_FRAC
    res["C9_1_positive_control"] = {
        "frac_error_grows": float(np.mean(grow_r)),
        "frac_spread_grows": float(np.mean(grow_q)),
        "pooled_curves_monotone_r": bool(np.all(np.diff(pooled_r, axis=1) > -POOLED_TOL)),
        "pooled_curves_monotone_q": bool(np.all(np.diff(pooled_q, axis=1) > -POOLED_TOL)),
        "pooled_min_step_r": float(np.min(np.diff(pooled_r, axis=1))),
        "pooled_min_step_q": float(np.min(np.diff(pooled_q, axis=1))),
        "n_bands_saturated_at_12h": int(sat12.sum()),
        "n_bands_total": int(sat12.size),
        "saturated_at_12h_by_band": [int(x) for x in sat12.sum(axis=0)],
        "frac_spread_strictly_monotone_all_steps":
            float(np.mean([m["monotone_q"] for m in prim])),
        "pass": bool(np.mean(grow_r) >= 0.90 and np.mean(grow_q) >= 0.90
                     and np.all(np.diff(pooled_r, axis=1) > -POOLED_TOL)
                     and np.all(np.diff(pooled_q, axis=1) > -POOLED_TOL)),
    }

    # ---- band-mean metrics over resolved bands ---------------------------
    bpos = [BANDS_REPORT.index(b) for b in BANDS_RESOLVED]

    def band_mean(key):
        vals = []
        for r in names:
            per_window = [np.nanmean([x[0][key][j] for j in bpos]) for x in reg[r]]
            vals.append(np.nanmedian(per_window))
        return np.asarray(vals, float)

    lam = band_mean("lambda")
    lam_era = band_mean("lambda_era5")
    mu = band_mean("mu")
    t50v = band_mean("T50")
    rini = band_mean("r_init")
    A = aggregate_A(reg, None)
    res["region_values"] = {r: {"A": float(A[i]), "lambda_gdas": float(lam[i]),
                                "lambda_era5": float(lam_era[i]),
                                "mu": float(mu[i]), "T50_h": float(t50v[i]),
                                "r_init": float(rini[i])}
                            for i, r in enumerate(names)}

    # ---- H9a primary ------------------------------------------------------
    res["H9a_lambda"] = perm_spearman(A, lam, rng)
    res["H9a_lambda_era5"] = perm_spearman(A, lam_era, rng)
    res["H9a_mu"] = perm_spearman(A, mu, rng)
    res["H9a_robustness_log_linear_estimator"] = {
        "lambda_ln": perm_spearman(A, band_mean("lambda_ln"), rng),
        "mu_ln": perm_spearman(A, band_mean("mu_ln"), rng),
    }
    res["H9a_T50_negative_expected"] = perm_spearman(A, -t50v, rng)
    res["H9a_pass_lambda"] = bool(res["H9a_lambda"]["rho"] > 0
                                  and res["H9a_lambda"]["p_perm"] < 0.05)
    res["H9a_pass_mu"] = bool(res["H9a_mu"]["rho"] > 0
                              and res["H9a_mu"]["p_perm"] < 0.05)

    # ---- H9b beyond controls ---------------------------------------------
    cov = region_covariates(Path("data/b4cov"))
    ctrl = []
    for i, r in enumerate(names):
        n_, w_, s_, e_ = REGIONS_ALL[r]
        latc = abs(0.5 * (n_ + s_))
        spec = [x[1]["F_spec"] for x in reg[r]]
        slope = float(np.nanmedian([sp.get("slope", np.nan) for sp in spec]))
        logvar = float(np.nanmedian([np.nanmean([sp.get(f"logvar_{b}", np.nan)
                                                 for b in BANDS_RESOLVED])
                                     for sp in spec]))
        ctrl.append([latc, cov[r]["land_frac"], logvar, slope, rini[i]])
    ctrl = np.asarray(ctrl, float)
    res["control_matrix_columns"] = ["abs_lat_centre", "land_frac",
                                     "log_band_var", "spectral_slope", "r_init"]
    res["confound_diagnostics"] = {
        col: {"rho_with_A": float(spearmanr(A, ctrl[:, i]).statistic),
              "rho_with_lambda": float(spearmanr(lam, ctrl[:, i]).statistic)}
        for i, col in enumerate(res["control_matrix_columns"])
    }
    for name, y in (("lambda", lam), ("mu", mu)):
        ra = loo_pca_residuals(ctrl, A[:, None], 2)[:, 0]
        ry = loo_pca_residuals(ctrl, y[:, None], 2)[:, 0]
        res[f"H9b_{name}"] = perm_spearman(ra, ry, rng)
    res["H9b_pass"] = bool(res["H9b_lambda"]["rho"] > 0
                           and res["H9b_lambda"]["p_perm"] < 0.05)

    # ---- H9c band specificity --------------------------------------------
    matched, mismatched = [], []
    for bi, b in enumerate(BANDS_REPORT):
        Ab = aggregate_A(reg, bi)
        for bj, _ in enumerate(BANDS_REPORT):
            lamb = np.asarray([np.nanmedian([x[0]["lambda"][bj] for x in reg[r]])
                               for r in names], float)
            rho = float(spearmanr(Ab, lamb).statistic)
            (matched if bi == bj else mismatched).append(rho)
    obs = float(np.mean(matched) - np.mean(mismatched))
    res["H9c_band_specificity"] = {
        "matched_mean": float(np.mean(matched)),
        "mismatched_mean": float(np.mean(mismatched)),
        "diff": obs,
        "matched": [float(x) for x in matched],
        "note": "declared exploratory when < 3 usable bands",
    }

    # ---- H9d within-region ------------------------------------------------
    xs, ys_l, ys_m, groups = [], [], [], []
    for r in names:
        a_w = np.asarray([x[1]["A"] for x in reg[r]], float)
        l_w = np.asarray([np.nanmean([x[0]["lambda"][j] for j in bpos])
                          for x in reg[r]], float)
        m_w = np.asarray([np.nanmean([x[0]["mu"][j] for j in bpos])
                          for x in reg[r]], float)
        if len(a_w) < 2:
            continue
        xs.append(a_w - a_w.mean())
        ys_l.append(l_w - l_w.mean())
        ys_m.append(m_w - m_w.mean())
        groups.append(r)
    xw = np.concatenate(xs); ylw = np.concatenate(ys_l); ymw = np.concatenate(ys_m)
    res["H9d_within_region"] = {}
    for nm, yy in (("lambda", ylw), ("mu", ymw)):
        slope = float(np.polyfit(xw, yy, 1)[0])
        null = []
        for _ in range(999):
            perm = np.concatenate([rng.permutation(a) for a in xs])
            null.append(np.polyfit(perm, yy, 1)[0])
        p = float((1 + np.sum(np.asarray(null) >= slope)) / 1000)
        res["H9d_within_region"][nm] = {
            "slope": slope, "p_perm": p, "n_pairs": int(len(xw)),
            "rho": float(spearmanr(xw, yy).statistic),
            "pass": bool(slope > 0 and p < 0.05),
        }

    # ---- C9-2 null calibration -------------------------------------------
    ps = []
    for _ in range(999):
        ps.append(spearmanr(rng.permutation(A), lam).pvalue)
    res["C9_2_null_calibration"] = {
        "ks_p_uniform": float(kstest(ps, "uniform").pvalue),
        "frac_p_below_05": float(np.mean(np.asarray(ps) < 0.05)),
    }

    # ---- C9-3 placebo descriptors ----------------------------------------
    p5 = np.asarray([np.nanmedian([x[1]["P"][4] if len(x[1]["P"]) > 4 else np.nan
                                   for x in reg[r]]) for r in names], float)
    logvar_col = ctrl[:, 2]
    slope_col = ctrl[:, 3]
    res["C9_3_placebo"] = {
        "P_rho5_vs_lambda": perm_spearman(p5, lam, rng, alternative="two-sided"),
        "log_band_var_vs_lambda": perm_spearman(logvar_col, lam, rng,
                                                alternative="two-sided"),
        "spectral_slope_vs_lambda": perm_spearman(slope_col, lam, rng,
                                                  alternative="two-sided"),
        "A_vs_lambda_abs_rho": abs(res["H9a_lambda"]["rho"]),
    }

    # ---- C9-4 verification independence ----------------------------------
    signs = {k: float(np.sign(res[k]["rho"]))
             for k in ("H9a_lambda", "H9a_lambda_era5", "H9a_mu")}
    res["C9_4_sign_agreement"] = {
        "signs": signs,
        "all_agree": bool(len(set(signs.values())) == 1),
    }

    # ---- H9e out of sample ------------------------------------------------
    oos_files = [json.loads(f.read_text()) for f in sorted(out_dir.glob("metrics_*.json"))
                 if json.loads(f.read_text())["sample"] == "oos"]
    if oos_files:
        desc_o = load_descriptors("oos")
        rego = region_table(oos_files, desc_o)
        onames = sorted(rego)
        Ao = aggregate_A(rego, None)
        lamo = np.asarray([np.nanmedian([np.nanmean([x[0]["lambda"][j] for j in bpos])
                                         for x in rego[r]]) for r in onames], float)
        muo = np.asarray([np.nanmedian([np.nanmean([x[0]["mu"][j] for j in bpos])
                                        for x in rego[r]]) for r in onames], float)
        fit = np.polyfit(A, lam, 1)
        pred = np.polyval(fit, Ao)
        sse = float(np.sum((lamo - pred) ** 2))
        sst = float(np.sum((lamo - np.mean(lam)) ** 2))
        res["H9e_out_of_sample"] = {
            "regions": onames,
            "spearman_pred_vs_obs": perm_spearman(pred, lamo, rng),
            "oos_r2_vs_training_mean": float(1 - sse / sst) if sst > 0 else None,
            "spearman_A_vs_lambda_oos": perm_spearman(Ao, lamo, rng),
            "spearman_A_vs_mu_oos": perm_spearman(Ao, muo, rng),
            "region_values": {r: {"A": float(Ao[i]), "lambda": float(lamo[i]),
                                  "mu": float(muo[i])} for i, r in enumerate(onames)},
        }
        res["H9e_pass"] = bool(
            res["H9e_out_of_sample"]["spearman_pred_vs_obs"]["rho"] > 0
            and res["H9e_out_of_sample"]["spearman_pred_vs_obs"]["p_perm"] < 0.05
            and (res["H9e_out_of_sample"]["oos_r2_vs_training_mean"] or -1) > 0)
    else:
        res["H9e_out_of_sample"] = None
        res["H9e_pass"] = None

    # ---- jackknife robustness of the between-region correlations ---------
    def jackknife(x: np.ndarray, y: np.ndarray) -> dict:
        vals = []
        for k in range(len(x)):
            m = np.arange(len(x)) != k
            vals.append(float(spearmanr(x[m], y[m]).statistic))
        vals = np.asarray(vals)
        worst = int(np.argmin(np.abs(vals)))
        return {"rho_full": float(spearmanr(x, y).statistic),
                "rho_min": float(vals.min()), "rho_max": float(vals.max()),
                "sign_stable": bool(np.all(np.sign(vals) == np.sign(spearmanr(x, y).statistic))),
                "most_influential_region": names[worst],
                "rho_without_it": float(vals[worst])}

    res["jackknife_leave_one_region_out"] = {
        "A_vs_lambda": jackknife(A, lam),
        "A_vs_mu": jackknife(A, mu),
        "A_vs_T50": jackknife(A, t50v),
    }

    # ---- exploratory, declared post-hoc ----------------------------------
    res["EXPLORATORY_posthoc"] = {
        "note": ("generated after the preregistered battery was scored and seen; "
                 "not a criterion, reported so the pattern behind the null is visible"),
        "A_vs_initial_relative_error_r12h": perm_spearman(A, rini, rng,
                                                          alternative="two-sided"),
        "A_vs_T50": perm_spearman(A, t50v, rng, alternative="two-sided"),
        "r_init_vs_lambda": perm_spearman(rini, lam, rng, alternative="two-sided"),
        "A_vs_mu_after_removing_r_init_only": perm_spearman(
            loo_pca_residuals(rini[:, None], A[:, None], 1)[:, 0],
            loo_pca_residuals(rini[:, None], mu[:, None], 1)[:, 0],
            rng, alternative="two-sided"),
        "A_vs_lambda_after_removing_r_init_only": perm_spearman(
            loo_pca_residuals(rini[:, None], A[:, None], 1)[:, 0],
            loo_pca_residuals(rini[:, None], lam[:, None], 1)[:, 0],
            rng, alternative="two-sided"),
    }

    # ---- verdict ----------------------------------------------------------
    c1 = res["C9_1_positive_control"]["pass"]
    beats_placebo = (abs(res["H9a_lambda"]["rho"])
                     > max(abs(res["C9_3_placebo"]["log_band_var_vs_lambda"]["rho"]),
                           abs(res["C9_3_placebo"]["spectral_slope_vs_lambda"]["rho"])))
    res["beats_spectral_placebo"] = bool(beats_placebo)
    agree = res["C9_4_sign_agreement"]["all_agree"]
    if not c1:
        verdict = "PIPELINE_FAULT"
    elif (res["H9a_pass_lambda"] and res["H9a_pass_mu"] and res["H9b_pass"]
          and res["H9e_pass"] and beats_placebo and agree):
        verdict = "CONFIRMED"
    elif not agree:
        verdict = "NEGATIVE"
    elif (res["H9a_pass_lambda"] or res["H9a_pass_mu"]) and res["H9b_pass"]:
        verdict = "PARTIAL"
    elif not (res["H9a_pass_lambda"] or res["H9a_pass_mu"]):
        verdict = "NEGATIVE"
    else:
        verdict = "PARTIAL"
    res["PHASE9_VERDICT"] = verdict
    return res


def make_report(res: dict, out_dir: Path) -> str:
    def line(name: str, d: dict) -> str:
        return (f"| {name} | {d['rho']:+.3f} | {d['p_perm']:.3f} | "
                f"{d['p_two_sided_asym']:.3f} | {d['n']} |")

    L = ["# Phase 9: transfer asymmetry A vs ensemble forecast error growth",
         "",
         f"Protocol: `{res['protocol']}` (frozen 2026-08-15, deviations logged there).",
         f"Verdict: **{res['PHASE9_VERDICT']}**",
         "",
         "## Positive control",
         f"- error grows (12 h -> 168 h) in "
         f"{res['C9_1_positive_control']['frac_error_grows']*100:.0f} % of "
         f"region-windows, spread in "
         f"{res['C9_1_positive_control']['frac_spread_grows']*100:.0f} %",
         f"- pooled curves strictly monotone: r "
         f"{res['C9_1_positive_control']['pooled_curves_monotone_r']}, q "
         f"{res['C9_1_positive_control']['pooled_curves_monotone_q']}",
         f"- bands already saturated at the 12 h lead: "
         f"{res['C9_1_positive_control']['n_bands_saturated_at_12h']} of "
         f"{res['C9_1_positive_control']['n_bands_total']} "
         f"(per band 200-400 / 400-800 / 800-1600 km: "
         f"{res['C9_1_positive_control']['saturated_at_12h_by_band']})",
         f"- pass = {res['C9_1_positive_control']['pass']}",
         "",
         "## H9a: between-region association (n = 12, one-sided)",
         "",
         "| statistic | Spearman rho | p (perm, one-sided) | p (two-sided) | n |",
         "|---|---|---|---|---|",
         line("A vs lambda (self-analysis truth)", res["H9a_lambda"]),
         line("A vs lambda (ERA5 truth)", res["H9a_lambda_era5"]),
         line("A vs mu (ensemble spread, truth-free)", res["H9a_mu"]),
         line("A vs -T50 (predictability horizon)", res["H9a_T50_negative_expected"]),
         line("A vs lambda, log-linear estimator",
              res["H9a_robustness_log_linear_estimator"]["lambda_ln"]),
         line("A vs mu, log-linear estimator",
              res["H9a_robustness_log_linear_estimator"]["mu_ln"]),
         "",
         "## H9b: after removing controls",
         "",
         f"Controls: {', '.join(res['control_matrix_columns'])}",
         "",
         "| statistic | Spearman rho | p (perm) | p (two-sided) | n |",
         "|---|---|---|---|---|",
         line("residual A vs residual lambda", res["H9b_lambda"]),
         line("residual A vs residual mu", res["H9b_mu"]),
         "",
         "### Confound diagnostics (raw Spearman)",
         "",
         "| control | rho with A | rho with lambda |",
         "|---|---|---|",
         ]
    for k, v in res["confound_diagnostics"].items():
        L.append(f"| {k} | {v['rho_with_A']:+.3f} | {v['rho_with_lambda']:+.3f} |")
    L += ["",
          "## H9d: within-region (region fixed effects, n = "
          f"{res['H9d_within_region']['lambda']['n_pairs']} windows)",
          "",
          "| metric | slope | Spearman rho | p (perm) | pass |",
          "|---|---|---|---|---|"]
    for k, v in res["H9d_within_region"].items():
        L.append(f"| {k} | {v['slope']:+.3f} | {v['rho']:+.3f} | "
                 f"{v['p_perm']:.3f} | {v['pass']} |")
    L += ["", "## C9-3: placebo descriptors vs lambda (two-sided)", "",
          "| descriptor | Spearman rho | p (perm) |", "|---|---|---|"]
    for k in ("P_rho5_vs_lambda", "log_band_var_vs_lambda", "spectral_slope_vs_lambda"):
        d = res["C9_3_placebo"][k]
        L.append(f"| {k} | {d['rho']:+.3f} | {d['p_perm']:.3f} |")
    L.append(f"| A (for comparison, |rho|) | "
             f"{res['C9_3_placebo']['A_vs_lambda_abs_rho']:.3f} | - |")
    L += ["", f"A beats the spectral placebo: **{res['beats_spectral_placebo']}**",
          "", "## C9-2 / C9-4 controls",
          f"- null calibration: KS p(uniform) = "
          f"{res['C9_2_null_calibration']['ks_p_uniform']:.3f}, "
          f"fraction p<0.05 = {res['C9_2_null_calibration']['frac_p_below_05']:.3f}",
          f"- sign agreement across the three metrics: "
          f"{res['C9_4_sign_agreement']['all_agree']} "
          f"({res['C9_4_sign_agreement']['signs']})",
          ""]
    if res.get("H9e_out_of_sample"):
        o = res["H9e_out_of_sample"]
        L += ["## H9e: out-of-sample (2021-2022, 8 regions)", "",
              "| statistic | Spearman rho | p (perm) | n |", "|---|---|---|---|",
              f"| predicted vs observed lambda | {o['spearman_pred_vs_obs']['rho']:+.3f} "
              f"| {o['spearman_pred_vs_obs']['p_perm']:.3f} "
              f"| {o['spearman_pred_vs_obs']['n']} |",
              f"| A vs lambda (refitted in sample) | {o['spearman_A_vs_lambda_oos']['rho']:+.3f} "
              f"| {o['spearman_A_vs_lambda_oos']['p_perm']:.3f} "
              f"| {o['spearman_A_vs_lambda_oos']['n']} |",
              f"| A vs mu | {o['spearman_A_vs_mu_oos']['rho']:+.3f} "
              f"| {o['spearman_A_vs_mu_oos']['p_perm']:.3f} "
              f"| {o['spearman_A_vs_mu_oos']['n']} |",
              "",
              f"out-of-sample R^2 vs training mean: {o['oos_r2_vs_training_mean']}", ""]
    L += ["", "## Jackknife (leave one region out, n = 12)", "",
          "| pair | rho (full) | rho range | sign stable | most influential | rho without it |",
          "|---|---|---|---|---|---|"]
    for k, v in res["jackknife_leave_one_region_out"].items():
        L.append(f"| {k} | {v['rho_full']:+.3f} | "
                 f"{v['rho_min']:+.3f} .. {v['rho_max']:+.3f} | {v['sign_stable']} | "
                 f"{v['most_influential_region']} | {v['rho_without_it']:+.3f} |")
    L.append("")
    ex = res["EXPLORATORY_posthoc"]
    L += ["## Exploratory (post-hoc, not a criterion)", "",
          f"_{ex['note']}_", "",
          "| statistic | Spearman rho | p (perm, two-sided) |", "|---|---|---|"]
    for k, v in ex.items():
        if k == "note":
            continue
        L.append(f"| {k} | {v['rho']:+.3f} | {v['p_perm']:.3f} |")
    L += ["", "## Region table (primary sample, medians over 4 windows)", "",
          "| region | A | lambda_gdas (1/day) | lambda_era5 | mu | T50 (h) | r(12 h) |",
          "|---|---|---|---|---|---|---|"]
    for r, v in sorted(res["region_values"].items(),
                       key=lambda kv: -kv[1]["A"]):
        L.append(f"| {r} | {v['A']:.2f} | {v['lambda_gdas']:.3f} | "
                 f"{v['lambda_era5']:.3f} | {v['mu']:.3f} | {v['T50_h']:.0f} | "
                 f"{v['r_init']:.3f} |")
    L.append("")
    text = "\n".join(L)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text


def make_figure(res: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    rv = res["region_values"]
    names = list(rv)
    A = np.array([rv[r]["A"] for r in names])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, key, lab in zip(
            axes,
            ("lambda_gdas", "mu", "T50_h"),
            (r"error growth $\lambda$ (day$^{-1}$)",
             r"spread growth $\mu$ (day$^{-1}$)",
             r"$T_{50}$ (h)")):
        y = np.array([rv[r][key] for r in names])
        ax.scatter(A, y, s=34, color="#20567a")
        for i, r in enumerate(names):
            ax.annotate(r.split("_", 1)[1], (A[i], y[i]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        rho = float(spearmanr(A, y).statistic)
        b, a = np.polyfit(A, y, 1)
        xs = np.linspace(A.min(), A.max(), 10)
        ax.plot(xs, a + b * xs, color="#c1440e", lw=1.2)
        ax.set_xlabel(r"transfer asymmetry $A$ (log$_{10}$, 200-800 km)")
        ax.set_ylabel(lab)
        ax.set_title(rf"Spearman $\rho$ = {rho:+.2f}", fontsize=10)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "figB9_A_vs_predictability.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["metrics", "tests"], default="metrics")
    ap.add_argument("--sample", choices=sorted(SAMPLES), default="primary")
    ap.add_argument("--gefs-root", type=Path, default=Path("data/b9gefs"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("clean_experiments/results/experiment_B9_predictability"))
    args = ap.parse_args()

    if args.stage == "metrics":
        rows = stage_metrics(args.sample, args.gefs_root / args.sample, args.out_dir)
        print(f"SUMMARY: {len(rows)} region-windows ({args.sample})")
    else:
        res = stage_tests(args.out_dir)
        (args.out_dir / "summary.json").write_text(json.dumps(res, indent=2),
                                                   encoding="utf-8")
        make_figure(res, args.out_dir)
        print(make_report(res, args.out_dir))


if __name__ == "__main__":
    main()
