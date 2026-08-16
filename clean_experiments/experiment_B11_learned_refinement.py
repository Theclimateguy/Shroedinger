#!/usr/bin/env python3
"""Phase-11: does A predict the mesoscale variance deficit of learned models?

Preregistered protocol: docs/PROTOCOL_PHASE11_LEARNED_REFINEMENT.md
(frozen 2026-08-15).

  --stage metrics : per system/domain/window mesoscale variance deficit
  --stage tests   : the frozen battery H11a-H11f
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
    from clean_experiments.download_b11_wb2 import DOMAINS, LEADS, init_dates
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B9_predictability import (
        BANDS_REPORT, BANDS_RESOLVED, ELL_INTERIOR_KM, band_setup, bandpass,
        loo_pca_residuals, perm_spearman, vort,
    )
except ImportError:  # pragma: no cover
    from download_b11_wb2 import DOMAINS, LEADS, init_dates  # type: ignore
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B9_predictability import (  # type: ignore
        BANDS_REPORT, BANDS_RESOLVED, ELL_INTERIOR_KM, band_setup, bandpass,
        loo_pca_residuals, perm_spearman, vort,
    )

SEED = 20260815
DATA = Path("data/b11wb2")
OUT = Path("clean_experiments/results/experiment_B11_learned_refinement")
B3 = Path("clean_experiments/results/experiment_B3_scattering_benchmark")
B10 = Path("clean_experiments/results/experiment_B10_irrecoverability")

DETERMINISTIC = ("graphcast", "pangu")
CONTROLS_NEG = ("hres", "gencast_member")
CONTROLS_POS = ("gencast_mean",)
ALL_SYSTEMS = DETERMINISTIC + CONTROLS_POS + CONTROLS_NEG
PRIMARY_LEAD = 120
STEP = 0.25


def domain_grid(name: str) -> tuple[np.ndarray, np.ndarray]:
    n, w, s, e = DOMAINS[name]
    return (np.arange(n, s - 1e-9, -STEP), np.arange(w, e + 1e-9, STEP))


def _window_of(d: dt.datetime) -> str:
    return "W13_2020JFM" if d.month < 7 else "W14_2020JAS"


def stage_metrics() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inits = init_dates()
    cache: dict[str, dict] = {}
    for name in DOMAINS:
        lat, lon = domain_grid(name)
        ny, nx = len(lat), len(lon)
        masks = band_setup(lat, lon, ny, nx)
        dx_km, dy_km = _grid_geometry(lat, lon)
        interior = _interior_mask(ny, nx, ELL_INTERIOR_KM, dx_km, dy_km)
        cache[name] = {"lat": lat, "lon": lon, "masks": masks, "interior": interior}

    for system in ALL_SYSTEMS:
        for lead in LEADS:
            out_file = OUT / f"deficit_{system}_f{lead:03d}.json"
            if out_file.exists():
                print(f"[{system} f{lead}] cached", flush=True)
                continue
            acc: dict[tuple[str, str], dict] = {}
            used = 0
            for d0 in inits:
                fm = DATA / f"{system}_{d0:%Y%m%d%H}_f{lead:03d}.npz"
                valid = d0 + dt.timedelta(hours=lead)
                fe = DATA / f"era5_{valid:%Y%m%d%H}.npz"
                if not fm.exists() or not fe.exists():
                    continue
                used += 1
                with np.load(fm) as zm, np.load(fe) as ze:
                    for name in DOMAINS:
                        if name not in zm or name not in ze:
                            continue
                        c = cache[name]
                        wm = vort(zm[name][None, 0], zm[name][None, 1],
                                  c["lat"], c["lon"])[0]
                        we = vort(ze[name][None, 0], ze[name][None, 1],
                                  c["lat"], c["lon"])[0]
                        key = (name, _window_of(d0))
                        rec = acc.setdefault(key, {"vm": np.zeros(len(BANDS_REPORT)),
                                                   "ve": np.zeros(len(BANDS_REPORT)),
                                                   "n": 0})
                        for j, b in enumerate(BANDS_REPORT):
                            rec["vm"][j] += float(np.mean(
                                bandpass(wm, c["masks"][b])[c["interior"]] ** 2))
                            rec["ve"][j] += float(np.mean(
                                bandpass(we, c["masks"][b])[c["interior"]] ** 2))
                        rec["n"] += 1
            rows = []
            for (name, window), rec in sorted(acc.items()):
                if rec["n"] == 0:
                    continue
                S = rec["vm"] / rec["ve"]
                rows.append({"domain": name, "window": window, "system": system,
                             "lead_h": lead, "n_inits": rec["n"],
                             "bands": list(BANDS_REPORT),
                             "S": [float(x) for x in S],
                             "deficit": [float(1.0 - x) for x in S]})
            out_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"[{system} f{lead}] {len(rows)} domain-windows from {used} inits",
                  flush=True)


# --------------------------------------------------------------------------

def descriptors() -> dict[str, dict]:
    """domain -> A, log band variance, spectral slope (medians over windows)."""
    acc: dict[str, dict] = {}

    def add(domain, fn, fspec):
        fn = np.asarray(fn, dtype=float)
        rec = acc.setdefault(domain, {"A": [], "logvar": [], "slope": []})
        rec["A"].append(float(np.mean([np.log10(max(fn[b], 1e-12))
                                       for b in BANDS_RESOLVED])))
        rec["logvar"].append(float(np.mean([fspec.get(f"logvar_{b}", np.nan)
                                            for b in BANDS_RESOLVED])))
        rec["slope"].append(float(fspec.get("slope", np.nan)))

    for f in sorted(B3.glob("R*__W*.json")):
        d = json.loads(f.read_text())
        if "E2_curvature" not in d:
            continue
        add(d["region"], d["E2_curvature"]["Fnorm_profile"], d.get("F_spec", {}))
    for f in sorted(B10.glob("desc_*.json")):
        d = json.loads(f.read_text())
        add(d["region"], d["Fnorm_profile"], d.get("F_spec", {}))
    return {k: {kk: float(np.nanmedian(vv)) for kk, vv in v.items()}
            for k, v in acc.items()}


def load_deficits(system: str, lead: int) -> dict[str, dict[str, float]]:
    f = OUT / f"deficit_{system}_f{lead:03d}.json"
    if not f.exists():
        return {}
    bpos = [BANDS_REPORT.index(b) for b in BANDS_RESOLVED]
    out: dict[str, dict[str, float]] = {}
    for r in json.loads(f.read_text()):
        out.setdefault(r["domain"], {})[r["window"]] = float(
            np.mean([r["deficit"][j] for j in bpos]))
    return out


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    desc = descriptors()
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE11_LEARNED_REFINEMENT.md"}

    per_system = {s: load_deficits(s, PRIMARY_LEAD) for s in ALL_SYSTEMS}
    names = sorted(set(desc) & set(per_system.get("graphcast", {})))
    res["n_domains"] = len(names)
    if len(names) < 5:
        res["PHASE11_VERDICT"] = "INSUFFICIENT_DATA"
        return res

    A = np.array([desc[n]["A"] for n in names])
    logvar = np.array([desc[n]["logvar"] for n in names])
    slope = np.array([desc[n]["slope"] for n in names])

    def vec(system: str) -> np.ndarray:
        d = per_system.get(system, {})
        return np.array([np.nanmedian(list(d[n].values())) if n in d else np.nan
                         for n in names])

    dvec = {s: vec(s) for s in ALL_SYSTEMS}
    res["domain_values"] = {n: {"A": float(A[i]),
                                **{s: (float(dvec[s][i]) if np.isfinite(dvec[s][i])
                                       else None) for s in ALL_SYSTEMS}}
                            for i, n in enumerate(names)}

    # ---- C11-1 sanity ----------------------------------------------------
    gc, pg, hr = dvec["graphcast"], dvec["pangu"], dvec["hres"]
    res["C11_1_sanity"] = {
        "frac_deficit_positive_graphcast": float(np.nanmean(gc > 0)),
        "frac_deficit_positive_pangu": float(np.nanmean(pg > 0)),
        "median_deficit_graphcast": float(np.nanmedian(gc)),
        "median_deficit_pangu": float(np.nanmedian(pg)),
        "median_abs_deficit_hres": float(np.nanmedian(np.abs(hr))),
        "pass": bool(np.nanmean(gc > 0) >= 0.90 and np.nanmean(pg > 0) >= 0.90
                     and np.nanmedian(np.abs(hr)) < np.nanmedian(np.abs(gc))),
    }

    # ---- H11a / H11b ------------------------------------------------------
    for key, s in (("H11a_graphcast", "graphcast"), ("H11b_pangu", "pangu")):
        m = np.isfinite(dvec[s])
        res[key] = perm_spearman(A[m], dvec[s][m], rng)
    res["H11a_pass"] = bool(res["H11a_graphcast"]["rho"] > 0
                            and res["H11a_graphcast"]["p_perm"] < 0.05)
    res["H11b_pass"] = bool(res["H11b_pangu"]["rho"] > 0
                            and res["H11b_pangu"]["p_perm"] < 0.05)

    # ---- H11c beyond controls --------------------------------------------
    latc = np.array([abs(0.5 * (DOMAINS[n][0] + DOMAINS[n][2])) for n in names])
    try:
        from clean_experiments.experiment_B10_irrecoverability import _land_fraction
    except ImportError:  # pragma: no cover
        from experiment_B10_irrecoverability import _land_fraction  # type: ignore
    land = np.array([_land_fraction(DOMAINS[n]) for n in names])
    ctrl = np.column_stack([latc, land, logvar, slope])
    res["control_matrix_columns"] = ["abs_lat_centre", "land_frac",
                                     "log_band_var", "spectral_slope"]
    res["H11c"] = {}
    for s in DETERMINISTIC:
        m = np.isfinite(dvec[s])
        ra = loo_pca_residuals(ctrl[m], A[m][:, None], 2)[:, 0]
        ry = loo_pca_residuals(ctrl[m], dvec[s][m][:, None], 2)[:, 0]
        res["H11c"][s] = perm_spearman(ra, ry, rng)
    res["H11c_pass"] = bool(all(res["H11c"][s]["rho"] > 0
                                and res["H11c"][s]["p_perm"] < 0.05
                                for s in DETERMINISTIC))

    # ---- H11d mechanism discriminator ------------------------------------
    rhos = {}
    for s in ALL_SYSTEMS:
        m = np.isfinite(dvec[s])
        rhos[s] = (float(spearmanr(A[m], dvec[s][m]).statistic) if m.sum() >= 5
                   else None)
    det_rho = [rhos[s] for s in DETERMINISTIC if rhos[s] is not None]
    neg_rho = [rhos[s] for s in CONTROLS_NEG if rhos[s] is not None]
    controls_fail_criterion = []
    for s in CONTROLS_NEG:
        m = np.isfinite(dvec[s])
        if m.sum() >= 5:
            t = perm_spearman(A[m], dvec[s][m], rng)
            controls_fail_criterion.append(not (t["rho"] > 0 and t["p_perm"] < 0.05))
            res.setdefault("H11d_control_tests", {})[s] = t
    res["H11d"] = {
        "rho_by_system": rhos,
        "mean_rho_deterministic": float(np.mean(det_rho)) if det_rho else None,
        "mean_rho_negative_controls": float(np.mean(neg_rho)) if neg_rho else None,
        "separation": (float(np.mean(det_rho) - np.mean(neg_rho))
                       if det_rho and neg_rho else None),
        "controls_do_not_pass_H11a_criterion": bool(all(controls_fail_criterion))
        if controls_fail_criterion else None,
    }
    res["H11d_pass"] = bool(res["H11d"]["separation"] is not None
                            and res["H11d"]["separation"] > 0
                            and res["H11d"]["controls_do_not_pass_H11a_criterion"])

    # ---- H11e within-domain (seasonal) -----------------------------------
    res["H11e_within_domain"] = {}
    a_by_window = {}
    for f in sorted(B10.glob("desc_*.json")):
        d = json.loads(f.read_text())
        fn = np.asarray(d["Fnorm_profile"], float)
        a_by_window.setdefault(d["region"], {})[d["window"]] = float(
            np.mean([np.log10(max(fn[b], 1e-12)) for b in BANDS_RESOLVED]))
    for s in DETERMINISTIC:
        d = per_system.get(s, {})
        xs, ys = [], []
        for n in names:
            if n not in d or n not in a_by_window:
                continue
            aw = a_by_window[n]
            # JFM descriptor window paired with the JFM deficit window
            pairs = [(aw.get("W11_2024JFM"), d[n].get("W13_2020JFM")),
                     (aw.get("W12_2024JAS"), d[n].get("W14_2020JAS"))]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
            xs.append(x - x.mean()); ys.append(y - y.mean())
        if len(xs) >= 5:
            xw = np.concatenate(xs); yw = np.concatenate(ys)
            sl = float(np.polyfit(xw, yw, 1)[0])
            null = [np.polyfit(np.concatenate([rng.permutation(b) for b in xs]),
                               yw, 1)[0] for _ in range(999)]
            res["H11e_within_domain"][s] = {
                "slope": sl, "n_pairs": int(len(xw)),
                "rho": float(spearmanr(xw, yw).statistic),
                "p_perm": float((1 + np.sum(np.asarray(null) >= sl)) / 1000)}

    # ---- H11f placebo -----------------------------------------------------
    res["H11f_placebo"] = {}
    for s in DETERMINISTIC:
        m = np.isfinite(dvec[s])
        res["H11f_placebo"][s] = {
            "logvar": perm_spearman(logvar[m], dvec[s][m], rng,
                                    alternative="two-sided"),
            "slope": perm_spearman(slope[m], dvec[s][m], rng,
                                   alternative="two-sided"),
        }
    beats = all(abs(res[f"H11{k}_{s}"]["rho"])
                > max(abs(res["H11f_placebo"][s]["logvar"]["rho"]),
                      abs(res["H11f_placebo"][s]["slope"]["rho"]))
                for k, s in (("a", "graphcast"), ("b", "pangu")))
    res["H11f_A_beats_placebos"] = bool(beats)

    # ---- secondary lead ---------------------------------------------------
    res["secondary_lead_24h"] = {}
    for s in ALL_SYSTEMS:
        d24 = load_deficits(s, 24)
        v = np.array([np.nanmedian(list(d24[n].values())) if n in d24 else np.nan
                      for n in names])
        m = np.isfinite(v)
        if m.sum() >= 5:
            res["secondary_lead_24h"][s] = {
                "median_deficit": float(np.nanmedian(v[m])),
                "rho_with_A": float(spearmanr(A[m], v[m]).statistic)}

    # ---- verdict ----------------------------------------------------------
    if not res["C11_1_sanity"]["pass"]:
        verdict = "PIPELINE_FAULT"
    elif not beats:
        verdict = "NEGATIVE"
    elif not res["H11d_pass"]:
        verdict = "NEGATIVE"
    elif not (res["H11a_pass"] or res["H11b_pass"]):
        verdict = "NEGATIVE"
    elif res["H11a_pass"] and res["H11b_pass"] and res["H11c_pass"]:
        verdict = "CONFIRMED"
    else:
        verdict = "PARTIAL"
    res["PHASE11_VERDICT"] = verdict
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["metrics", "tests"], default="metrics")
    args = ap.parse_args()
    if args.stage == "metrics":
        stage_metrics()
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items()
                          if k.startswith(("H11", "C11", "PHASE", "n_", "secondary"))},
                         indent=2))


if __name__ == "__main__":
    main()
