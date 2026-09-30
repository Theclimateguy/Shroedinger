#!/usr/bin/env python3
"""EXPLORATORY (not frozen) analysis of T15 (boundary template vs moving part).
Register: docs/EXPLORATION_PHYSICS_2026-09-29.md, section 11."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import explore_physics_analysis as A  # noqa: E402

SEAS = ("JFM", "JAS")
PAIRS = {"50-100|100-200": (0, 1), "100-200|200-400": (1, 2), "50-100|200-400": (0, 2)}


def load(fz, s, key):
    d = {r["tid"]: r for r in json.loads((A.XP / f"tiles_template_{s}.json").read_text())["tiles"]}
    return (np.array([d[t][f"{key}_tot"] for t in fz.keep]),
            np.array([d[t][f"{key}_tpl"] for t in fz.keep]))


def corr(C, i, j, floor=None):
    den = np.sqrt(np.maximum(C[:, i, i], 1e-30) * np.maximum(C[:, j, j], 1e-30))
    r = C[:, i, j] / den
    if floor is not None:
        r = np.where((C[:, i, i] > floor[:, i]) & (C[:, j, j] > floor[:, j]), r, np.nan)
    return r


def main():
    fz = A.Frozen()
    oc, la = fz.is_ocean, fz.is_land
    res = {}
    for key in ("vort", "T"):
        tot = {s: load(fz, s, key)[0] for s in SEAS}
        tpl = {s: load(fz, s, key)[1] for s in SEAS}
        mov = {s: tot[s] - tpl[s] for s in SEAS}
        share = {s: np.stack([tpl[s][:, i, i] / tot[s][:, i, i] for i in range(3)], axis=1) for s in SEAS}
        sh = np.mean([share[s] for s in SEAS], axis=0)
        out = {"template_share_by_band": {
            "ocean_median": [round(float(np.median(sh[oc, i])), 4) for i in range(3)],
            "land_median": [round(float(np.median(sh[la, i])), 4) for i in range(3)],
            "land_q90": [round(float(np.quantile(sh[la, i], 0.9)), 3) for i in range(3)],
            "cross_season_rho_land": round(float(spearmanr(share["JFM"][la].mean(1), share["JAS"][la].mean(1)).statistic), 3),
            "cross_season_rho_ocean": round(float(spearmanr(share["JFM"][oc].mean(1), share["JAS"][oc].mean(1)).statistic), 3),
            "rho_with": {n: round(float(spearmanr(sh.mean(1), fz.raw[n]).statistic), 3)
                         for n in ("orog_std", "land_frac", "coast_var", "cape_mean", "eke_syn")}},
            "pairs": {}}
        for pn, (i, j) in PAIRS.items():
            r_tot = {s: corr(tot[s], i, j) for s in SEAS}
            r_mov = {s: corr(mov[s], i, j) for s in SEAS}
            # template correlation only where the template is non-negligible
            r_tpl = {s: corr(tpl[s], i, j, floor=0.02 * np.stack(
                [tot[s][:, k, k] for k in range(3)], axis=1)) for s in SEAS}
            c_tpl = {s: tpl[s][:, i, j] / np.sqrt(tot[s][:, i, i] * tot[s][:, j, j]) for s in SEAS}
            m = lambda d: np.nanmean([d[s] for s in SEAS], axis=0)  # noqa: E731
            T_, M_, P_, K_ = m(r_tot), m(r_mov), m(r_tpl), m(c_tpl)

            def cs(d, g):
                ok = g & np.isfinite(d["JFM"]) & np.isfinite(d["JAS"])
                return round(float(spearmanr(d["JFM"][ok], d["JAS"][ok]).statistic), 3)
            out["pairs"][pn] = {
                "total": {"ocean": round(float(T_[oc].mean()), 4), "land": round(float(T_[la].mean()), 4),
                          "ocean_minus_land": round(float(T_[oc].mean() - T_[la].mean()), 4),
                          "sd_land": round(float(T_[la].std()), 4), "sd_ocean": round(float(T_[oc].std()), 4),
                          "cross_season_land": cs(r_tot, la), "cross_season_ocean": cs(r_tot, oc)},
                "moving": {"ocean": round(float(M_[oc].mean()), 4), "land": round(float(M_[la].mean()), 4),
                           "ocean_minus_land": round(float(M_[oc].mean() - M_[la].mean()), 4),
                           "sd_land": round(float(M_[la].std()), 4), "sd_ocean": round(float(M_[oc].std()), 4),
                           "cross_season_land": cs(r_mov, la), "cross_season_ocean": cs(r_mov, oc)},
                "template_corr": {"land_median": round(float(np.nanmedian(P_[la])), 3),
                                  "land_n_defined": int(np.isfinite(P_[la]).sum()),
                                  "ocean_n_defined": int(np.isfinite(P_[oc]).sum()),
                                  "land_frac_positive": round(float(np.nanmean(P_[la] > 0)), 3)},
                "template_contribution": {"ocean": round(float(K_[oc].mean()), 4), "land": round(float(K_[la].mean()), 4),
                                          "cross_season_land": cs(c_tpl, la),
                                          "rho_with_total_land": round(float(spearmanr(K_[la], T_[la]).statistic), 3),
                                          "rho_moving_with_total_land": round(float(spearmanr(M_[la], T_[la]).statistic), 3)},
                "moving_loadings_land": {n: round(float(spearmanr(M_[la], fz.raw[n][la]).statistic), 3)
                                         for n in ("orog_std", "cape_mean", "eke_syn", "abs_lat")},
                "moving_vs_template_share_land": round(float(spearmanr(M_[la], sh[la].mean(1)).statistic), 3)}
        res[key] = out
    # cross-field: same-band T vs vort, joint matrices (indices 0-2 vort, 3-5 T)
    tot = {s: load(fz, s, "joint")[0] for s in SEAS}
    tpl = {s: load(fz, s, "joint")[1] for s in SEAS}
    out = {}
    for b, nm in enumerate(("50-100", "100-200", "200-400")):
        T_ = np.mean([corr(tot[s], b, b + 3) for s in SEAS], axis=0)
        M_ = np.mean([corr(tot[s] - tpl[s], b, b + 3) for s in SEAS], axis=0)
        out[nm] = {"total": {"ocean": round(float(T_[oc].mean()), 4), "land": round(float(T_[la].mean()), 4)},
                   "moving": {"ocean": round(float(M_[oc].mean()), 4), "land": round(float(M_[la].mean()), 4)}}
    res["cross_field_T_vort"] = out
    (A.XP / "analysis_t15.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
