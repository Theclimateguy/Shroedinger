#!/usr/bin/env python3
"""EXPLORATORY (not frozen) analysis of T14 (temperature and cross-field
coupling) plus block split-half reliability of the corrected map.
Register: docs/EXPLORATION_PHYSICS_2026-09-29.md, section 9."""
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
SUF = "_mirror" if "--mirror" in sys.argv else ""


def load(fz, s):
    d = {r["tid"]: r for r in json.loads(
        (A.XP / f"tiles_scalarT_iso{SUF}_{s}.json").read_text())["tiles"]}
    real = np.array([d[t]["real"] for t in fz.keep])      # n,3,7
    sur = np.array([d[t]["sur"] for t in fz.keep])
    return real - sur, real, sur


def sb(r):
    return 2 * r / (1 + r)


def main():
    fz = A.Frozen()
    an = {s: load(fz, s)[0] for s in SEAS}
    raw = {s: load(fz, s)[1] for s in SEAS}
    oc, la = fz.is_ocean, fz.is_land
    a_abs = np.abs(fz.lat)

    def q(s, row, cols):
        return an[s][:, row, cols].mean(axis=1)
    V, T, X = [0, 1], [2, 3], [4, 5, 6]
    Pv = np.mean([q(s, 0, V) for s in SEAS], axis=0)
    Pt = np.mean([q(s, 0, T) for s in SEAS], axis=0)
    Xc = np.mean([q(s, 0, X) for s in SEAS], axis=0)
    iso = A.corrected(fz)
    res = {"control_Pvort_vs_Pc": float(spearmanr(Pv, iso["Pc"]).statistic)}

    rel = {}
    for nm, cols in (("P_vort", V), ("P_T", T), ("X", X)):
        rel[nm] = {}
        for s in SEAS:
            h0, h1 = q(s, 1, cols), q(s, 2, cols)
            rel[nm][s] = {g: round(float(np.corrcoef(h0[m], h1[m])[0, 1]), 3)
                          for g, m in (("all", np.ones(fz.n, bool)), ("ocean", oc), ("land", la))}
            rel[nm][s]["all_SB"] = round(sb(rel[nm][s]["all"]), 3)
        rel[nm]["cross_season"] = {g: round(float(spearmanr(
            q("JFM", 0, cols)[m], q("JAS", 0, cols)[m]).statistic), 3)
            for g, m in (("all", np.ones(fz.n, bool)), ("ocean", oc), ("land", la))}
    res["reliability_block_halves"] = rel

    def desc(P):
        out = {"mean": float(P.mean()), "sd": float(P.std()),
               "ocean": float(P[oc].mean()), "land": float(P[la].mean()),
               "zonal_share": A.zonal_share(fz, P), "by_belt": {}}
        for nm, b in A.belts(fz).items():
            out["by_belt"][nm] = {"ocean": round(float(P[oc & b].mean()), 4),
                                  "land": round(float(P[la & b].mean()), 4)}
        out["ocean_by_abs_lat"] = {f"{lo}-{lo+12}": round(float(
            P[oc & (a_abs >= lo) & (a_abs < lo + 12)].mean()), 4) for lo in (0, 12, 24, 36, 48)}
        out["loadings"] = {n: round(float(spearmanr(P, fz.raw[n]).statistic), 3) for n in A.ALL_COV}
        out["loadings_nonzonal"] = {n: round(float(spearmanr(
            A.nonzonal(fz, P), A.nonzonal(fz, fz.raw[n])).statistic), 3)
            for n in A.ALL_COV if n != "abs_lat"}
        out["loso_r2_8cov"] = round(float(A.loso_r2(fz.X_cov, P, fz.sector)), 4)
        return out
    res["P_vort"], res["P_T"], res["X"] = desc(Pv), desc(Pt), desc(Xc)
    res["raw_levels"] = {nm: [round(float(np.mean([raw[s][:, 0, c].mean() for s in SEAS])), 3)
                              for c in cols] for nm, cols in (("vort", V), ("T", T), ("X", X))}

    def pair(a, b, nm):
        rho, p = fz.spearman_rot(a, b, n_rot=499)
        return {"rho": round(rho, 3), "p_rot": p,
                "nonzonal": round(float(spearmanr(A.nonzonal(fz, a), A.nonzonal(fz, b)).statistic), 3),
                "ocean": round(float(spearmanr(a[oc], b[oc]).statistic), 3),
                "land": round(float(spearmanr(a[la], b[la]).statistic), 3),
                "cross_season": [round(float(spearmanr(x_, y_).statistic), 3) for x_, y_ in nm]}
    res["P_T_vs_P_vort"] = pair(Pt, Pv, [(q("JFM", 0, T), q("JAS", 0, V)), (q("JAS", 0, T), q("JFM", 0, V))])
    res["X_vs_P_vort"] = pair(Xc, Pv, [(q("JFM", 0, X), q("JAS", 0, V)), (q("JAS", 0, X), q("JFM", 0, V))])
    res["X_vs_P_T"] = pair(Xc, Pt, [(q("JFM", 0, X), q("JAS", 0, T)), (q("JAS", 0, X), q("JFM", 0, T))])
    res["gain_over_8cov_on_P_vort"] = {"P_T": fz.loso_gain(Pt, Pv, n_rot=199),
                                       "X": fz.loso_gain(Xc, Pv, n_rot=199)}
    bg = A.below_ground_frac(fz, 850)
    clean = bg == 0
    res["clean_tiles_bg0"] = {"n": int(clean.sum()),
                              "rho_P_T_vs_P_vort": round(float(spearmanr(Pt[clean], Pv[clean]).statistic), 3),
                              "P_T_land": float(Pt[la & clean].mean()), "P_T_ocean": float(Pt[oc].mean())}
    o = np.argsort(Pt)

    def row(i):
        return {"lat": float(fz.lat[i]), "lon": round(float(fz.lon[i]), 1),
                "land": round(float(fz.land[i]), 2), "P_T": round(float(Pt[i]), 3),
                "P_vort": round(float(Pv[i]), 3), "X": round(float(Xc[i]), 3)}
    res["P_T_lowest"] = [row(i) for i in o[:10]]
    res["P_T_highest"] = [row(i) for i in o[-10:][::-1]]
    ox = np.argsort(Xc)
    res["X_lowest"] = [row(i) for i in ox[:8]]
    res["X_highest"] = [row(i) for i in ox[-8:][::-1]]
    np.savez(A.XP / f"t14_maps{SUF}.npz", tid=np.array(fz.keep), P_vort=Pv, P_T=Pt, X=Xc)
    (A.XP / f"analysis_t14{SUF}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
