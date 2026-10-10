#!/usr/bin/env python3
"""Phase 32, checks A3-A7: re-analysis of archived region-window profiles.

Protocol: docs/PROTOCOL_PHASE32_REVIEW_CHECKS.md (frozen 2026-10-08).

Inputs (all archived JSONs, nothing is recomputed from raw fields here):
  results/experiment_B2_scale_irreversibility/R*.json   16 windows, R1-R4, 2017-2019
  results/experiment_B2b_heldout_invariants/R*.json     32 windows, R5-R12, 2021-2022
  results/experiment_B3_scattering_benchmark/R*.json    48 windows, R1-R12, 2023-2024
  results/experiment_B18_equal_km_regions/R*.json       80 windows, equal-km boxes
  results/experiment_B13_free_running/summary.json      39 domains, ERA5 vs free run

Checks:
  A0  sanity: archived Table-2 numbers are reproduced from the JSONs
  A3  steps 2-5 (first pair removed) for every Table-2 row
  A4  region-dependent effective resolution (step 5 only; regionally resolved)
  A5  stratified permutations (season / year / period) and variance components
  A6  spatial-block bootstrap for the ERA5 vs free-run comparison
  A7  bootstrap intervals for the borderline rows (2, 3, 6)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "experiment_B32_review_checks"
SEED = 20261008
N_PERM = 999
N_BOOT = 999

REGION_CENTRE = {  # (lat, lon) of the degree boxes, Table 1 of the manuscript
    "R1_WPWP": (0.0, 150.0), "R2_NATL": (45.0, -40.0), "R3_AMAZ": (-5.0, -55.0),
    "R4_CASIA": (45.0, 80.0), "R5_SPCZ": (-10.0, -160.0), "R6_SATL": (-45.0, -30.0),
    "R7_CONGO": (-5.0, 25.0), "R8_AUS": (-25.0, 135.0), "R9_NPAC": (45.0, -160.0),
    "R10_INDO": (0.0, 80.0), "R11_EURO": (45.0, 10.0), "R12_SAM": (-30.0, -55.0),
}
TROPICAL = {r for r, (la, _) in REGION_CENTRE.items() if abs(la) < 25.0}
PERIOD = {"W1": "I", "W2": "I", "W3": "I", "W4": "I",
          "W5": "II", "W6": "II", "W7": "II", "W8": "II",
          "W9": "III", "W10": "III", "W11": "III", "W12": "III"}
STEPS_ALL = [0, 1, 2, 3, 4]
STEPS_2_5 = [1, 2, 3, 4]
STEPS_4_5 = [3, 4]
STEP_5 = [4]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        if "summary" in f or "e1_" in f:
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if "region" in d and "window" in d:
            out.append(d)
    return out


def _meta(rec: dict) -> dict:
    w = rec["window"]                      # e.g. W10_2023JAS
    code, rest = w.split("_", 1)
    return {"season": rest[-3:], "year": rest[:4], "period": PERIOD[code]}


def load_degree(suffix: str = "") -> list[dict]:
    recs = (_load(str(RES / ("experiment_B2_scale_irreversibility" + suffix) / "R*.json"))
            + _load(str(RES / ("experiment_B2b_heldout_invariants" + suffix) / "R*.json"))
            + _load(str(RES / ("experiment_B3_scattering_benchmark" + suffix) / "R*.json")))
    for r in recs:
        r.update(_meta(r))
        r["P"] = np.asarray(r["P_real"], float)
        r["Psur"] = np.asarray(r["P_surrogate_median"], float)
        r["Panch"] = r["P"] - r["Psur"]
    assert len(recs) == 96, len(recs)
    return recs


def load_km(suffix: str = "") -> list[dict]:
    recs = _load(str(RES / ("experiment_B18_equal_km_regions" + suffix) / "R*.json"))
    out = []
    for r in recs:
        k = r["km_1.0"]
        rec = {"region": r["region"], "window": r["window"], **_meta(r),
               "P": np.asarray(k["P_real"], float),
               "Psur": np.asarray(k["P_surrogate_median"], float),
               "Panch": np.asarray(k["P_anchored"], float),
               "F_spec": k["F_spec"]}
        out.append(rec)
    assert len(out) == 80, len(out)
    return out


def feat(recs: list[dict], key: str) -> np.ndarray:
    keys = sorted(recs[0][key].keys())
    return np.asarray([[r[key][k] for k in keys] for r in recs], float)


# ---------------------------------------------------------------------------
# statistics (Phase 2b conventions, vectorised)
# ---------------------------------------------------------------------------

def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def pdist(x: np.ndarray) -> np.ndarray:
    d = x[:, None, :] - x[None, :, :]
    return np.sqrt((d * d).sum(-1))


def delta(dist: np.ndarray, labels: np.ndarray, pair_mask: np.ndarray | None = None) -> tuple[float, float, float]:
    n = len(labels)
    iu = np.triu_indices(n, 1)
    same = labels[iu[0]] == labels[iu[1]]
    d = dist[iu]
    keep = np.ones(len(d), bool) if pair_mask is None else pair_mask[iu]
    w = np.median(d[same & keep]) if np.any(same & keep) else np.nan
    b = np.median(d[~same & keep]) if np.any(~same & keep) else np.nan
    return float(b - w), float(w), float(b)


def perm_labels(labels: np.ndarray, rng: np.random.Generator, strata: np.ndarray | None) -> np.ndarray:
    if strata is None:
        return labels[rng.permutation(len(labels))]
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = labels[idx][rng.permutation(len(idx))]
    return out


def perm_test(x: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
              strata: np.ndarray | None = None, pair_mask: np.ndarray | None = None,
              n_perm: int = N_PERM) -> dict:
    """Δ = median between − median within; p over permutations of region labels
    (within strata if given); pair_mask restricts which pairs enter Δ."""
    dist = pdist(zscore(np.atleast_2d(x.T).T if x.ndim == 1 else x))
    d_obs, w, b = delta(dist, labels, pair_mask)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = delta(dist, perm_labels(labels, rng, strata), pair_mask)[0]
    p = float((1 + np.sum(null >= d_obs)) / (n_perm + 1))
    return {"diff": round(d_obs, 4), "median_within": round(w, 4), "median_between": round(b, 4),
            "p": p, "pass": bool(d_obs > 0 and p < 0.05), "n": int(len(labels))}


def loo_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    xf = np.hstack([np.ones((n, 1)), x])
    res = np.zeros_like(y, dtype=float)
    for k in range(n):
        tr = np.arange(n) != k
        coef, *_ = np.linalg.lstsq(xf[tr], y[tr], rcond=None)
        res[k] = y[k] - xf[k] @ coef
    return res


def loo_pca_residuals(x_feat: np.ndarray, y: np.ndarray, n_comp: int) -> np.ndarray:
    n = x_feat.shape[0]
    res = np.zeros_like(y, dtype=float)
    for k in range(n):
        tr = np.arange(n) != k
        mu, sd = x_feat[tr].mean(0), x_feat[tr].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        xtr = (x_feat[tr] - mu) / sd
        xte = (x_feat[k] - mu) / sd
        _, _, vt = np.linalg.svd(xtr, full_matrices=False)
        basis = vt[:n_comp].T
        a_tr = np.hstack([np.ones((xtr.shape[0], 1)), xtr @ basis])
        a_te = np.hstack([[1.0], xte @ basis])
        coef, *_ = np.linalg.lstsq(a_tr, y[tr], rcond=None)
        res[k] = y[k] - a_te @ coef
    return res


def region_median_test(x: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                       n_perm: int = N_PERM) -> dict:
    """Region-level test: statistic = mean pairwise distance between region
    medians (standardised space); null = windows reassigned to regions at random,
    keeping the number of windows per region."""
    xz = zscore(x)
    regs = np.unique(labels)

    def stat(lab: np.ndarray) -> float:
        med = np.asarray([np.median(xz[lab == r], axis=0) for r in regs])
        d = pdist(med)
        return float(d[np.triu_indices(len(regs), 1)].mean())

    s_obs = stat(labels)
    null = np.asarray([stat(labels[rng.permutation(len(labels))]) for _ in range(n_perm)])
    return {"stat": round(s_obs, 4), "null_median": round(float(np.median(null)), 4),
            "p": float((1 + np.sum(null >= s_obs)) / (n_perm + 1)), "n_regions": int(len(regs))}


def loo_centroid_accuracy(x: np.ndarray, labels: np.ndarray) -> float:
    xz = zscore(x)
    regs = np.unique(labels)
    hit = 0
    for k in range(len(labels)):
        tr = np.arange(len(labels)) != k
        cents = np.asarray([xz[tr & (labels == r)].mean(axis=0) for r in regs])
        hit += int(regs[np.argmin(np.linalg.norm(cents - xz[k], axis=1))] == labels[k])
    return round(hit / len(labels), 4)


def pair_mask_from(strata: np.ndarray, same: bool) -> np.ndarray:
    m = strata[:, None] == strata[None, :]
    return m if same else ~m


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def rows_table2(deg: list[dict], km: list[dict], steps: list[int], rng: np.random.Generator) -> dict:
    """All Table-2 clustering rows on a chosen subset of steps."""
    b2b = [r for r in deg if r["period"] == "II"]
    b3 = [r for r in deg if r["period"] == "III"]
    km_p = [r for r in km if r["period"] == "III"]
    km_h = [r for r in km if r["period"] == "II"]
    lab = lambda rs: np.asarray([r["region"] for r in rs])  # noqa: E731
    P = lambda rs, key="P": np.asarray([r[key][steps] for r in rs])  # noqa: E731

    out = {}
    out["row1_heldout_raw"] = perm_test(P(b2b), lab(b2b), rng)
    out["row1_b3_raw_48"] = perm_test(P(b3), lab(b3), rng)
    out["row2a_beyond_semifractal_b3"] = perm_test(loo_residuals(P(b3), feat(b3, "F_frac")), lab(b3), rng)
    out["row2b_beyond_scattering_pca10_b3"] = perm_test(loo_pca_residuals(feat(b3, "F_scat"), P(b3), 10), lab(b3), rng)
    out["row3_beyond_spectrum_heldout"] = perm_test(loo_residuals(P(b2b), feat(b2b, "F_spec")), lab(b2b), rng)
    out["row3_beyond_spectrum_b3_48_not_in_table"] = perm_test(loo_residuals(P(b3), feat(b3, "F_spec")), lab(b3), rng)
    out["row3_beyond_spectrum_all96_not_in_table"] = perm_test(loo_residuals(P(deg), feat(deg, "F_spec")), lab(deg), rng)
    out["row5_km_anchored_primary"] = perm_test(P(km_p, "Panch"), lab(km_p), rng)
    out["row5_km_anchored_heldout"] = perm_test(P(km_h, "Panch"), lab(km_h), rng)
    out["row6_km_anchored_beyond_spectrum"] = perm_test(loo_residuals(P(km_p, "Panch"), feat(km_p, "F_spec")), lab(km_p), rng)
    out["row6_km_raw_beyond_spectrum"] = perm_test(loo_residuals(P(km_p), feat(km_p, "F_spec")), lab(km_p), rng)
    out["all96_raw"] = perm_test(P(deg), lab(deg), rng)
    out["all80_km_anchored"] = perm_test(P(km, "Panch"), lab(km), rng)
    return out


def fine_ordering(km: list[dict], steps: list[int]) -> list[tuple[str, float]]:
    per: dict[str, list[float]] = {}
    for r in km:
        if r["period"] == "III":
            per.setdefault(r["region"], []).append(float(np.mean(r["Panch"][steps])))
    order = sorted(((reg, round(float(np.median(v)), 4)) for reg, v in per.items()), key=lambda t: -t[1])
    return order


def check_a4(deg: list[dict], km: list[dict], rng: np.random.Generator) -> dict:
    out = {}
    b2b = [r for r in deg if r["period"] == "II"]
    b3 = [r for r in deg if r["period"] == "III"]
    km_p = [r for r in km if r["period"] == "III"]
    km_h = [r for r in km if r["period"] == "II"]
    lab = lambda rs: np.asarray([r["region"] for r in rs])  # noqa: E731

    def regional(rs, key):
        v = []
        for r in rs:
            st = STEP_5 if r["region"] in TROPICAL else STEPS_4_5
            v.append([float(np.mean(r[key][st]))])
        return np.asarray(v)

    for name, rs, key in [("heldout_raw", b2b, "P"), ("b3_raw", b3, "P"),
                          ("km_anchored_primary", km_p, "Panch"), ("km_anchored_heldout", km_h, "Panch")]:
        out[f"{name}_steps45"] = perm_test(np.asarray([r[key][STEPS_4_5] for r in rs]), lab(rs), rng)
        out[f"{name}_step5_only"] = perm_test(np.asarray([r[key][STEP_5] for r in rs]), lab(rs), rng)
        out[f"{name}_steps345_provenance_check"] = perm_test(np.asarray([r[key][[2, 3, 4]] for r in rs]), lab(rs), rng)
        out[f"{name}_regionally_resolved"] = perm_test(regional(rs, key), lab(rs), rng)
    out["tropical_regions_step5_only"] = sorted(TROPICAL)
    out["note_row7"] = ("Manuscript row 7 (+0.61, p=0.002) is not reproduced on steps 4-5 of the 32 held-out "
                        "windows (see heldout_raw_steps45); +0.612 is obtained on steps 3-5 (heldout_raw_steps345_provenance_check).")
    return out


def variance_components(deg: list[dict], steps: list[int], key: str = "P") -> dict:
    """SS decomposition per step on the 12 regions x 8 windows (degree boxes):
    region | season within region | year within region+season | residual.
    ICC(1) from the one-way region model."""
    regs = sorted({r["region"] for r in deg})
    out = {}
    for s in steps:
        y = np.asarray([r[key][s] for r in deg])
        reg = np.asarray([r["region"] for r in deg])
        sea = np.asarray([r["season"] for r in deg])
        yr = np.asarray([r["year"] for r in deg])
        gm = y.mean()
        ss_tot = float(((y - gm) ** 2).sum())
        # region
        m_reg = {g: y[reg == g].mean() for g in regs}
        ss_reg = float(sum(((reg == g).sum()) * (m_reg[g] - gm) ** 2 for g in regs))
        # season within region
        ss_sea = 0.0
        for g in regs:
            for se in ("JFM", "JAS"):
                m = (reg == g) & (sea == se)
                if m.any():
                    ss_sea += m.sum() * (y[m].mean() - m_reg[g]) ** 2
        # year within region+season
        ss_yr = 0.0
        for g in regs:
            for se in ("JFM", "JAS"):
                m0 = (reg == g) & (sea == se)
                if not m0.any():
                    continue
                m_gs = y[m0].mean()
                for yy in np.unique(yr[m0]):
                    m = m0 & (yr == yy)
                    ss_yr += m.sum() * (y[m].mean() - m_gs) ** 2
        ss_res = ss_tot - ss_reg - ss_sea - ss_yr
        # one-way ICC(1)
        n_i = np.asarray([(reg == g).sum() for g in regs])
        k = len(regs)
        n = len(y)
        msb = ss_reg / (k - 1)
        msw = (ss_tot - ss_reg) / (n - k)
        n0 = (n - (n_i ** 2).sum() / n) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        out[f"step{s + 1}"] = {
            "share_region": round(ss_reg / ss_tot, 3), "share_season_in_region": round(ss_sea / ss_tot, 3),
            "share_year_in_region_season_(=residual,_one_window_per_cell)": round(ss_yr / ss_tot, 3), "share_residual_check": round(ss_res / ss_tot, 3),
            "icc1_region": round(float(icc), 3), "pass_icc_ge_0.5": bool(icc >= 0.5),
        }
    return out


def check_a5(deg: list[dict], km: list[dict], rng: np.random.Generator) -> dict:
    out = {}
    for name, rs, key in [("degree96_raw", deg, "P"), ("degree96_raw_steps25", deg, "P"),
                          ("km80_anchored", km, "Panch"), ("km80_anchored_steps25", km, "Panch")]:
        steps = STEPS_2_5 if name.endswith("steps25") else STEPS_ALL
        x = np.asarray([r[key][steps] for r in rs])
        lab = np.asarray([r["region"] for r in rs])
        block = {"unstratified": perm_test(x, lab, rng)}
        for strat in ("season", "year", "period"):
            st = np.asarray([r[strat] for r in rs])
            block[f"{strat}_stratified_all_pairs"] = perm_test(x, lab, rng, strata=st)
            block[f"{strat}_stratified_same_{strat}_pairs"] = perm_test(x, lab, rng, strata=st, pair_mask=pair_mask_from(st, True))
            block[f"{strat}_stratified_cross_{strat}_pairs"] = perm_test(x, lab, rng, strata=st, pair_mask=pair_mask_from(st, False))
        block["region_level_median_test"] = region_median_test(x, lab, rng)
        block["loo_nearest_centroid_accuracy"] = loo_centroid_accuracy(x, lab)
        out[name] = block
    out["variance_components_degree96_raw"] = variance_components(deg, STEPS_ALL, "P")
    out["variance_components_degree96_anchored"] = variance_components(deg, STEPS_ALL, "Panch")
    # km set is unbalanced (R1-R4: 4 windows); reported for completeness
    out["variance_components_km80_anchored"] = variance_components(km, STEPS_ALL, "Panch")
    return out


def _domain_centre(name: str) -> tuple[float, float]:
    if name in REGION_CENTRE:
        return REGION_CENTRE[name]
    code = name.split("_", 1)[1]            # 55N100E -> lat edge 55N, lon west edge 100E
    i = code.index("N") if "N" in code else code.index("S")
    lat = float(code[:i]) * (1 if code[i] == "N" else -1)
    lon_s = code[i + 1:]
    lon = float(lon_s[:-1]) * (1 if lon_s[-1] == "E" else -1)
    return lat - 10.0, lon + 20.0


def check_a6(rng: np.random.Generator, deg: list[dict]) -> dict:
    s = json.loads((RES / "experiment_B13_free_running" / "summary.json").read_text(encoding="utf-8"))
    dv = s["domain_values"]
    names = sorted(dv)
    x = np.asarray([dv[n]["P_era5"] for n in names])
    y = np.asarray([dv[n]["P_model"] for n in names])
    cen = np.asarray([_domain_centre(n) for n in names])
    rho_all = float(spearmanr(x, y).statistic)
    out = {"n": len(names), "rho_full": round(rho_all, 4), "archived_H13a": s["H13a"]["rho"]}
    for bname, blk in [("lon_sectors_60deg", ((cen[:, 1] + 180.0) // 60.0).astype(int)),
                       ("lat_rows", np.round(cen[:, 0] / 20.0).astype(int))]:
        ids = np.unique(blk)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(ids, size=len(ids), replace=True)
            idx = np.concatenate([np.where(blk == b)[0] for b in pick])
            if len(np.unique(x[idx])) < 3:
                continue
            boots.append(spearmanr(x[idx], y[idx]).statistic)
        boots = np.asarray(boots)
        jack = [float(spearmanr(x[blk != b], y[blk != b]).statistic) for b in ids]
        out[bname] = {"n_blocks": int(len(ids)),
                      "block_sizes": [int((blk == b).sum()) for b in ids],
                      "boot_ci90": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)],
                      "boot_median": round(float(np.median(boots)), 3),
                      "jackknife_leave_block_out": [round(j, 3) for j in jack],
                      "pass_lower_gt_0.4": bool(np.percentile(boots, 5) > 0.4)}
    # ceiling: ERA5 across periods on the degree boxes, mean of steps 4-5
    per = {}
    for r in deg:
        per.setdefault(r["region"], {}).setdefault(r["period"], []).append(float(np.mean(r["P"][STEPS_4_5])))
    regs = sorted(per)
    early = np.asarray([np.median(per[g]["I" if "I" in per[g] else "II"]) for g in regs])
    late = np.asarray([np.median(per[g]["III"]) for g in regs])
    regs8 = [g for g in regs if "II" in per[g]]
    out["ceiling_recomputed"] = {
        "rho_12_regions_early_vs_III": round(float(spearmanr(early, late).statistic), 4),
        "rho_8_regions_II_vs_III": round(float(spearmanr(
            [np.median(per[g]["II"]) for g in regs8], [np.median(per[g]["III"]) for g in regs8]).statistic), 4),
        "archived_ceiling_P": s["era5_across_period_ceiling"]["P"],
        "jackknife_12_leave_one_region": [round(float(spearmanr(np.delete(early, i), np.delete(late, i)).statistic), 3) for i in range(len(regs))],
    }
    return out


def check_a7(deg: list[dict], km: list[dict], steps: list[int], rng: np.random.Generator) -> dict:
    """Bootstrap interval for Δ of the residual rows.  Residualisation is done
    once on the full sample (fixed preprocessing); windows are then resampled
    with replacement within each region and pairs made of the same source
    window are excluded from the within/between medians (otherwise duplicate
    pairs at distance 0 bias Δ upward)."""
    b2b = [r for r in deg if r["period"] == "II"]
    b3 = [r for r in deg if r["period"] == "III"]
    km_p = [r for r in km if r["period"] == "III"]

    def boot_delta(rs, key, feature, n_pca=None):
        lab = np.asarray([r["region"] for r in rs])
        y = np.asarray([r[key][steps] for r in rs])
        x = feat(rs, feature)
        res = loo_residuals(y, x) if n_pca is None else loo_pca_residuals(x, y, n_pca)
        dist = pdist(zscore(res))
        regs = np.unique(lab)
        idx_by = {g: np.where(lab == g)[0] for g in regs}
        obs = delta(dist, lab)[0]
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([rng.choice(idx_by[g], size=len(idx_by[g]), replace=True) for g in regs])
            d_b = dist[np.ix_(idx, idx)]
            distinct = idx[:, None] != idx[None, :]
            boots.append(delta(d_b, lab[idx], distinct)[0])
        boots = np.asarray(boots)
        ci = [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)]
        return {"diff_obs": round(obs, 4), "boot_ci90": ci, "boot_median": round(float(np.median(boots)), 3),
                "ci_excludes_0": bool(ci[0] > 0),
                "region_median_test": region_median_test(res, lab, rng)}

    return {
        "row2a_semifractal_b3": boot_delta(b3, "P", "F_frac"),
        "row2b_scattering_pca10_b3": boot_delta(b3, "P", "F_scat", 10),
        "row3_spectrum_heldout": boot_delta(b2b, "P", "F_spec"),
        "row6_km_anchored_spectrum": boot_delta(km_p, "Panch", "F_spec"),
    }


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--quick", action="store_true", help="99 permutations / bootstraps")
    args = parser.parse_args()
    global N_PERM, N_BOOT
    if args.quick:
        N_PERM = N_BOOT = 99
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    deg = load_degree()
    km = load_km()
    res: dict = {"protocol": "docs/PROTOCOL_PHASE32_REVIEW_CHECKS.md", "seed": SEED,
                 "n_perm": N_PERM, "n_boot": N_BOOT,
                 "n_degree_windows": len(deg), "n_km_windows": len(km)}

    archived = {"row1_heldout_raw": 1.1191, "row2a_beyond_semifractal_b3": 0.6489,
                "row2b_beyond_scattering_pca10_b3": 0.2979, "row3_beyond_spectrum_heldout": 0.3368,
                "row5_km_anchored_primary": 1.2559, "row5_km_anchored_heldout": 1.5636,
                "row6_km_anchored_beyond_spectrum": 0.2179, "row6_km_raw_beyond_spectrum": 0.3468}
    print("A0 sanity: full 5 steps", flush=True)
    full = rows_table2(deg, km, STEPS_ALL, rng)
    res["A0_sanity_full_steps"] = {k: {**v, "archived_diff": archived.get(k),
                                       "reproduced": (abs(v["diff"] - archived[k]) < 0.005) if k in archived else None}
                                   for k, v in full.items()}
    res["A0_sanity_full_steps"]["fine_ordering_steps123_km_anchored"] = fine_ordering(km, [0, 1, 2])
    res["A0_sanity_full_steps"]["note_row2a"] = (
        "Archived +0.649 (e1_semifractal.json) is not reproduced by LOO linear residualisation on the 11 F_frac "
        "features of the Phase-3 JSONs (+0.332, p=0.02); the producing script is not in the repository history. "
        "Nearest variants: PCA5 of F_frac +0.618, slope+break_km +0.764.")

    print("A3 steps 2-5", flush=True)
    a3 = rows_table2(deg, km, STEPS_2_5, rng)
    a3["fine_ordering_steps23_km_anchored"] = fine_ordering(km, [1, 2])
    a3["fine_ordering_steps2_only_km_anchored"] = fine_ordering(km, [1])
    head = ["row1_heldout_raw", "row5_km_anchored_primary", "row5_km_anchored_heldout"]
    signs = ["row2a_beyond_semifractal_b3", "row2b_beyond_scattering_pca10_b3",
             "row3_beyond_spectrum_heldout", "row6_km_anchored_beyond_spectrum"]
    a3["H32-3_pass"] = bool(all(a3[k]["diff"] >= 0.8 and a3[k]["p"] <= 0.01 for k in head)
                            and all(a3[k]["diff"] > 0 for k in signs))
    loso = {}
    for drop in STEPS_ALL:
        sub = [k for k in STEPS_ALL if k != drop]
        rr = rows_table2(deg, km, sub, rng)
        loso[f"drop_step{drop + 1}"] = {k: rr[k]["diff"] for k in head}
    a3["leave_one_step_out_headline_rows"] = loso
    res["A3_steps_2_5"] = a3

    print("A4 resolution", flush=True)
    a4 = check_a4(deg, km, rng)
    a4["H32-4_pass"] = bool(all(a4[k]["p"] <= 0.01 and a4[k]["diff"] > 0 for k in
                                ("heldout_raw_step5_only", "heldout_raw_regionally_resolved")))
    res["A4_effective_resolution"] = a4

    print("A5 stratified / variance components", flush=True)
    a5 = check_a5(deg, km, rng)
    ok = []
    for name in ("degree96_raw", "km80_anchored"):
        for strat in ("season", "year", "period"):
            t = a5[name][f"{strat}_stratified_all_pairs"]
            ok.append(t["diff"] > 0 and t["p"] <= 0.01)
    a5["H32-5a_pass"] = bool(all(ok))
    a5["H32-5b_pass_steps2_5"] = bool(all(a5["variance_components_degree96_raw"][f"step{s + 1}"]["icc1_region"] >= 0.5
                                          for s in STEPS_2_5))
    res["A5_stratified_and_variance"] = a5

    print("A6 spatial blocks", flush=True)
    a6 = check_a6(rng, deg)
    a6["H32-6_pass"] = bool(a6["lon_sectors_60deg"]["pass_lower_gt_0.4"] and a6["lat_rows"]["pass_lower_gt_0.4"])
    res["A6_spatial_block_uncertainty"] = a6

    print("A7 bootstrap intervals (full steps and steps 2-5)", flush=True)
    res["A7_borderline_rows_full_steps"] = check_a7(deg, km, STEPS_ALL, rng)
    res["A7_borderline_rows_steps_2_5"] = check_a7(deg, km, STEPS_2_5, rng)

    (args.out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    # compact report
    lines = ["# B32 checks A0, A3-A7 (auto-generated)", ""]
    for k, v in res["A0_sanity_full_steps"].items():
        if isinstance(v, dict) and "diff" in v:
            lines.append(f"A0 {k}: diff={v['diff']:+.3f} p={v['p']:.3f} archived={v['archived_diff']} reproduced={v['reproduced']}")
    for k, v in a3.items():
        if isinstance(v, dict) and "diff" in v:
            lines.append(f"A3 {k}: diff={v['diff']:+.3f} p={v['p']:.3f} (full: {full[k]['diff']:+.3f} p={full[k]['p']:.3f})")
    lines.append(f"A3 H32-3 pass={a3['H32-3_pass']}; fine ordering steps 2-3: {a3['fine_ordering_steps23_km_anchored']}")
    for k, v in a3["leave_one_step_out_headline_rows"].items():
        lines.append(f"A3 leave-one-step-out {k}: " + ", ".join(f"{kk}={vv:+.3f}" for kk, vv in v.items()))
    for k, v in a4.items():
        if isinstance(v, dict) and "diff" in v:
            lines.append(f"A4 {k}: diff={v['diff']:+.3f} p={v['p']:.3f}")
    lines.append(f"A4 H32-4 pass={a4['H32-4_pass']}")
    for name in ("degree96_raw", "degree96_raw_steps25", "km80_anchored", "km80_anchored_steps25"):
        for k, v in a5[name].items():
            if isinstance(v, dict) and "diff" in v:
                lines.append(f"A5 {name} {k}: diff={v['diff']:+.3f} p={v['p']:.3f}")
        lines.append(f"A5 {name} region-median test p={a5[name]['region_level_median_test']['p']:.3f}; "
                     f"LOO centroid acc={a5[name]['loo_nearest_centroid_accuracy']}")
    for key in ("variance_components_degree96_raw", "variance_components_degree96_anchored"):
        for st, v in a5[key].items():
            lines.append(f"A5 {key} {st}: region={v['share_region']:.2f} season={v['share_season_in_region']:.2f} "
                         f"year(=resid)={v['share_year_in_region_season_(=residual,_one_window_per_cell)']:.2f} ICC1={v['icc1_region']:.2f}")
    lines.append(f"A5 H32-5a pass={a5['H32-5a_pass']}  H32-5b pass(steps 2-5)={a5['H32-5b_pass_steps2_5']}")
    lines.append(f"A6 rho_full={a6['rho_full']} lon-blocks CI90={a6['lon_sectors_60deg']['boot_ci90']} "
                 f"lat-rows CI90={a6['lat_rows']['boot_ci90']} jack(lon)={a6['lon_sectors_60deg']['jackknife_leave_block_out']} "
                 f"pass={a6['H32-6_pass']}; ceiling recomputed={a6['ceiling_recomputed']}")
    for tag in ("A7_borderline_rows_full_steps", "A7_borderline_rows_steps_2_5"):
        for k, v in res[tag].items():
            lines.append(f"{tag} {k}: diff={v['diff_obs']:+.3f} CI90={v['boot_ci90']} excl0={v['ci_excludes_0']} "
                         f"region-median p={v['region_median_test']['p']:.3f}")
    (args.out / "report_A3_A7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
