#!/usr/bin/env python3
"""Descriptive geography of the global P map (paper 2, sections 3 and 6).

DESCRIPTIVE ONLY - no protocol, no decision rule, no p-values. Everything
here is computed from the committed Phase-20 Arm-B tiles
(results/experiment_B20_armB_global_map/tiles_{JFM,JAS}2023.json), the
frozen Arm-B covariates and the AUDIT-2b split-half shards; nothing is
re-estimated from raw fields.

Outputs (results/describe_B20_geography/):
  summary.json            zonal profile, hemispheric mirror, land/ocean by
                          belt, season contrast, extremes, class table,
                          programme-region means, residual zonal profile
  provinces.json / .csv   per-tile table: lat, lon, P (mean/JFM/JAS),
                          land fraction, eke, cape, class, orography-adjacent
  fig_zonal_profiles.png  paper-2 figure: P vs latitude (all / ocean / land),
                          NH vs SH mirror, JFM vs JAS
  fig_regionalization.png paper-2 figure: bivariate typology map
                          (organizer x coupling tercile) + programme boxes
  fig_residual_map.png    paper-2 figure: AUDIT-4 residual (P minus
                          covariates + intermittency prediction, regression
                          fitted on the opposite time-parity half), mean of
                          the two seasons - recomputed with the AUDIT-4 code
                          path so the figure has a producer

Typology (descriptive, global terciles over the 902 retained tiles):
  organizer: 'storm' if synoptic EKE >= upper tercile, 'conv' if CAPE
             climatology >= upper tercile (a tile meeting both is 'storm'
             if its EKE rank exceeds its CAPE rank, else 'conv'), else 'none'
  coupling:  'low' / 'mid' / 'high' terciles of the two-season mean P
  orog_adj:  tile shares an edge or corner with an excluded (>1200 m) tile
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch, Rectangle

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B20_armB_global_map import (  # noqa: E402
    ALL_COV, OROG_EXCL_M, SEASONS, OUT as ARMB, tile_covariates, tile_grid,
)
from experiment_B18_equal_km_regions import REGIONS, REGIONS_ALL  # noqa: E402
from experiment_A4_residual_structure import (  # noqa: E402
    A2B, HALVES, INT_COLS, ols_predict, z,
)

OUT = _HERE / "results" / "describe_B20_geography"
INV = _HERE.parent / "data" / "b4cov" / "era5_invariants_global.nc"
KM_PER_DEG = 6371.0 * np.pi / 180.0

REGION_RU = {
    "R1_WPWP": "Тёплый бассейн зап. Пацифики", "R2_NATL": "Сев. Атлантика",
    "R3_AMAZ": "Амазония", "R4_CASIA": "Центральная Азия",
    "R5_SPCZ": "Юж.-Тихоокеанская зона конвергенции", "R6_SATL": "Юж. Атлантика",
    "R7_CONGO": "Бассейн Конго", "R8_AUS": "Австралия",
    "R9_NPAC": "Сев. Пацифика", "R10_INDO": "Тропическая часть Индийского океана",
    "R11_EURO": "Европа", "R12_SAM": "Юж. Америка (субтропики)",
}


# Gazetteer: every place named in paper 2 (Table 1 and the text) -> nominal
# tile centre (lat, lon 0-360).  Resolved to the nearest retained tile and
# written to summary.json["named_cells"], so each geographical name in the
# manuscript is traceable to a tile id and its value.
NAMED_PLACES = [
    ("Южнее Новой Зеландии", -45, 166.5), ("Залив Аляска", 51, 231.4),
    ("Море Лабрадор", 57, 296.1), ("Гудзонов залив", 57, 273.0),
    ("Норвежское море", 57, 6.0), ("Побережье Калифорнии", 33, 241.3),
    ("Побережье центрального Чили", -33, 287.2), ("Канарское побережье", 27, 342.0),
    ("Португальское побережье", 33, 348.5), ("Запад Австралии", -21, 112.1),
    ("Акватория к югу от Явы", -9, 112.5),
    ("Тихоокеанское побережье Центральной Америки", 15, 265.1),
    ("Запад Амазонии (1)", -3, 287.4), ("Запад Амазонии (2)", -3, 281.1),
    ("Бассейн Конго", -3, 15.8), ("Калимантан", -3, 116.8),
    ("Южный Китай (1)", 27, 118.8), ("Южный Китай (2)", 27, 111.6),
    ("Побережье Красного моря", 21, 37.4), ("Аравийский полуостров", 15, 42.5),
    ("Гран-Чако", -15, 297.8), ("Предгорья Анд", -21, 288.7),
    ("Окраина Мексиканского нагорья", 27, 248.4), ("Иранское нагорье", 33, 57.4),
    ("Запад Малой Азии", 39, 28.6), ("Патагония", -39, 290.5),
    ("Бискайский залив", 45, 355.5), ("Пампа", -33, 302.6),
    ("Юг Австралии", -33, 141.7), ("Новая Гвинея", -9, 144.6),
    ("Гангская равнина", 27, 82.8), ("Бенгальский залив", 21, 91.7),
    ("Север внутренней Евразии (1)", 57, 63.9), ("Север внутренней Евразии (2)", 57, 110.3),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    seasons = {}
    for s in SEASONS:
        d = json.loads((ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))
        seasons[s] = {r["tid"]: r for r in d["tiles"]}
    common = sorted(set(seasons["JFM"]) & set(seasons["JAS"]))
    cov = tile_covariates([by_tid[t] for t in common])
    excl = {t for t in common if cov[t]["orog_mean"] > OROG_EXCL_M}
    keep = [t for t in common if t not in excl
            and all(np.isfinite(seasons[s][t]["P_fine_anch_mean"]) for s in SEASONS)]

    # ---- per-tile table ---------------------------------------------------
    lat = np.array([by_tid[t]["lat_c"] for t in keep])
    lon = np.array([by_tid[t]["lon_c"] for t in keep])
    pj = np.array([seasons["JFM"][t]["P_fine_anch_mean"] for t in keep])
    pa = np.array([seasons["JAS"][t]["P_fine_anch_mean"] for t in keep])
    pm = 0.5 * (pj + pa)
    land = np.array([cov[t]["land_frac"] for t in keep])
    eke = np.array([np.mean([seasons[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    cape = np.array([cov[t]["cape_mean"] for t in keep])
    is_ocean = land < 0.5

    # orography-adjacent flag: any excluded tile among the 8 neighbours
    excl_rc = {(by_tid[t]["row"], by_tid[t]["col"]) for t in excl}
    n_lon_row = {t["row"]: t["n_lon_row"] for t in grid}

    def adjacent(t):
        r, c = by_tid[t]["row"], by_tid[t]["col"]
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr not in n_lon_row:
                continue
            # neighbouring rows have different column counts: map by longitude
            lon_c = by_tid[t]["lon_c"]
            n = n_lon_row[rr]
            cc = int(np.floor(lon_c / (360.0 / n)))
            for dc in (-1, 0, 1):
                if (rr, (cc + dc) % n) in excl_rc:
                    return True
        return False

    orog_adj = np.array([adjacent(t) for t in keep])

    # ---- typology ---------------------------------------------------------
    q_p = np.quantile(pm, [1 / 3, 2 / 3])
    coupling = np.where(pm >= q_p[1], "high", np.where(pm <= q_p[0], "low", "mid"))
    q_e, q_c = np.quantile(eke, 2 / 3), np.quantile(cape, 2 / 3)
    rank_e = np.argsort(np.argsort(eke)) / len(eke)
    rank_c = np.argsort(np.argsort(cape)) / len(cape)
    organizer = np.full(len(keep), "none", dtype=object)
    for i in range(len(keep)):
        se, sc = eke[i] >= q_e, cape[i] >= q_c
        if se and sc:
            organizer[i] = "storm" if rank_e[i] >= rank_c[i] else "conv"
        elif se:
            organizer[i] = "storm"
        elif sc:
            organizer[i] = "conv"

    # ---- residual (AUDIT-4 code path) ------------------------------------
    sh = {(s, h): {r["tid"]: r for r in json.loads(
        (A2B / f"tiles_{s}2023_{h}.json").read_text(encoding="utf-8"))["tiles"]}
        for s in SEASONS for h in HALVES}
    keep_r = [t for t in keep if all(t in sh[(s, h)] for s in SEASONS for h in HALVES)]
    Xz = {n: z([cov[t][n] for t in keep_r]) for n in ALL_COV if n != "eke_syn"}
    Xz["eke_syn"] = z([np.mean([seasons[s][t]["eke_syn"] for s in SEASONS]) for t in keep_r])
    X_cov = np.column_stack([Xz[n] for n in ALL_COV])

    def P(s, h):
        return np.array([sh[(s, h)][t]["P_anch_mean"] for t in keep_r])

    def INT(s, h):
        return np.column_stack([z([sh[(s, h)][t][f"{k}_anch_mean"] for t in keep_r])
                                for k in INT_COLS])

    resid = {}
    for s in SEASONS:
        rf = {}
        for h, ho in (("odd", "even"), ("even", "odd")):
            Xf = np.column_stack([X_cov, INT(s, ho)])
            rf[h] = P(s, h) - ols_predict(Xf, P(s, ho), Xf)
        resid[s] = 0.5 * (rf["odd"] + rf["even"])
    r_mean = 0.5 * (resid["JFM"] + resid["JAS"])
    resid_by_tid = dict(zip(keep_r, r_mean))

    # ---- descriptive statistics -----------------------------------------
    def band_stats(mask):
        return {"n": int(mask.sum()), "P": float(pm[mask].mean()) if mask.any() else None}

    zonal = []
    for lc in sorted(set(lat)):
        k = lat == lc
        zonal.append({
            "lat_c": float(lc), "n": int(k.sum()), "P": float(pm[k].mean()),
            "P_ocean": float(pm[k & is_ocean].mean()) if (k & is_ocean).any() else None,
            "n_ocean": int((k & is_ocean).sum()),
            "P_land": float(pm[k & ~is_ocean].mean()) if (k & ~is_ocean).any() else None,
            "n_land": int((k & ~is_ocean).sum()),
            "P_JFM": float(pj[k].mean()), "P_JAS": float(pa[k].mean()),
            "residual": float(np.mean([resid_by_tid[t] for t in np.array(keep)[k]
                                       if t in resid_by_tid])),
        })
    mirror = []
    for al in sorted(set(np.abs(lat))):
        kn, ks = lat == al, lat == -al
        mirror.append({"abs_lat": float(al), "P_NH": float(pm[kn].mean()),
                       "P_SH": float(pm[ks].mean()), "n_NH": int(kn.sum()),
                       "n_SH": int(ks.sum()),
                       "P_NH_ocean": float(pm[kn & is_ocean].mean()) if (kn & is_ocean).any() else None,
                       "P_SH_ocean": float(pm[ks & is_ocean].mean()) if (ks & is_ocean).any() else None})
    belts = {"tropics": (0, 15), "subtropics": (15, 35), "midlat": (35, 60)}
    land_ocean = {}
    for name, (a, b) in belts.items():
        k = (np.abs(lat) >= a) & (np.abs(lat) < b)
        land_ocean[name] = {"ocean": band_stats(k & is_ocean), "land": band_stats(k & ~is_ocean)}
    land_ocean["all"] = {"ocean": band_stats(is_ocean), "land": band_stats(~is_ocean)}

    gm = pm.mean()
    ss_tot = float(((pm - gm) ** 2).sum())
    ss_zonal = float(sum(((pm[lat == lc].mean() - gm) ** 2) * (lat == lc).sum()
                         for lc in set(lat)))
    d_season = pj - pa
    season = {
        "median_abs_diff": float(np.median(np.abs(d_season))),
        "q90_abs_diff": float(np.quantile(np.abs(d_season), 0.9)),
        "NH_midlat_JFM_minus_JAS": float(d_season[lat >= 35].mean()),
        "SH_midlat_JFM_minus_JAS": float(d_season[lat <= -35].mean()),
        "tropics_JFM_minus_JAS": float(d_season[np.abs(lat) < 15].mean()),
        "NH_tropical_land_JFM_minus_JAS": float(d_season[(lat > 0) & (lat < 25) & ~is_ocean].mean()),
        "SH_tropical_land_JFM_minus_JAS": float(d_season[(lat < 0) & (lat > -25) & ~is_ocean].mean()),
    }

    def tile_row(i):
        return {"tid": keep[i], "lat": float(lat[i]), "lon": float(lon[i]),
                "P": round(float(pm[i]), 4), "land": round(float(land[i]), 2),
                "eke": round(float(eke[i]), 1), "cape": round(float(cape[i]), 0),
                "organizer": str(organizer[i]), "coupling": str(coupling[i]),
                "orog_adj": bool(orog_adj[i])}

    order = np.argsort(pm)
    o_s = np.argsort(d_season)
    extremes = {"lowest": [tile_row(i) for i in order[:20]],
                "highest": [tile_row(i) for i in order[-20:][::-1]],
                "JFM_gt_JAS": [dict(tile_row(i), d=round(float(d_season[i]), 4)) for i in o_s[-12:][::-1]],
                "JAS_gt_JFM": [dict(tile_row(i), d=round(float(d_season[i]), 4)) for i in o_s[:12]]}

    # class table: organizer x coupling
    table = {}
    for org in ("storm", "none", "conv"):
        table[org] = {}
        for cp in ("high", "mid", "low"):
            k = (organizer == org) & (coupling == cp)
            table[org][cp] = {"n": int(k.sum()), "P": float(pm[k].mean()) if k.any() else None,
                              "ocean_share": float(is_ocean[k].mean()) if k.any() else None,
                              "orog_adj_share": float(orog_adj[k].mean()) if k.any() else None}
    # organizer summaries
    org_summary = {org: {"n": int((organizer == org).sum()), "P": float(pm[organizer == org].mean()),
                         "P_sd": float(pm[organizer == org].std())}
                   for org in ("storm", "none", "conv")}
    orog_summary = {"n_adjacent": int(orog_adj.sum()),
                    "P_adjacent": float(pm[orog_adj].mean()),
                    "P_not_adjacent": float(pm[~orog_adj].mean()),
                    "share_of_lowest_decile_adjacent": float(orog_adj[order[:len(keep) // 10]].mean())}

    # programme regions: tiles whose centre falls inside the box
    regions = {}
    for r in REGIONS:
        n, w, so, e = REGIONS_ALL[r]
        lon_r = ((lon + 180.0) % 360.0) - 180.0
        w2, e2 = (((w + 180.0) % 360.0) - 180.0), (((e + 180.0) % 360.0) - 180.0)
        if w2 <= e2:
            kl = (lon_r >= w2) & (lon_r <= e2)
        else:
            kl = (lon_r >= w2) | (lon_r <= e2)
        k = kl & (lat >= so) & (lat <= n)
        orgs, cnt = np.unique(organizer[k], return_counts=True) if k.any() else ([], [])
        regions[r] = {"name_ru": REGION_RU[r], "n_tiles": int(k.sum()),
                      "P": float(pm[k].mean()) if k.any() else None,
                      "P_min": float(pm[k].min()) if k.any() else None,
                      "P_max": float(pm[k].max()) if k.any() else None,
                      "dominant_organizer": str(orgs[np.argmax(cnt)]) if k.any() else None,
                      "coupling_high_share": float((coupling[k] == "high").mean()) if k.any() else None,
                      "coupling_low_share": float((coupling[k] == "low").mean()) if k.any() else None}

    named = []
    for name, la, lo in NAMED_PLACES:
        dlon = np.abs(((lon - lo + 180.0) % 360.0) - 180.0) * np.cos(np.deg2rad(la))
        i = int(np.argmin((lat - la) ** 2 + dlon ** 2))
        named.append(dict(tile_row(i), name_ru=name))

    summary = {
        "note": "DESCRIPTIVE - no protocol, no decision rule",
        "named_cells": named,
        "n_tiles": len(keep), "n_excluded_orog": len(excl),
        "global": {"mean": float(gm), "sd": float(pm.std()), "min": float(pm.min()),
                   "max": float(pm.max()), "q_terciles": [float(x) for x in q_p]},
        "zonal_share_of_variance": ss_zonal / ss_tot,
        "zonal_profile": zonal, "hemispheric_mirror": mirror,
        "land_ocean": land_ocean, "season": season, "extremes": extremes,
        "typology_terciles": {"eke_q67": float(q_e), "cape_q67": float(q_c)},
        "typology_table": table, "organizer_summary": org_summary,
        "orography_adjacent": orog_summary, "programme_regions": regions,
        "residual": {"n": len(keep_r),
                     "zonal_profile_10deg": [
                         {"lat_band": f"{int(lo)}..{int(lo) + 10}",
                          "mean_residual": float(np.mean([resid_by_tid[t] for t in keep_r
                                                          if lo <= by_tid[t]["lat_c"] < lo + 10]))}
                         for lo in np.arange(-60, 60, 10)]},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    # the same per-tile table as JSON: results/*.csv are git-ignored, *.json are tracked
    (OUT / "provinces.json").write_text(json.dumps([
        {"tid": t, "lat": float(lat[i]), "lon": float(lon[i]), "P_mean": round(float(pm[i]), 5),
         "P_JFM": round(float(pj[i]), 5), "P_JAS": round(float(pa[i]), 5),
         "land_frac": round(float(land[i]), 3), "eke_syn": round(float(eke[i]), 2),
         "cape": round(float(cape[i]), 1), "organizer": str(organizer[i]),
         "coupling": str(coupling[i]), "orog_adjacent": bool(orog_adj[i]),
         "residual": (round(float(resid_by_tid[t]), 5) if t in resid_by_tid else None)}
        for i, t in enumerate(keep)], ensure_ascii=False), encoding="utf-8")
    with (OUT / "provinces.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tid", "lat", "lon", "P_mean", "P_JFM", "P_JAS", "land_frac",
                    "eke_syn", "cape", "organizer", "coupling", "orog_adjacent", "residual"])
        for i, t in enumerate(keep):
            w.writerow([t, lat[i], lon[i], round(pm[i], 5), round(pj[i], 5), round(pa[i], 5),
                        round(land[i], 3), round(eke[i], 2), round(cape[i], 1),
                        organizer[i], coupling[i], int(orog_adj[i]),
                        round(resid_by_tid[t], 5) if t in resid_by_tid else ""])

    # ---- figures ----------------------------------------------------------
    plt.rcParams.update({"font.family": ["Times New Roman",
                                         "DejaVu Serif"], "font.size": 10})

    # fig 1: zonal profiles
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    lc = np.array([b["lat_c"] for b in zonal])
    ax.plot(lc, [b["P"] for b in zonal], "k-o", ms=3, label="все ячейки")
    ax.plot(lc, [b["P_ocean"] if b["P_ocean"] is not None else np.nan for b in zonal],
            "-s", color="tab:blue", ms=3, label="океан")
    ax.plot(lc, [b["P_land"] if b["P_land"] is not None else np.nan for b in zonal],
            "-^", color="tab:brown", ms=3, label="суша")
    ax.set_xlabel("широта, °"); ax.set_ylabel("сопряженность уровней $P$")
    ax.set_title("а) широтный ход, среднее двух сезонов", loc="left")
    ax.set_xticks(np.arange(-60, 61, 20)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    al = np.array([m["abs_lat"] for m in mirror])
    ax.plot(al, [m["P_NH"] for m in mirror], "-o", color="tab:red", ms=3, label="Северное полушарие")
    ax.plot(al, [m["P_SH"] for m in mirror], "-o", color="tab:blue", ms=3, label="Южное полушарие")
    ax.plot(al, [m["P_NH_ocean"] if m["P_NH_ocean"] is not None else np.nan for m in mirror],
            "--", color="tab:red", lw=1, label="СП, только океан")
    ax.plot(al, [m["P_SH_ocean"] if m["P_SH_ocean"] is not None else np.nan for m in mirror],
            "--", color="tab:blue", lw=1, label="ЮП, только океан")
    ax.set_xlabel("модуль широты, °"); ax.set_title("б) полушарная асимметрия", loc="left")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(lc, [b["P_JFM"] for b in zonal], "-o", color="tab:purple", ms=3, label="январь–март")
    ax.plot(lc, [b["P_JAS"] for b in zonal], "-o", color="tab:green", ms=3, label="июль–сентябрь")
    ax.set_xlabel("широта, °"); ax.set_title("в) два сезона 2023 г.", loc="left")
    ax.set_xticks(np.arange(-60, 61, 20)); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for a in axes:
        a.set_ylim(0.20, 0.37)
    fig.tight_layout()
    fig.savefig(OUT / "fig_zonal_profiles.png", dpi=300)
    plt.close(fig)

    # map helpers
    inv = xr.open_dataset(INV)
    lsm = inv["lsm"].squeeze().values
    glat = inv["latitude"].values
    glon = inv["longitude"].values
    shift = np.argmin(np.abs(glon - 180.0))
    lsm_r = np.roll(lsm, -shift, axis=1)
    glon_r = np.sort(((glon + 180.0) % 360.0) - 180.0)
    inv.close()

    def tile_spans(t):
        lo0 = ((t["lon0"] + 180.0) % 360.0) - 180.0
        w = t["dlon"]
        if lo0 + w > 180.0:
            return [(lo0, 180.0 - lo0), (-180.0, lo0 + w - 180.0)]
        return [(lo0, w)]

    def finish_map(ax, title):
        ax.contour(glon_r, glat, lsm_r, levels=[0.5], colors="k", linewidths=0.5)
        for tid in excl:
            t = by_tid[tid]
            for x0, wid in tile_spans(t):
                ax.add_patch(Rectangle((x0, t["lat0"]), wid, 6.0, facecolor="0.85",
                                       edgecolor="none"))
        for r in REGIONS:
            n, w, so, e = REGIONS_ALL[r]
            ax.add_patch(Rectangle((w, so), e - w, n - so, fill=False,
                                   edgecolor="k", lw=0.8, ls="--"))
        ax.set_xlim(-180, 180); ax.set_ylim(-62, 62)
        ax.set_xticks(np.arange(-180, 181, 60)); ax.set_yticks(np.arange(-60, 61, 30))
        ax.set_xlabel("долгота, °"); ax.set_ylabel("широта, °")
        ax.set_title(title, loc="left")

    # fig 2: bivariate typology map
    hue = {"storm": (0.12, 0.35, 0.75), "conv": (0.80, 0.20, 0.15), "none": (0.45, 0.45, 0.45)}
    light = {"high": 1.0, "mid": 0.62, "low": 0.28}

    def colour(org, cp):
        base = np.array(hue[org]); f = light[cp]
        # blend towards white for low coupling of the organizer hue: dark = high P
        return tuple(1.0 - (1.0 - base) * f)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    for i, t in enumerate(keep):
        tt = by_tid[t]
        fc = colour(organizer[i], coupling[i])
        for x0, wid in tile_spans(tt):
            ax.add_patch(Rectangle((x0, tt["lat0"]), wid, 6.0, facecolor=fc,
                                   edgecolor="none"))
            if orog_adj[i]:
                ax.add_patch(Rectangle((x0, tt["lat0"]), wid, 6.0, facecolor="none",
                                       edgecolor="k", hatch="////", lw=0))
    finish_map(ax, "")
    handles = []
    for org, name in (("storm", "штормовой путь"),
                      ("conv", "конвективный режим"),
                      ("none", "без ведущего процесса")):
        for cp, cpn in (("high", "высокая"), ("mid", "средняя"), ("low", "низкая")):
            handles.append(Patch(facecolor=colour(org, cp), edgecolor="0.3",
                                 label=f"{name}: сопряженность {cpn}"))
    handles.append(Patch(facecolor="white", edgecolor="k", hatch="////",
                         label="ячейка граничит с исключенной"))
    handles.append(Patch(facecolor="0.85", edgecolor="none", label="исключено (высота более 1200 м)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regionalization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # fig 3: residual map
    fig, ax = plt.subplots(figsize=(13, 5.6))
    cm = plt.get_cmap("RdBu_r"); vmax = 0.08
    for t in keep_r:
        tt = by_tid[t]; v = resid_by_tid[t]
        fc = cm(0.5 + 0.5 * np.clip(v / vmax, -1, 1))
        for x0, wid in tile_spans(tt):
            ax.add_patch(Rectangle((x0, tt["lat0"]), wid, 6.0, facecolor=fc, edgecolor="none"))
    finish_map(ax, "")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.01,
                 label="отклонение $P$ от ожидаемого значения")
    fig.tight_layout()
    fig.savefig(OUT / "fig_residual_map.png", dpi=300)
    plt.close(fig)

    # ---- console digest -----------------------------------------------------
    print(f"tiles {len(keep)} (excluded {len(excl)}); zonal share {ss_zonal / ss_tot:.3f}")
    print("organizer x coupling (n):")
    for org in ("storm", "none", "conv"):
        print(f"  {org:6s}", {cp: table[org][cp]["n"] for cp in ("high", "mid", "low")},
              {cp: round(table[org][cp]["P"], 3) for cp in ("high", "mid", "low") if table[org][cp]["P"]})
    print("orography-adjacent:", orog_summary)
    print("programme regions:")
    for r, v in regions.items():
        print(f"  {r:9s} n={v['n_tiles']:2d} P={v['P']:.3f} [{v['P_min']:.3f}..{v['P_max']:.3f}] "
              f"org={v['dominant_organizer']} high={v['coupling_high_share']:.2f} low={v['coupling_low_share']:.2f}")
    print("season:", {k: round(v, 4) for k, v in season.items()})
    print("residual zonal:", [(b["lat_band"], round(b["mean_residual"], 4)) for b in summary["residual"]["zonal_profile_10deg"]])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
