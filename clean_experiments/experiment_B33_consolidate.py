#!/usr/bin/env python3
"""Phase 32 part 2: article-1 numbers "old (archived pipeline) → new (V12)".

Reads results/<exp>/ and results/<exp>_v12/ for Phases 2, 2b, 3, 18, 13 and the
MERRA-2 variants of experiment_B32_vorticity_mask, and writes
results/experiment_B32_review_checks/table2_compare.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import experiment_B32_review_checks as c

RES = c.RES
OUT = RES / "experiment_B32_review_checks"
REGIONS = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA", "R5_SPCZ", "R6_SATL", "R7_CONGO",
           "R8_AUS", "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]


def _lab(rs):
    return np.asarray([r["region"] for r in rs])


def _P(rs, steps, key="P"):
    return np.asarray([r[key][steps] for r in rs])


def table(suffix: str, rng: np.random.Generator) -> dict:
    t: dict = {"suffix": suffix or "(archived)"}
    deg = c.load_degree(suffix)
    b2b = [r for r in deg if r["period"] == "II"]
    # --- row 1 / row 7 (held-out, raw)
    for name, steps in [("steps1_5", c.STEPS_ALL), ("steps2_5", c.STEPS_2_5), ("steps4_5", c.STEPS_4_5), ("step5", c.STEP_5)]:
        t[f"row1_heldout_raw_{name}"] = c.perm_test(_P(b2b, steps), _lab(b2b), rng)
    # --- stratified / ICC on all 96 (new rows)
    for name, steps in [("steps1_5", c.STEPS_ALL), ("steps2_5", c.STEPS_2_5)]:
        x = _P(deg, steps)
        lab = _lab(deg)
        blk = {"unstratified": c.perm_test(x, lab, rng)}
        for strat in ("season", "year", "period"):
            st = np.asarray([r[strat] for r in deg])
            blk[f"{strat}_stratified"] = c.perm_test(x, lab, rng, strata=st)
            blk[f"cross_{strat}_pairs"] = c.perm_test(x, lab, rng, strata=st, pair_mask=c.pair_mask_from(st, False))
        t[f"all96_raw_{name}"] = blk
    t["variance_components_raw"] = c.variance_components(deg, c.STEPS_ALL, "P")
    # --- geography on steps 4-5 (8 windows per region) and Congo-Amazon
    per: dict[str, list] = {}
    per5: dict[str, list] = {}
    for r in deg:
        per.setdefault(r["region"], []).append(float(np.mean(r["P"][c.STEPS_4_5])))
        per5.setdefault(r["region"], []).append(float(r["P"][4]))
    geo = {reg: round(float(np.median(per[reg])), 4) for reg in REGIONS}
    t["geography_steps4_5_median8"] = dict(sorted(geo.items(), key=lambda kv: -kv[1]))
    t["congo_amazon_step5"] = {
        "congo_median": round(float(np.median(per5["R7_CONGO"])), 3),
        "congo_range": [round(min(per5["R7_CONGO"]), 3), round(max(per5["R7_CONGO"]), 3)],
        "amazon_median": round(float(np.median(per5["R3_AMAZ"])), 3),
        "amazon_range": [round(min(per5["R3_AMAZ"]), 3), round(max(per5["R3_AMAZ"]), 3)],
        "step4_median_diff": round(float(np.median([r["P"][3] for r in deg if r["region"] == "R7_CONGO"])
                                         - np.median([r["P"][3] for r in deg if r["region"] == "R3_AMAZ"])), 3),
        "overlap": bool(min(per5["R7_CONGO"]) <= max(per5["R3_AMAZ"])),
        "tropical_oceanic_step5": {reg: round(float(np.median(per5[reg])), 3) for reg in ("R1_WPWP", "R5_SPCZ", "R10_INDO")},
    }
    # --- ERA5 across-period reference (12 regions, mean steps 4-5)
    pp: dict = {}
    for r in deg:
        pp.setdefault(r["region"], {}).setdefault(r["period"], []).append(float(np.mean(r["P"][c.STEPS_4_5])))
    early = [np.median(pp[g]["I" if "I" in pp[g] else "II"]) for g in REGIONS]
    late = [np.median(pp[g]["III"]) for g in REGIONS]
    t["era5_across_period_reference_12"] = round(float(spearmanr(early, late).statistic), 3)
    # --- km boxes (Phase 18)
    try:
        km = c.load_km(suffix)
        kp = [r for r in km if r["period"] == "III"]
        kh = [r for r in km if r["period"] == "II"]
        for name, steps in [("steps1_5", c.STEPS_ALL), ("steps2_5", c.STEPS_2_5), ("steps4_5", c.STEPS_4_5)]:
            t[f"row5_km_anchored_primary_{name}"] = c.perm_test(_P(kp, steps, "Panch"), _lab(kp), rng)
            t[f"row5_km_anchored_heldout_{name}"] = c.perm_test(_P(kh, steps, "Panch"), _lab(kh), rng)
            t[f"km_raw_primary_{name}"] = c.perm_test(_P(kp, steps), _lab(kp), rng)
        t["fine_ordering_km_anchored_steps2_3"] = c.fine_ordering(km, [1, 2])
        t["fine_ordering_km_anchored_steps1_3_old_definition"] = c.fine_ordering(km, [0, 1, 2])
        s18 = json.loads((RES / ("experiment_B18_equal_km_regions" + suffix) / "summary.json").read_text(encoding="utf-8"))
        t["b18_summary"] = {
            "H18b_anchored_P_km": s18["H18b_anchored_P_km"], "H18d_heldout": s18["H18d_heldout"],
            "H18c_geography": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s18["H18c_geography"].items() if k != "per_region_rho"},
            "H18c_per_region_rho": {k: round(v, 2) for k, v in s18["H18c_geography"]["per_region_rho"].items()},
            "H18e_congo_amazon": s18["H18e_congo_amazon"],
            "C18_4_extent_min_rho": {k: round(min(v["per_region_rho"].values()), 3) for k, v in s18["C18_4_extent_sensitivity"].items() if isinstance(v, dict) and "per_region_rho" in v},
            "P18_3_dynamic_anchored": s18["P18_3_dynamic_scale_arm"]["anchored_P_clustering"],
        }
    except (AssertionError, FileNotFoundError) as exc:
        t["km_boxes"] = f"not available: {exc}"
    # --- MERRA-2 (row 4)
    mer = {}
    for f in sorted((OUT / "merra").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        mer[r["tag"]] = r
    key = "V12" if suffix else "V0"
    if mer and all(key in r for r in mer.values()):
        tags = sorted(mer)
        mk = np.asarray([mer[tg][key] for tg in tags])
        lab = np.asarray([mer[tg]["region"] for tg in tags])
        t["row4_merra_signature"] = c.perm_test(mk, lab, rng)
        t["row4_merra_signature_steps2_3"] = c.perm_test(mk[:, 1:], lab, rng)
        m_geo = {}
        for tg, row in zip(tags, mk):
            m_geo.setdefault(mer[tg]["region"], []).append(float(np.mean(row[1:3])))
        e_geo = {}
        for r in deg:
            if r["period"] == "III":
                e_geo.setdefault(r["region"], []).append(float(np.mean(r["P"][c.STEPS_4_5])))
        x = [np.median(e_geo[g]) for g in REGIONS]
        y = [np.median(m_geo[g]) for g in REGIONS]
        s = spearmanr(x, y)
        t["row4_geography_era5_vs_merra2"] = {"rho": round(float(s.statistic), 3), "p": round(float(s.pvalue), 5), "n": 12}
    # --- free run (row 8)
    try:
        s13 = json.loads((RES / ("experiment_B13_free_running" + suffix) / "summary.json").read_text(encoding="utf-8"))
        dv = s13["domain_values"]
        names = sorted(dv)
        x = np.asarray([dv[n]["P_era5"] for n in names])
        y = np.asarray([dv[n]["P_model"] for n in names])
        cen = np.asarray([c._domain_centre(n) for n in names])
        blk = ((cen[:, 1] + 180.0) // 60.0).astype(int)
        ids = np.unique(blk)
        boots = []
        for _ in range(c.N_BOOT):
            pick = rng.choice(ids, size=len(ids), replace=True)
            idx = np.concatenate([np.where(blk == b)[0] for b in pick])
            boots.append(spearmanr(x[idx], y[idx]).statistic)
        t["row8_free_run"] = {"rho": round(float(spearmanr(x, y).statistic), 3), "n": len(names),
                              "archived_H13a": s13["H13a"], "ceiling_P": s13["era5_across_period_ceiling"]["P"],
                              "block_boot_ci90_lon_sectors": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)],
                              "u_rho": s13.get("H13c", {}).get("rho") if isinstance(s13.get("H13c"), dict) else None}
        # content checks (D4 of verify_article1_descriptives): P vs u, P vs persistence over 39 domains
        ue = np.asarray([dv[n]["u_era5"] for n in names])
        pe = np.asarray([dv[n]["persist_era5"] for n in names])
        t["content_checks_39"] = {"rho_P_vs_u": round(float(spearmanr(x, ue).statistic), 3),
                                  "rho_P_vs_persistence": round(float(spearmanr(x, pe).statistic), 3)}
    except FileNotFoundError as exc:
        t["row8_free_run"] = f"not available: {exc}"
    return t


def _fmt(v):
    if isinstance(v, dict) and "diff" in v:
        return f"Δ={v['diff']:+.2f}, p={v['p']:.3f}"
    if isinstance(v, dict) and "rho" in v:
        return f"ρ={v['rho']}" + (f", p={v['p']}" if "p" in v else "")
    return str(v)


def markdown(old: dict, new: dict) -> str:
    rows = [
        ("Строка 1: сигнатура, отложенные 32, ступени 1–5", "row1_heldout_raw_steps1_5"),
        ("Строка 1: ступени 2–5", "row1_heldout_raw_steps2_5"),
        ("Строка 7: ступени 4–5", "row1_heldout_raw_steps4_5"),
        ("Только ступень 5", "row1_heldout_raw_step5"),
        ("Строка 4: сигнатура MERRA-2 (48, 3 ступени)", "row4_merra_signature"),
        ("Строка 4: география ERA5–MERRA-2 (12)", "row4_geography_era5_vs_merra2"),
        ("Строка 5: равновеликие, 48 основных, ступени 1–5", "row5_km_anchored_primary_steps1_5"),
        ("Строка 5: равновеликие, 48 основных, ступени 2–5", "row5_km_anchored_primary_steps2_5"),
        ("Строка 5: равновеликие, 32 отложенных, ступени 1–5", "row5_km_anchored_heldout_steps1_5"),
        ("Строка 5: равновеликие, 32 отложенных, ступени 2–5", "row5_km_anchored_heldout_steps2_5"),
        ("Строка 8: свободный расчёт vs ERA5 (39)", "row8_free_run"),
        ("Ориентир ERA5 между периодами (12 областей)", "era5_across_period_reference_12"),
    ]
    lines = ["| Величина | Было (архивный пайплайн) | Стало (V12) |", "|---|---|---|"]
    for title, key in rows:
        lines.append(f"| {title} | {_fmt(old.get(key, '—'))} | {_fmt(new.get(key, '—'))} |")
    for name in ("steps1_5", "steps2_5"):
        for strat in ("season", "year", "period"):
            k = f"{strat}_stratified"
            lines.append(f"| Все 96, {name}, перестановки внутри {strat} | {_fmt(old[f'all96_raw_{name}'][k])} | {_fmt(new[f'all96_raw_{name}'][k])} |")
            k = f"cross_{strat}_pairs"
            lines.append(f"| Все 96, {name}, только пары из разных {strat} | {_fmt(old[f'all96_raw_{name}'][k])} | {_fmt(new[f'all96_raw_{name}'][k])} |")
    for st in range(1, 6):
        vo = old["variance_components_raw"][f"step{st}"]
        vn = new["variance_components_raw"][f"step{st}"]
        lines.append(f"| Доля дисперсии «регион» / ICC, ступень {st} | {vo['share_region']:.2f} / {vo['icc1_region']:.2f} | {vn['share_region']:.2f} / {vn['icc1_region']:.2f} |")
    ca_o, ca_n = old["congo_amazon_step5"], new["congo_amazon_step5"]
    lines.append(f"| Конго / Амазония, ступень 5, медиана (размах) | {ca_o['congo_median']} ({ca_o['congo_range']}) / {ca_o['amazon_median']} ({ca_o['amazon_range']}) | "
                 f"{ca_n['congo_median']} ({ca_n['congo_range']}) / {ca_n['amazon_median']} ({ca_n['amazon_range']}) |")
    lines.append("")
    lines.append("География, ступени 4–5 (медиана по 8 окнам), было: " + ", ".join(f"{k} {v}" for k, v in old["geography_steps4_5_median8"].items()))
    lines.append("")
    lines.append("География, ступени 4–5, стало: " + ", ".join(f"{k} {v}" for k, v in new["geography_steps4_5_median8"].items()))
    if "fine_ordering_km_anchored_steps2_3" in new:
        lines.append("")
        lines.append("Тонкие ступени 2–3 (с поправкой, равновеликие), стало: " + ", ".join(f"{k} {v}" for k, v in new["fine_ordering_km_anchored_steps2_3"]))
    if "b18_summary" in new:
        lines.append("")
        lines.append(f"B18 география градусы–км (стало): {new['b18_summary']['H18c_geography']}; по регионам {new['b18_summary']['H18c_per_region_rho']}")
        lines.append(f"B18 чувствительность к протяжённости (мин. ρ по регионам), стало: {new['b18_summary']['C18_4_extent_min_rho']}; динамическая лестница: {_fmt(new['b18_summary']['P18_3_dynamic_anchored'])}")
        lines.append(f"B18 Конго–Амазония (с поправкой), стало: {new['b18_summary']['H18e_congo_amazon']['anchored']}")
    if isinstance(new.get("row8_free_run"), dict):
        lines.append("")
        lines.append(f"Строка 8 подробно, стало: {new['row8_free_run']}; проверки содержания: {new.get('content_checks_39')}")
        lines.append(f"Строка 8 подробно, было: {old['row8_free_run']}; проверки содержания: {old.get('content_checks_39')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-only", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(c.SEED)
    old = table("", rng)
    if args.old_only:
        print(json.dumps({k: v for k, v in old.items() if "all96" not in k and "variance" not in k}, indent=1, ensure_ascii=False)[:5000])
        return
    new = table("_v12", rng)
    (OUT / "table2_compare.json").write_text(json.dumps({"old": old, "new": new}, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(old, new)
    (OUT / "table2_compare.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
