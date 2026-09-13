#!/usr/bin/env python3
"""Descriptive quantities quoted in article 1 that no summary.json stores.

Four groups of numbers appear in the manuscript as descriptive readouts rather
than as pre-registered test statistics:

  D1  regional ordering of the profile on the RESOLVED steps 4-5
      (medians over the eight seasonal windows of each of the twelve main
       regions, degree boxes) -- the ordering quoted in section 3.1
  D2  regional ordering of the surrogate-anchored profile on the FINE steps 1-3
      (equal-area boxes, experiment B18) -- the ordering the second paper uses
  D3  the Congo/Amazon contrast on step 5 (medians and ranges over eight
      windows each), and the range of the tropical oceanic regions
  D4  the content checks of section 3.4: the profile against the
      information-theoretic irrecoverability u, against band amplitude, and
      against the temporal persistence of the activity maps
  D5  the cross-reanalysis geography of the PROFILE itself (ERA5 vs MERRA-2,
      twelve regions, windows W9-W12, resolved steps only).  Phase 8 froze its
      two hypotheses (H8a clustering, H8b geography) for the curvature profile
      ||F||, not for P; for P the B8 run stored only per-window profiles and a
      descriptive clustering statistic.  D5 closes that gap: it is the number
      the manuscript quotes as "geography of the profile agrees between the
      two reanalyses".  MERRA-2's ladder starts at 200 km, so its three steps
      are (<200|200-400), (200-400|400-800), (400-800|800-1600); the last two
      coincide with ERA5 steps 4-5 and only those are compared.

Output: results/verify_article1_descriptives/summary.json
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
RES = _HERE / "results"
OUT = RES / "verify_article1_descriptives"

DEG_SOURCES = ("experiment_B2_scale_irreversibility",
               "experiment_B3_scattering_benchmark",
               "experiment_B2b_heldout_invariants")
TROPICAL_OCEANIC = ("R1_WPWP", "R5_SPCZ", "R10_INDO")


def _degree_box_profiles() -> dict[str, dict[str, list]]:
    """region -> window -> raw profile (5 steps), degree boxes, all windows."""
    per: dict[str, dict[str, list]] = defaultdict(dict)
    for d in DEG_SOURCES:
        for f in glob.glob(str(RES / d / "R*__*.json")):
            rec = json.loads(Path(f).read_text(encoding="utf-8"))
            prof = rec.get("P_real")
            if prof is None or len(prof) != 5:
                continue
            per[rec["region"]][rec["window"]] = prof
    return per


def d1_resolved_ordering(per):
    out = {}
    for reg, wins in per.items():
        arr = np.asarray(list(wins.values()))
        out[reg] = {"n_windows": int(arr.shape[0]),
                    "median_mean_steps45": float(np.median(arr[:, 3:5].mean(axis=1))),
                    "median_step5": float(np.median(arr[:, 4]))}
    order = sorted(out, key=lambda r: -out[r]["median_mean_steps45"])
    return {"per_region": out, "descending_order": order}


def d2_fine_anchored_ordering():
    per = defaultdict(list)
    for f in glob.glob(str(RES / "experiment_B18_equal_km_regions" / "R*__*.json")):
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        per[rec["region"]].append(float(np.mean(rec["km_1.0"]["P_anchored"][:3])))
    out = {r: float(np.median(v)) for r, v in per.items()}
    return {"per_region_median": out,
            "descending_order": sorted(out, key=lambda r: -out[r])}


def d3_congo_amazon(per):
    out = {}
    for reg in ("R7_CONGO", "R3_AMAZ"):
        v = np.asarray([p[4] for p in per[reg].values()])
        out[reg] = {"n_windows": int(v.size), "median_step5": float(np.median(v)),
                    "min": float(v.min()), "max": float(v.max())}
    trop = [float(np.median([p[4] for p in per[r].values()])) for r in TROPICAL_OCEANIC]
    out["tropical_oceanic_step5_range"] = [min(trop), max(trop)]
    out["contrast_step5"] = out["R7_CONGO"]["median_step5"] - out["R3_AMAZ"]["median_step5"]
    return out


def d4_content_checks(per):
    b13 = json.loads((RES / "experiment_B13_free_running" / "summary.json")
                     .read_text(encoding="utf-8"))["domain_values"]
    b12 = json.loads((RES / "experiment_B12_estimator_validity" / "summary.json")
                     .read_text(encoding="utf-8"))["domain_values"]
    names = [n for n in b13 if n in b12]
    P = [b13[n]["P_era5"] for n in names]
    res = {"n_domains": len(names),
           "rho_P_vs_irrecoverability_u": float(spearmanr(P, [b12[n]["u"] for n in names]).statistic),
           "rho_P_vs_envelope_persistence": float(
               spearmanr(P, [b13[n]["persist_era5"] for n in names]).statistic)}
    # band amplitude: log variance of the resolved bands, twelve main regions
    amp, prof = defaultdict(list), defaultdict(list)
    for f in glob.glob(str(RES / "experiment_B3_scattering_benchmark" / "R*__*.json")):
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        prof[rec["region"]].append(float(np.mean(rec["P_real"][3:5])))
        amp[rec["region"]].append(float(np.mean([rec["F_spec"]["logvar_3"],
                                                 rec["F_spec"]["logvar_4"]])))
    regs = sorted(prof)
    res["n_regions_amplitude"] = len(regs)
    res["rho_P_vs_log_band_variance"] = float(spearmanr(
        [np.median(prof[r]) for r in regs], [np.median(amp[r]) for r in regs]).statistic)
    return res


def d5_cross_reanalysis_profile():
    b8_dir = RES / "experiment_B8_cross_reanalysis"
    b3_dir = RES / "experiment_B3_scattering_benchmark"
    merra, era = defaultdict(list), defaultdict(list)
    n_windows = 0
    for f in sorted(glob.glob(str(b8_dir / "R*__*.json"))):
        rec = json.loads(Path(f).read_text(encoding="utf-8"))
        twin = b3_dir / Path(f).name
        if not twin.exists() or len(rec.get("P_merra", [])) != 3:
            continue
        e = json.loads(twin.read_text(encoding="utf-8"))
        merra[rec["region"]].append(float(np.mean(rec["P_merra"][1:3])))
        era[rec["region"]].append(float(np.mean(e["P_real"][3:5])))
        n_windows += 1
    regs = sorted(set(merra) & set(era))
    x = [float(np.median(era[r])) for r in regs]
    y = [float(np.median(merra[r])) for r in regs]
    rs = spearmanr(x, y)
    summ = json.loads((b8_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "note": "H8a/H8b in experiment_B8 summary.json refer to the curvature "
                "profile ||F||; the profile P entered Phase 8 descriptively only",
        "n_regions": len(regs), "n_region_windows": n_windows,
        "steps_compared": "MERRA-2 steps 2-3 vs ERA5 steps 4-5 (200-1600 km)",
        "rho_P_era5_vs_merra2": float(rs.statistic),
        "p_two_sided_asym": float(rs.pvalue),
        "era5_median_steps45_by_region": dict(zip(regs, x)),
        "merra2_median_steps23_by_region": dict(zip(regs, y)),
        "P_signature_merra2_raw_descriptive": summ["P_signature_merra2_descriptive"],
    }


def main() -> None:
    per = _degree_box_profiles()
    res = {"note": "descriptive readouts quoted in article 1; no thresholds, "
                   "no null models -- recomputation aid only",
           "D1_resolved_steps45_ordering": d1_resolved_ordering(per),
           "D2_fine_anchored_ordering": d2_fine_anchored_ordering(),
           "D3_congo_amazon_step5": d3_congo_amazon(per),
           "D4_content_checks": d4_content_checks(per),
           "D5_cross_reanalysis_profile": d5_cross_reanalysis_profile()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "D1_resolved_steps45_ordering"},
                     indent=2))
    d1 = res["D1_resolved_steps45_ordering"]
    for r in d1["descending_order"]:
        e = d1["per_region"][r]
        print("%-10s n=%d mean45=%.3f step5=%.3f" % (r, e["n_windows"],
              e["median_mean_steps45"], e["median_step5"]))


if __name__ == "__main__":
    main()
