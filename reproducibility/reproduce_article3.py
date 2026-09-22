#!/usr/bin/env python3
"""Reproduce paper 3 of the series end to end - the single entry point.

Paper: "Hierarchy of the atmospheric circulation: the law of inheritance
between scale levels and its meaning for regionalisation"
(journal-format source manuscript_v2/article3/article3_v2_src.tex).
Per-result mapping (every number, table and figure -> script -> artifact):
reproducibility/article3_README.md.

Paper 3 re-uses three committed artifacts of papers 1-2 as inputs
(they are NOT recomputed here; run reproduce_article1.py / reproduce_article2.py
for them): results/experiment_A3_robustness/units/*.json (stored regional P),
results/experiment_B20_armB_global_map/tiles_*.json (stored tile P, eke_syn)
and results/experiment_A2_intermittency/tiles/*/*.json (intermittency block).

Stages
------
  download    fetch the primary fields: ERA5 850 hPa wind for the 12 regions
              (windows W9-W12), ERA5 global wind + monthly covariates (JFM/JAS
              2023), ERA5 invariants, HighResMIP free run (2014)
  compute     Phase 27 (all-pair envelope statistics: 48 units, 902 tiles x 2
              seasons, 0.7-extent units), Phase 28 (law form, depth), Phase 29
              (tile replicate, free-running model, amplitude vs intermittency),
              Phase 30 (unification P <-> law)
  figures     Russian-label journal figures 1-5 (300 dpi) into
              manuscript_v2/article3/figures/
  manuscript  build the journal-format manuscript (tex, pdf, docx) from
              article3_v2_src.tex; needs pandoc, latexmk; a missing tool is
              reported and skipped, not fatal
  check       verify that every expected artifact exists; print SHA-256
  all         download + compute + figures + manuscript + check (default)

Examples
--------
  python reproducibility/reproduce_article3.py --stage check
  python reproducibility/reproduce_article3.py --stage compute --only B27
  python reproducibility/reproduce_article3.py --stage figures

Credentials: ERA5 needs ~/.cdsapirc (CDS API); HighResMIP (ESGF) needs none.
Tile and unit computations cache per shard and can be interrupted and resumed.
All derived tables are committed, so `--stage check` and `--stage figures`
work without any download.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CE = REPO / "clean_experiments"

# name, script, needs-credential, note
DOWNLOADS = [
    ("era5-b3", "download_b3_era5_wind.py", "cds",
     "ERA5 850 hPa wind: 12 regions, windows W9-W12 (2023-2024)"),
    ("era5-global-b20", "download_b20_era5_global.py", "cds",
     "ERA5 global 850 hPa wind JFM+JAS 2023 + monthly covariates (902-tile grid)"),
    ("covariates-b4", "download_b4_covariates.py", "cds",
     "ERA5 invariants: orography, land-sea mask (tile covariates)"),
    ("highresmip-b13", "download_b13_highresmip.py", None,
     "ECMWF-IFS-HR highresSST-present, global 850 hPa wind, 2014 (ESGF)"),
]

# name, script, extra argv, produces (relative to clean_experiments/results/)
EXPERIMENTS = [
    ("B27-units", "experiment_B27_mechanism.py", ["--stage", "units"], None),
    ("B27-tiles-JFM", "experiment_B27_mechanism.py", ["--stage", "tiles", "--season", "JFM"], None),
    ("B27-tiles-JAS", "experiment_B27_mechanism.py", ["--stage", "tiles", "--season", "JAS"], None),
    ("B27-tests", "experiment_B27_mechanism.py", ["--stage", "tests"],
     "experiment_B27_mechanism/summary.json"),
    ("B27-units-x0.7", "experiment_B27_mechanism.py", ["--stage", "units", "--extent", "0.7"], None),
    ("B28-tests", "experiment_B28_inheritance_depth.py", ["--stage", "tests"],
     "experiment_B28_inheritance_depth/summary.json"),
    ("B29-model", "experiment_B29_structure_law.py", ["--stage", "model"], None),
    ("B29-tests", "experiment_B29_structure_law.py", ["--stage", "tests"],
     "experiment_B29_structure_law/summary.json"),
    ("B30-tests", "experiment_B30_unification.py", ["--stage", "tests"],
     "experiment_B30_unification/summary.json"),
]

FIGURES = [
    ("fig-ru", "visualize_article3_ru.py"),          # figures 2-5 (matrix, decay, tiles, P vs h)
    ("fig-scheme", "visualize_article3_scheme_ru.py"),  # figure 1 (illustrated scheme)
]
MANUSCRIPT_TOOL = REPO / "manuscript_v2" / "article3" / "tools" / "build_article3_journal.py"

EXPECTED = [e[3] for e in EXPERIMENTS if e[3]] + [
    "experiment_B29_structure_law/tile_amplitude.npz",
    "experiment_B30_unification/tile_fields.npz",
]
INPUTS_FROM_PAPERS_1_2 = [
    "experiment_A3_robustness/units/R1_WPWP__W9_2023JFM.json",
    "experiment_B20_armB_global_map/tiles_JFM2023.json",
    "experiment_B20_armB_global_map/tiles_JAS2023.json",
    "experiment_A2_intermittency/tiles/JFM/t00_000.json",
    "experiment_B13_free_running/model_R1_WPWP__W15_2014JFM.json",
]


def run(script: str, extra: list[str]) -> int:
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    return subprocess.run([sys.executable, str(CE / script)] + extra,
                          cwd=REPO).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage",
                    choices=["download", "compute", "figures", "manuscript", "check", "all"],
                    default="all")
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on step names (e.g. 'era5', 'B27')")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    failed: list[str] = []

    if args.stage in ("download", "all"):
        steps = [d for d in DOWNLOADS
                 if not args.only or args.only.lower() in d[0].lower()]
        if any(d[2] == "cds" for d in steps) and not (Path.home() / ".cdsapirc").exists():
            print("WARNING: ~/.cdsapirc not found - CDS downloads will fail "
                  "(register at cds.climate.copernicus.eu)")
        for name, script, _, note in steps:
            print(f"\n[{name}] {note}")
            if run(script, []) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at download step {name}")

    if args.stage in ("compute", "all"):
        missing_inputs = [p for p in INPUTS_FROM_PAPERS_1_2 if not (CE / "results" / p).exists()]
        if missing_inputs:
            print("Inputs from papers 1-2 are missing (run reproduce_article1/2.py first):")
            for p in missing_inputs:
                print("  ", p)
            sys.exit(1)
        for name, script, extra, _ in EXPERIMENTS:
            if args.only and args.only.lower() not in name.lower():
                continue
            if run(script, extra) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at experiment {name}")

    if args.stage in ("figures", "all"):
        for name, script in FIGURES:
            if args.only and args.only.lower() not in name.lower():
                continue
            if run(script, []) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at figure step {name}")

    if args.stage in ("manuscript", "all"):
        import shutil as _sh
        missing_tools = [x for x in ("pandoc", "latexmk") if not _sh.which(x)]
        if missing_tools:
            print(f"\nmanuscript: missing {', '.join(missing_tools)} - the corresponding "
                  "outputs are skipped")
        extra = (["--no-docx"] if "pandoc" in missing_tools else []) + \
                (["--no-pdf"] if "latexmk" in missing_tools else [])
        print(f"\n=== {MANUSCRIPT_TOOL.name} {' '.join(extra)} ===", flush=True)
        rc = subprocess.run([sys.executable, str(MANUSCRIPT_TOOL)] + extra, cwd=REPO).returncode
        if rc != 0:
            print("manuscript build failed (non-fatal for the numerical route)")

    if args.stage in ("check", "all"):
        print("\n=== artifact check ===")
        missing = 0
        for rel in EXPECTED:
            p = CE / "results" / rel
            if p.exists():
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                print(f"  OK  {rel}  sha256={digest}")
            else:
                print(f"  MISSING  {rel}")
                missing += 1
        print(f"\n{len(EXPECTED) - missing}/{len(EXPECTED)} artifacts present."
              " Reference hashes: reproducibility/article3_README.md, sect. 5.")

    if failed:
        sys.exit("Failed steps: " + ", ".join(failed))


if __name__ == "__main__":
    main()
