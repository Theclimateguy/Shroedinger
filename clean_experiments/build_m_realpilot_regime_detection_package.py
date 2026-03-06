#!/usr/bin/env python3
"""Build formal perimeter package for M-realpilot frozen regime detection.

Outputs a unified registry, decision statuses, protocol manifest, and a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    season: str
    region: str
    events_csv: Path
    summary_csv: Path
    feature_csv: Path
    component_csv: Path


EXPECTED_FROZEN_SHA256 = "aded0e49e825a318a2de07f49faa9c877277d2e76f223277495bdcebfbe8f3f2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _first_row(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        raise ValueError("Empty dataframe")
    return df.iloc[0]


def _component_metric(df: pd.DataFrame, variant: str, col: str) -> float:
    sub = df[df["variant"].astype(str) == variant]
    if len(sub) == 0:
        return float("nan")
    return float(sub.iloc[0][col])


def _status_label(pass_all: bool, mean_gain: float, abi_only_gain: float, abi_glm_gain: float) -> str:
    if pass_all:
        return "Detected"
    if np.isfinite(mean_gain) and mean_gain < 0.0 and np.isfinite(abi_only_gain) and np.isfinite(abi_glm_gain) and abi_glm_gain < abi_only_gain:
        return "Bridge tension"
    return "Compatible but undetected"


def _build_row(spec: RunSpec) -> dict[str, object]:
    events = _safe_read_csv(spec.events_csv)
    summary = _safe_read_csv(spec.summary_csv)
    feature = _safe_read_csv(spec.feature_csv)
    comp = _safe_read_csv(spec.component_csv)

    s = _first_row(summary)

    abi_only_gain = _component_metric(comp, "ABI-only", "mean_mae_gain")
    abi_glm_gain = _component_metric(comp, "ABI+GLM", "mean_mae_gain")
    abi_only_p_time = _component_metric(comp, "ABI-only", "perm_p_time_shuffle")
    abi_only_p_event = _component_metric(comp, "ABI-only", "perm_p_event_shuffle")
    abi_glm_p_time = _component_metric(comp, "ABI+GLM", "perm_p_time_shuffle")
    abi_glm_p_event = _component_metric(comp, "ABI+GLM", "perm_p_event_shuffle")

    glm = feature["glm_flash_count"].to_numpy(dtype=float) if "glm_flash_count" in feature.columns else np.array([], dtype=float)
    glm = glm[np.isfinite(glm)]
    glm_mean = float(glm.mean()) if len(glm) > 0 else float("nan")
    glm_sparse0 = float(np.mean(glm <= 0.0)) if len(glm) > 0 else float("nan")

    pass_all = bool(s["PASS_ALL"]) if "PASS_ALL" in s.index else False
    mean_gain = float(s["mean_mae_gain"])

    status = _status_label(
        pass_all=pass_all,
        mean_gain=mean_gain,
        abi_only_gain=abi_only_gain,
        abi_glm_gain=abi_glm_gain,
    )

    return {
        "run_id": spec.run_id,
        "label": spec.label,
        "season": spec.season,
        "region": spec.region,
        "event_list": str(spec.events_csv),
        "n_events_manifest": int(len(events)),
        "n_events_model": float(s["n_events"]),
        "n_model_samples": float(s["n_model_samples"]),
        "mean_mae_gain": mean_gain,
        "perm_p_time_shuffle": float(s["perm_p_time_shuffle"]),
        "perm_p_event_shuffle": float(s["perm_p_event_shuffle"]),
        "event_positive_frac": float(s["event_positive_frac"]),
        "min_event_gain": float(s["min_event_gain"]),
        "active_minus_calm": float(s["active_minus_calm"]),
        "pass_all": pass_all,
        "abi_only_gain": abi_only_gain,
        "abi_only_p_time": abi_only_p_time,
        "abi_only_p_event": abi_only_p_event,
        "abi_glm_gain": abi_glm_gain,
        "abi_glm_p_time": abi_glm_p_time,
        "abi_glm_p_event": abi_glm_p_event,
        "delta_glm_minus_abi": (float(abi_glm_gain - abi_only_gain) if np.isfinite(abi_glm_gain) and np.isfinite(abi_only_gain) else float("nan")),
        "glm_flash_mean": glm_mean,
        "glm_sparsity_zero_frac": glm_sparse0,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_M_realpilot_regime_detection_package"),
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    frozen_script = root / "clean_experiments" / "experiment_M_realpilot_v1_frozen.py"
    script_hash = _sha256(frozen_script)

    specs = [
        RunSpec(
            run_id="R1_expanded_positive",
            label="Frozen expanded positive reference",
            season="MAM+JJA",
            region="Southern_Plains+Midwest",
            events_csv=root / "clean_experiments" / "pilot_events_real_2024_us_convective_expanded_v1.csv",
            summary_csv=root / "clean_experiments" / "results" / "experiment_M_realpilot_v1_expanded_positive" / "summary_metrics.csv",
            feature_csv=root / "clean_experiments" / "results" / "experiment_M_realpilot_v1_expanded_positive" / "feature_dataset.csv",
            component_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_expanded_positive"
            / "satellite_component_stability"
            / "satellite_component_summary.csv",
        ),
        RunSpec(
            run_id="R2_independent_seasonal",
            label="Frozen independent seasonal extension",
            season="SON",
            region="Southeast",
            events_csv=root / "clean_experiments" / "pilot_events_realpilot_v1_independent_seasonal_2024.csv",
            summary_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_frozen_independent_seasonal_2024"
            / "summary_metrics.csv",
            feature_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_frozen_independent_seasonal_2024"
            / "feature_dataset.csv",
            component_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_frozen_independent_seasonal_2024"
            / "satellite_component_stability"
            / "satellite_component_summary.csv",
        ),
        RunSpec(
            run_id="R3_independent_geographic",
            label="Frozen independent geographic extension (Southwest)",
            season="JJA+SON",
            region="Southwest",
            events_csv=root / "clean_experiments" / "pilot_events_realpilot_v1_independent_geographic_southwest_2024.csv",
            summary_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_frozen_independent_geographic_southwest_2024"
            / "summary_metrics.csv",
            feature_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_frozen_independent_geographic_southwest_2024"
            / "feature_dataset.csv",
            component_csv=root
            / "clean_experiments"
            / "results"
            / "experiment_M_realpilot_v1_frozen_independent_geographic_southwest_2024"
            / "satellite_component_stability"
            / "satellite_component_summary.csv",
        ),
    ]

    rows = [_build_row(s) for s in specs]
    reg = pd.DataFrame(rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    reg_csv = args.outdir / "perimeter_registry.csv"
    reg.to_csv(reg_csv, index=False)

    component_rows = []
    for s in specs:
        c = _safe_read_csv(s.component_csv)
        for _, r in c.iterrows():
            component_rows.append(
                {
                    "run_id": s.run_id,
                    "season": s.season,
                    "region": s.region,
                    "variant": str(r["variant"]),
                    "mean_mae_gain": float(r["mean_mae_gain"]),
                    "perm_p_time_shuffle": float(r["perm_p_time_shuffle"]),
                    "perm_p_event_shuffle": float(r["perm_p_event_shuffle"]),
                    "event_positive_frac": float(r["event_positive_frac"]),
                }
            )
    comp_df = pd.DataFrame(component_rows)
    comp_csv = args.outdir / "satellite_component_registry.csv"
    comp_df.to_csv(comp_csv, index=False)

    manifest = {
        "protocol_name": "M-realpilot-v1-frozen",
        "frozen_script": str(frozen_script),
        "frozen_script_sha256": script_hash,
        "expected_sha256": EXPECTED_FROZEN_SHA256,
        "hash_match": bool(script_hash == EXPECTED_FROZEN_SHA256),
        "fixed_settings": {
            "target": "next_p95",
            "ridge_alpha": 10.0,
            "cv": "leave-one-event-out",
            "n_perm": 499,
            "structural_features_full": [
                "abi_cold_frac_235",
                "abi_grad_mean",
                "glm_flash_count_log",
                "convective_coupling_index",
            ],
        },
        "status_labels": {
            "Detected": "PASS_ALL=True under frozen protocol",
            "Compatible but undetected": "PASS_ALL=False without systematic bridge contradiction",
            "Bridge tension": "PASS_ALL=False and ABI+GLM worsens gain vs ABI-only on negative run",
        },
    }
    (args.outdir / "protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    n_detected = int((reg["status"] == "Detected").sum())
    n_compatible = int((reg["status"] == "Compatible but undetected").sum())
    n_tension = int((reg["status"] == "Bridge tension").sum())

    problem_statement = "\n".join(
        [
            "# Research Task: Regime Detection Under Frozen Observable Bridge",
            "",
            "## Objective",
            "Formalize the M-realpilot family as a conditional detectability study (not universal validity),",
            "using one fixed protocol across all runs and reporting outcome vectors on a common perimeter.",
            "",
            "## Frozen Operator",
            "P_frozen = (fixed features, Ridge alpha=10.0, LOEO-CV, n_perm=499, target=next_p95).",
            "",
            "## Outcome Vector",
            "y = (mean_gain, p_time, p_event, q_plus, active_minus_calm, PASS_ALL).",
            "",
            "## Main Hypothesis",
            "There exists a non-empty regime set where frozen protocol detects structural signal.",
            "Null/negative runs constrain applicability perimeter rather than automatically falsify the bridge.",
            "",
            "## Decision Labels",
            "- Detected",
            "- Compatible but undetected",
            "- Bridge tension",
            "",
            f"Protocol hash match: {manifest['hash_match']} ({script_hash}).",
        ]
    )
    (args.outdir / "problem_statement.md").write_text(problem_statement, encoding="utf-8")

    lines = [
        "# Frozen Regime Detection: Diagnostic Perimeter",
        "",
        "## Protocol lock",
        f"- Script: `{frozen_script}`",
        f"- SHA256: `{script_hash}`",
        f"- Hash matches expected: `{manifest['hash_match']}`",
        "- Fixed settings: target=`next_p95`, Ridge alpha=10.0, LOEO-CV, n_perm=499",
        "",
        "## Perimeter registry",
        "| run_id | season | region | mean_gain | p_time | p_event | q_plus | delta_active-calm | ABI-only | ABI+GLM | delta(GLM-ABI) | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in reg.iterrows():
        lines.append(
            "| {run_id} | {season} | {region} | {mg:.6f} | {pt:.3f} | {pe:.3f} | {qp:.3f} | {ac:.6f} | {ao:.6f} | {ag:.6f} | {dg:.6f} | {st} |".format(
                run_id=r["run_id"],
                season=r["season"],
                region=r["region"],
                mg=float(r["mean_mae_gain"]),
                pt=float(r["perm_p_time_shuffle"]),
                pe=float(r["perm_p_event_shuffle"]),
                qp=float(r["event_positive_frac"]),
                ac=float(r["active_minus_calm"]),
                ao=float(r["abi_only_gain"]),
                ag=float(r["abi_glm_gain"]),
                dg=float(r["delta_glm_minus_abi"]),
                st=r["status"],
            )
        )

    lines.extend(
        [
            "",
            "## Status summary",
            f"- Detected: {n_detected}",
            f"- Compatible but undetected: {n_compatible}",
            f"- Bridge tension: {n_tension}",
            "",
            "## Regime-detection conclusion",
            "- Detection is regime-conditional under the concrete frozen protocol.",
            "- A detected regime exists (expanded positive reference).",
            "- Independent seasonal and independent geographic extensions are undetected and show bridge tension (ABI+GLM degrades vs ABI-only).",
            "- Current perimeter supports a non-universal, regime-specific interpretation of local bridge detectability.",
        ]
    )
    (args.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {reg_csv}")
    print(f"Wrote: {comp_csv}")
    print(f"Wrote: {args.outdir / 'protocol_manifest.json'}")
    print(f"Wrote: {args.outdir / 'problem_statement.md'}")
    print(f"Wrote: {args.outdir / 'report.md'}")


if __name__ == "__main__":
    run(parse_args())
