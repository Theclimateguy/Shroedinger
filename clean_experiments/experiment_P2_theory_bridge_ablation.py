#!/usr/bin/env python3
"""Ablation for P2 theory bridge calibration.

Workflow:
1) Build base tile dataset once from MRMS panel.
2) Re-apply theory bridge (rho_occ + F_comm + lambda_local) for many configs.
3) Fast screening without permutation.
4) Final evaluation on top-K configs with permutation test.
5) Save selected stable config and ready-to-run command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_P2_noncommuting_coarse_graining import (
        _apply_theory_bridge,
        _evaluate_single_scale,
        _safe_r2,
        _build_tile_dataset,
    )
except ModuleNotFoundError:
    from experiment_P2_noncommuting_coarse_graining import (  # type: ignore
        _apply_theory_bridge,
        _evaluate_single_scale,
        _safe_r2,
        _build_tile_dataset,
    )


def _parse_float_list(text: str) -> list[float]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    return [float(x) for x in vals]


def _parse_weight_sets(text: str) -> list[np.ndarray]:
    sets = [s.strip() for s in text.split(";") if s.strip()]
    out: list[np.ndarray] = []
    for s in sets:
        arr = np.asarray(_parse_float_list(s), dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"Each weight set must have 4 values, got '{s}'")
        out.append(arr)
    if len(out) == 0:
        raise ValueError("No weight sets parsed.")
    return out


def _evaluate_config(
    *,
    df_cfg: pd.DataFrame,
    scales_cells: list[int],
    target_col: str,
    ridge_alpha: float,
    n_perm: int,
    seed: int,
    active_quantile: float,
    comm_floor: float,
    max_perm_p: float,
) -> tuple[dict[str, float | bool | str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_cols = [
        "fine_density_mean",
        "fine_density_std",
        "fine_occ_mean",
        "fine_occ_std",
        "fine_rate_mean",
        "fine_rate_std",
        "hour_sin",
        "hour_cos",
    ]
    full_cols = baseline_cols + ["lambda_local"]

    per_rows: list[dict[str, float | bool | str]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    perm_parts: list[pd.DataFrame] = []

    for scale_l in scales_cells:
        ds = df_cfg[df_cfg["scale_l"].to_numpy(dtype=float) == float(scale_l)].copy().reset_index(drop=True)
        if len(ds) < 300:
            continue

        row, oof_df, fold_df, perm_df = _evaluate_single_scale(
            df_scale=ds,
            target_col=target_col,
            baseline_cols=baseline_cols,
            full_cols=full_cols,
            ridge_alpha=ridge_alpha,
            n_perm=n_perm,
            seed=seed,
            active_quantile=active_quantile,
            comm_floor=comm_floor,
            max_perm_p=max_perm_p,
        )
        per_rows.append({"scale_l": float(scale_l), **row})
        oof_parts.append(oof_df)
        fold_parts.append(fold_df)
        if len(perm_df) > 0:
            perm_parts.append(perm_df)

    if len(per_rows) == 0:
        raise RuntimeError("No valid scales after filtering.")

    oof_all = pd.concat(oof_parts, ignore_index=True)
    fold_all = pd.concat(fold_parts, ignore_index=True)
    perm_all = pd.concat(perm_parts, ignore_index=True) if len(perm_parts) > 0 else pd.DataFrame()
    per_df = pd.DataFrame(per_rows)

    mae_base = float(np.mean(np.abs(oof_all["target_value"] - oof_all["pred_baseline"])))
    mae_full = float(np.mean(np.abs(oof_all["target_value"] - oof_all["pred_full"])))
    r2_base = _safe_r2(oof_all["target_value"].to_numpy(dtype=float), oof_all["pred_baseline"].to_numpy(dtype=float))
    r2_full = _safe_r2(oof_all["target_value"].to_numpy(dtype=float), oof_all["pred_full"].to_numpy(dtype=float))

    h1_all = bool(np.all(per_df["H1_space"].astype(bool).to_numpy()))
    h2_all = bool(np.all(per_df["H2_space"].astype(bool).to_numpy()))
    h3_all = bool(np.all(per_df["H3_space"].astype(bool).to_numpy()))
    pass_all_scales = bool(np.all(per_df["PASS_ALL"].astype(bool).to_numpy()))

    overall = {
        "n_rows": float(len(oof_all)),
        "n_events": float(oof_all["event_id"].nunique()),
        "n_scales": float(len(per_df)),
        "mae_baseline": mae_base,
        "mae_full": mae_full,
        "overall_mae_gain": float(mae_base - mae_full),
        "r2_baseline": r2_base,
        "r2_full": r2_full,
        "overall_r2_gain": float((r2_full - r2_base) if np.isfinite(r2_base) and np.isfinite(r2_full) else np.nan),
        "min_scale_mae_gain": float(np.nanmin(per_df["mae_gain"].to_numpy(dtype=float))),
        "mean_scale_mae_gain": float(np.nanmean(per_df["mae_gain"].to_numpy(dtype=float))),
        "mean_scale_r2_gain": float(np.nanmean(per_df["r2_gain"].to_numpy(dtype=float))),
        "mean_comm_defect": float(np.nanmean(per_df["comm_defect_mean"].to_numpy(dtype=float))),
        "mean_perm_p": float(np.nanmean(per_df["perm_p_value"].to_numpy(dtype=float))) if n_perm > 0 else np.nan,
        "H1_all_scales": h1_all,
        "H2_all_scales": h2_all,
        "H3_all_scales": h3_all,
        "PASS_all_scales": pass_all_scales,
        "n_scales_pass": float(np.sum(per_df["PASS_ALL"].astype(bool).to_numpy())),
    }

    return overall, per_df, oof_all, fold_all, perm_all


def _config_id(i: int) -> str:
    return f"C{i:03d}"


def run(args: argparse.Namespace) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    panel_df = pd.read_csv(args.panel_csv)
    scales_cells = sorted({int(s) for s in args.scales_cells if int(s) >= 2})

    alphas = [float(x) for x in args.decoherence_alphas]
    powers = [float(x) for x in args.lambda_scale_powers]
    weight_sets = _parse_weight_sets(args.weight_sets)

    print("[ablation] step 1/5 build base dataset once", flush=True)
    base_df = _build_tile_dataset(
        panel_df=panel_df,
        scales_cells=scales_cells,
        mrms_downsample=args.mrms_downsample,
        threshold=args.mrms_threshold,
        min_valid_frac=args.min_valid_frac,
        lambda_weights=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float),
        lambda_scale_power=0.0,
        decoherence_alpha=1.0,
        max_rows=args.max_rows,
    )
    base_df.to_csv(outdir / "base_tile_dataset.csv", index=False)

    target_col = "target_density_coarse" if args.target == "density" else "target_occ_coarse"

    print("[ablation] step 2/5 screening grid (no permutation)", flush=True)
    screen_rows: list[dict[str, float | bool | str]] = []
    cfg_per_scale_rows: list[pd.DataFrame] = []

    cfg_idx = 0
    for alpha in alphas:
        for pwr in powers:
            for w in weight_sets:
                cfg_idx += 1
                cid = _config_id(cfg_idx)
                print(f"[screen] {cid} alpha={alpha} power={pwr} w={w.tolist()}", flush=True)

                df_cfg = _apply_theory_bridge(
                    base_df,
                    lambda_weights=w,
                    lambda_scale_power=pwr,
                    decoherence_alpha=alpha,
                )

                overall, per_df, _, _, _ = _evaluate_config(
                    df_cfg=df_cfg,
                    scales_cells=scales_cells,
                    target_col=target_col,
                    ridge_alpha=args.ridge_alpha,
                    n_perm=0,
                    seed=args.seed,
                    active_quantile=args.active_quantile,
                    comm_floor=args.comm_floor,
                    max_perm_p=args.max_perm_p,
                )

                row = {
                    "config_id": cid,
                    "decoherence_alpha": float(alpha),
                    "lambda_scale_power": float(pwr),
                    "w_occ": float(w[0]),
                    "w_sq": float(w[1]),
                    "w_log": float(w[2]),
                    "w_grad": float(w[3]),
                    **overall,
                }
                screen_rows.append(row)

                p = per_df.copy()
                p.insert(0, "config_id", cid)
                p["decoherence_alpha"] = float(alpha)
                p["lambda_scale_power"] = float(pwr)
                p["w_occ"] = float(w[0])
                p["w_sq"] = float(w[1])
                p["w_log"] = float(w[2])
                p["w_grad"] = float(w[3])
                cfg_per_scale_rows.append(p)

    screening_df = pd.DataFrame(screen_rows)
    screening_df = screening_df.sort_values(
        ["PASS_all_scales", "overall_mae_gain", "overall_r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    screening_df.to_csv(outdir / "ablation_screening_configs.csv", index=False)

    per_scale_screen_df = pd.concat(cfg_per_scale_rows, ignore_index=True)
    per_scale_screen_df.to_csv(outdir / "ablation_screening_per_scale.csv", index=False)

    top_k = max(1, int(args.top_k))
    top_df = screening_df.head(top_k).copy()

    print("[ablation] step 3/5 final evaluation on top-K with permutation", flush=True)
    final_rows: list[dict[str, float | bool | str]] = []
    final_per_scale_parts: list[pd.DataFrame] = []

    for _, r in top_df.iterrows():
        cid = str(r["config_id"])
        alpha = float(r["decoherence_alpha"])
        pwr = float(r["lambda_scale_power"])
        w = np.asarray([r["w_occ"], r["w_sq"], r["w_log"], r["w_grad"]], dtype=float)

        print(f"[final] {cid} alpha={alpha} power={pwr} w={w.tolist()}", flush=True)

        df_cfg = _apply_theory_bridge(
            base_df,
            lambda_weights=w,
            lambda_scale_power=pwr,
            decoherence_alpha=alpha,
        )

        overall, per_df, _, _, _ = _evaluate_config(
            df_cfg=df_cfg,
            scales_cells=scales_cells,
            target_col=target_col,
            ridge_alpha=args.ridge_alpha,
            n_perm=args.n_perm_final,
            seed=args.seed,
            active_quantile=args.active_quantile,
            comm_floor=args.comm_floor,
            max_perm_p=args.max_perm_p,
        )

        row = {
            "config_id": cid,
            "decoherence_alpha": alpha,
            "lambda_scale_power": pwr,
            "w_occ": float(w[0]),
            "w_sq": float(w[1]),
            "w_log": float(w[2]),
            "w_grad": float(w[3]),
            **overall,
        }
        final_rows.append(row)

        p = per_df.copy()
        p.insert(0, "config_id", cid)
        p["decoherence_alpha"] = alpha
        p["lambda_scale_power"] = pwr
        p["w_occ"] = float(w[0])
        p["w_sq"] = float(w[1])
        p["w_log"] = float(w[2])
        p["w_grad"] = float(w[3])
        final_per_scale_parts.append(p)

    final_df = pd.DataFrame(final_rows)
    final_df = final_df.sort_values(
        ["PASS_all_scales", "overall_mae_gain", "overall_r2_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    final_df.to_csv(outdir / "ablation_final_configs.csv", index=False)

    final_per_scale_df = pd.concat(final_per_scale_parts, ignore_index=True)
    final_per_scale_df.to_csv(outdir / "ablation_final_per_scale.csv", index=False)

    selected = final_df.iloc[0].copy()
    selected_df = pd.DataFrame([selected])
    selected_df.to_csv(outdir / "selected_config.csv", index=False)

    command = (
        "python clean_experiments/experiment_P2_noncommuting_coarse_graining.py "
        f"--panel-csv {args.panel_csv} "
        "--outdir clean_experiments/results/experiment_P2_noncommuting_coarse_graining_calibrated "
        f"--target {args.target} "
        f"--scales-cells {' '.join(str(s) for s in scales_cells)} "
        f"--mrms-downsample {args.mrms_downsample} "
        f"--mrms-threshold {args.mrms_threshold} "
        f"--lambda-weights {selected['w_occ']} {selected['w_sq']} {selected['w_log']} {selected['w_grad']} "
        f"--lambda-scale-power {selected['lambda_scale_power']} "
        f"--decoherence-alpha {selected['decoherence_alpha']} "
        f"--ridge-alpha {args.ridge_alpha} "
        f"--n-perm {args.n_perm_final}"
    )
    (outdir / "selected_run_command.sh").write_text(command + "\n", encoding="utf-8")

    print("[ablation] step 4/5 run calibrated config", flush=True)
    df_sel = _apply_theory_bridge(
        base_df,
        lambda_weights=np.asarray(
            [selected["w_occ"], selected["w_sq"], selected["w_log"], selected["w_grad"]],
            dtype=float,
        ),
        lambda_scale_power=float(selected["lambda_scale_power"]),
        decoherence_alpha=float(selected["decoherence_alpha"]),
    )
    df_sel.to_csv(outdir / "selected_config_tile_dataset.csv", index=False)

    print("[ablation] step 5/5 report", flush=True)
    rep = [
        "# P2 Theory Bridge Ablation",
        "",
        f"- panel: `{args.panel_csv}`",
        f"- scales: `{', '.join(str(s) for s in scales_cells)}`",
        f"- screening configs: `{len(screening_df)}`",
        f"- final top-k: `{top_k}`",
        f"- final permutations per config: `{args.n_perm_final}`",
        "",
        "## Selected config",
        f"- config_id: `{selected['config_id']}`",
        f"- decoherence_alpha: `{float(selected['decoherence_alpha'])}`",
        f"- lambda_scale_power: `{float(selected['lambda_scale_power'])}`",
        (
            "- weights [occ,sq,log,grad]: "
            f"`[{float(selected['w_occ'])}, {float(selected['w_sq'])}, {float(selected['w_log'])}, {float(selected['w_grad'])}]`"
        ),
        f"- overall_mae_gain: `{float(selected['overall_mae_gain']):.6e}`",
        f"- overall_r2_gain: `{float(selected['overall_r2_gain']):.6e}`",
        f"- PASS_all_scales: `{bool(selected['PASS_all_scales'])}`",
        "",
        "## Artifacts",
        "- `ablation_screening_configs.csv`",
        "- `ablation_screening_per_scale.csv`",
        "- `ablation_final_configs.csv`",
        "- `ablation_final_per_scale.csv`",
        "- `selected_config.csv`",
        "- `selected_run_command.sh`",
        "- `selected_config_tile_dataset.csv`",
    ]
    (outdir / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"Ablation complete: {outdir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("clean_experiments/results/realpilot_2024_dataset_panel_v1_expanded.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("clean_experiments/results/experiment_P2_theory_bridge_ablation"),
    )
    p.add_argument("--target", choices=["density", "occupancy"], default="density")

    p.add_argument("--scales-cells", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--mrms-downsample", type=int, default=16)
    p.add_argument("--mrms-threshold", type=float, default=3.0)
    p.add_argument("--min-valid-frac", type=float, default=0.90)
    p.add_argument("--max-rows", type=int, default=0)

    p.add_argument("--decoherence-alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--lambda-scale-powers", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    p.add_argument(
        "--weight-sets",
        type=str,
        default="1,1,1,1;1,0.5,1,0.5;1,1,0.5,0.5;1.5,1,1,1;0.8,1.2,1,1",
        help="Semicolon-separated 4-tuples: occ,sq,log,grad",
    )

    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-quantile", type=float, default=0.67)
    p.add_argument("--comm-floor", type=float, default=1e-4)
    p.add_argument("--max-perm-p", type=float, default=0.05)

    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--n-perm-final", type=int, default=49)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
