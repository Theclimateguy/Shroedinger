#!/usr/bin/env python3
"""Experiment K: bridge test between Lambda_matter observables and effective Lambda proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from clean_experiments.experiment_G2_single_qubit import run_profile
    from clean_experiments.experiment_wave1_user import run_experiment as run_wave1
except ImportError:
    from experiment_G2_single_qubit import run_profile
    from experiment_wave1_user import run_experiment as run_wave1


FEATURE_COLS = [
    "lm_w1_times_delta_alpha",
    "delta_alpha",
    "lambda_w_g2_mean",
]
PROFILES = ("constant", "gaussian", "oscillating")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-15))


def _fit_ridge_scaled(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    mu = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    x_train_z = (x_train - mu) / sd
    x_eval_z = (x_eval - mu) / sd

    x1 = np.column_stack([np.ones(len(x_train_z)), x_train_z])
    reg = np.diag(np.r_[0.0, np.full(x_train_z.shape[1], ridge_alpha, dtype=float)])
    beta_std = np.linalg.solve(x1.T @ x1 + reg, x1.T @ y_train)

    y_eval = np.column_stack([np.ones(len(x_eval_z)), x_eval_z]) @ beta_std

    coef_raw = beta_std[1:] / sd
    intercept_raw = float(beta_std[0] - np.sum(beta_std[1:] * (mu / sd)))
    return coef_raw, intercept_raw, y_eval


def _stratified_splits(
    labels: np.ndarray,
    train_frac: float,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_splits):
        train: list[int] = []
        test: list[int] = []
        for lab in unique:
            idx = np.where(labels == lab)[0].copy()
            rng.shuffle(idx)
            n_train = max(2, int(round(train_frac * len(idx))))
            n_train = min(n_train, len(idx) - 1)
            train.extend(idx[:n_train].tolist())
            test.extend(idx[n_train:].tolist())

        splits.append((np.array(train, dtype=int), np.array(test, dtype=int)))
    return splits


def _evaluate_splits(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float]]:
    train_r2 = []
    test_r2 = []
    coef_list = []
    intercept_list = []

    for train_idx, test_idx in splits:
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        coef, intercept, yhat_test = _fit_ridge_scaled(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_test,
            ridge_alpha=ridge_alpha,
        )
        _, _, yhat_train = _fit_ridge_scaled(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_train,
            ridge_alpha=ridge_alpha,
        )

        train_r2.append(_r2(y_train, yhat_train))
        test_r2.append(_r2(y_test, yhat_test))
        coef_list.append(coef)
        intercept_list.append(intercept)

    return (
        np.asarray(train_r2, dtype=float),
        np.asarray(test_r2, dtype=float),
        coef_list,
        intercept_list,
    )


def _permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
    n_perm: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    _, test_r2_real, _, _ = _evaluate_splits(
        x=x,
        y=y,
        splits=splits,
        ridge_alpha=ridge_alpha,
    )
    stat_real = float(np.median(test_r2_real))

    rows = []
    count_ge = 0
    for pid in range(n_perm):
        y_perm = rng.permutation(y)
        _, test_r2_perm, _, _ = _evaluate_splits(
            x=x,
            y=y_perm,
            splits=splits,
            ridge_alpha=ridge_alpha,
        )
        stat_perm = float(np.median(test_r2_perm))
        rows.append({"perm_id": int(pid), "stat_perm_median_test_r2": stat_perm})
        if stat_perm >= stat_real:
            count_ge += 1

    p_value = float((count_ge + 1) / (n_perm + 1))
    return p_value, pd.DataFrame(rows)


def _bootstrap_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_boot: int,
    ridge_alpha: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    for bid in range(n_boot):
        idx = rng.integers(0, n, size=n)
        coef, intercept, _ = _fit_ridge_scaled(
            x_train=x[idx],
            y_train=y[idx],
            x_eval=x[idx],
            ridge_alpha=ridge_alpha,
        )
        row = {"boot_id": int(bid), "intercept": float(intercept)}
        for i, fname in enumerate(feature_names):
            row[f"coef_{fname}"] = float(coef[i])
        rows.append(row)

    boot_df = pd.DataFrame(rows)
    sign_consistency = {}
    for fname in feature_names:
        coeff = boot_df[f"coef_{fname}"].to_numpy()
        p_pos = float(np.mean(coeff > 0.0))
        p_neg = float(np.mean(coeff < 0.0))
        sign_consistency[fname] = max(p_pos, p_neg)
    return boot_df, sign_consistency


def _profile_coefficients(
    df: pd.DataFrame,
    feature_names: list[str],
    ridge_alpha: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for profile, sub in df.groupby("profile"):
        if len(sub) < max(6, len(feature_names) + 1):
            continue
        x = sub[feature_names].to_numpy(dtype=float)
        y = sub["lambda_eff_proxy"].to_numpy(dtype=float)
        coef, intercept, yhat = _fit_ridge_scaled(
            x_train=x,
            y_train=y,
            x_eval=x,
            ridge_alpha=ridge_alpha,
        )
        row = {
            "profile": profile,
            "n_cases": int(len(sub)),
            "r2": float(_r2(y, yhat)),
            "intercept": float(intercept),
        }
        for i, fname in enumerate(feature_names):
            row[f"coef_{fname}"] = float(coef[i])
        rows.append(row)

    prof_df = pd.DataFrame(rows)
    sign_consistency: dict[str, float] = {}
    if len(prof_df) == 0:
        for fname in feature_names:
            sign_consistency[fname] = float("nan")
        return prof_df, sign_consistency

    for fname in feature_names:
        col = prof_df[f"coef_{fname}"].to_numpy(dtype=float)
        p_pos = float(np.mean(col > 0.0))
        p_neg = float(np.mean(col < 0.0))
        sign_consistency[fname] = max(p_pos, p_neg)
    return prof_df, sign_consistency


def _sample_case(rng: np.random.Generator, profile: str) -> dict[str, float | str]:
    z_scale = float(rng.uniform(0.0, 1.0))
    z_rot = float(rng.uniform(-1.0, 1.0))
    z_aux = float(rng.uniform(0.0, 1.0))

    delta_alpha = 0.10 + 0.80 * z_scale
    alpha1 = float(np.clip(0.07 + 0.34 * (1.0 - z_scale) + 0.015 * rng.normal(), 0.03, 0.48))
    alpha2 = alpha1 + delta_alpha

    theta2_base = 0.18 + 0.62 * (0.20 + 0.80 * z_scale)
    theta2_amp = 0.04 + 0.52 * (0.10 + 0.90 * abs(z_rot))
    theta1_base = 0.15 + 0.48 * (1.0 - 0.72 * z_scale)
    theta1_amp = 0.06 + 0.38 * (1.0 - abs(z_rot))
    alpha_mod_amp = 0.02 + 0.18 * (0.30 + 0.70 * z_aux)

    da_n = np.clip((delta_alpha - 0.10) / 0.80, 0.0, 1.0)

    omega0 = 0.36 + 0.48 * (0.50 + 0.50 * z_rot)
    if profile == "gaussian":
        omega0 *= 0.94
    elif profile == "oscillating":
        omega0 *= 1.08

    gamma_jump = 0.065 + 0.11 * (0.25 + 0.75 * da_n)
    delta_s_hor = 0.085 + 0.055 * (0.20 + 0.80 * (1.0 - da_n))
    alpha_state = np.pi * (0.18 + 0.64 * z_scale)

    return {
        "profile": profile,
        "z_scale": z_scale,
        "z_rot": z_rot,
        "z_aux": z_aux,
        "alpha1": float(alpha1),
        "alpha2": float(alpha2),
        "delta_alpha": float(delta_alpha),
        "alpha_mod_amp": float(alpha_mod_amp),
        "theta1_base": float(theta1_base),
        "theta1_amp": float(theta1_amp),
        "theta2_base": float(theta2_base),
        "theta2_amp": float(theta2_amp),
        "da_norm": float(da_n),
        "omega0": float(omega0),
        "gamma_jump": float(gamma_jump),
        "delta_s_hor": float(delta_s_hor),
        "alpha_state": float(alpha_state),
    }


def run_experiment(
    outdir: Path,
    *,
    seed: int = 20260225,
    n_cases: int = 48,
    train_frac: float = 0.70,
    n_splits: int = 10,
    n_perm: int = 140,
    n_boot: int = 220,
    g2_steps: int = 90,
    g2_traj: int = 360,
    g2_layers: int = 64,
    ridge_alpha: float = 0.03,
    min_test_r2_median: float = 0.25,
    min_test_r2_p25: float = 0.00,
    max_perm_p: float = 0.05,
    min_sign_consistency: float = 0.65,
    min_profile_sign_consistency: float = 2.0 / 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cases_dir = outdir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    profile_order = [PROFILES[i % len(PROFILES)] for i in range(n_cases)]
    rng.shuffle(profile_order)

    rows = []
    for case_id in range(n_cases):
        profile = profile_order[case_id]
        case = _sample_case(rng, profile=profile)
        case_dir = cases_dir / f"case_{case_id:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        wave1_dir = case_dir / "wave1"
        g2_prefix = case_dir / f"g2_{profile}"

        state_df, _, spatial_df, _ = run_wave1(
            outdir=wave1_dir,
            nx=120,
            alpha1_val=float(case["alpha1"]),
            alpha2_val=float(case["alpha2"]),
            alpha_mod_amp=float(case["alpha_mod_amp"]),
            theta1_base=float(case["theta1_base"]),
            theta1_amp=float(case["theta1_amp"]),
            theta1_center=float(np.pi / 2.0),
            theta1_width=0.35,
            theta2_base=float(case["theta2_base"]),
            theta2_amp=float(case["theta2_amp"]),
            theta2_center=float(3.0 * np.pi / 2.0),
            theta2_width=0.35,
            n_angle_fine=40,
            write_csv=True,
            verbose=False,
        )

        lm_w1 = float(spatial_df["Lambda_matter_plus"].mean())
        lm_n = float(np.tanh(lm_w1 / 1.3))
        da_n = float(case["da_norm"])
        z_rot = float(case["z_rot"])
        z_aux = float(case["z_aux"])

        r_amp = 0.23 + 0.55 * (0.45 * da_n + 0.35 * (0.5 + 0.5 * lm_n) + 0.20 * abs(z_rot))
        beta_g2 = 0.035 + 0.19 * (0.60 * da_n + 0.25 * (0.5 + 0.5 * lm_n) + 0.15 * z_aux)

        _, fit = run_profile(
            omega_profile=profile,
            out_prefix=g2_prefix,
            k_layers=int(g2_layers),
            omega0=float(case["omega0"]),
            r_amp=float(r_amp),
            beta=float(beta_g2),
            gamma_jump=float(case["gamma_jump"]),
            delta_s_hor=float(case["delta_s_hor"]),
            alpha_state=float(case["alpha_state"]),
            n_steps=g2_steps,
            n_traj=g2_traj,
            seed=seed + 10000 + case_id,
        )
        fit_row = fit.iloc[0]

        state90 = state_df[np.isclose(state_df["angle_deg"], 90.0)]
        spread90 = float(state90["tr_F_rho_real"].max() - state90["tr_F_rho_real"].min())

        lm_raw = float(fit_row["a_1_over_Teff"])
        lm_quality = float(np.clip(fit_row["R2"], 0.0, 1.0))
        lm_stable = float(np.sign(lm_raw) * np.log1p(abs(lm_raw)) * lm_quality)

        row = {
            "case_id": int(case_id),
            "profile": profile,
            "alpha1": float(case["alpha1"]),
            "alpha2": float(case["alpha2"]),
            "delta_alpha": float(case["delta_alpha"]),
            "lambda_matter_w1_mean": lm_w1,
            "lm_w1_times_delta_alpha": float(lm_w1 * float(case["delta_alpha"])),
            "lambda_vac_w1_mean": float(spatial_df["Lambda_vac"].mean()),
            "state_spread_phi90": spread90,
            "lambda_w_g2_mean": float(fit_row["mean_Lambda_w"]),
            "coh_ptr_g2_mean": float(fit_row["mean_Coh_ptr_w"]),
            "jump_rate_g2_mean": float(fit_row["mean_jump_rate"]),
            "lambda_eff_raw": lm_raw,
            "clausius_r2": lm_quality,
            "lambda_eff_proxy": lm_stable,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)

    x = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["lambda_eff_proxy"].to_numpy(dtype=float)
    labels = df["profile"].to_numpy(dtype=str)

    splits = _stratified_splits(
        labels=labels,
        train_frac=train_frac,
        n_splits=n_splits,
        seed=seed + 77,
    )
    train_r2_arr, test_r2_arr, coef_splits, intercept_splits = _evaluate_splits(
        x=x,
        y=y,
        splits=splits,
        ridge_alpha=ridge_alpha,
    )

    full_coef, full_intercept, yhat_full = _fit_ridge_scaled(
        x_train=x,
        y_train=y,
        x_eval=x,
        ridge_alpha=ridge_alpha,
    )

    coef_df = pd.DataFrame(
        [
            {"term": "intercept", "value": float(full_intercept)},
            *(
                {
                    "term": f,
                    "value": float(v),
                }
                for f, v in zip(FEATURE_COLS, full_coef)
            ),
        ]
    )

    split_rows = []
    for i, ((train_idx, test_idx), coef, intercept, train_r2, test_r2) in enumerate(
        zip(splits, coef_splits, intercept_splits, train_r2_arr, test_r2_arr)
    ):
        row = {
            "split_id": int(i),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "intercept": float(intercept),
        }
        for f, v in zip(FEATURE_COLS, coef):
            row[f"coef_{f}"] = float(v)
        split_rows.append(row)
    splits_df = pd.DataFrame(split_rows)

    p_perm, perm_df = _permutation_test(
        x=x,
        y=y,
        splits=splits,
        ridge_alpha=ridge_alpha,
        n_perm=n_perm,
        seed=seed + 1234,
    )

    boot_df, boot_sign = _bootstrap_coefficients(
        x=x,
        y=y,
        feature_names=FEATURE_COLS,
        n_boot=n_boot,
        ridge_alpha=ridge_alpha,
        seed=seed + 4321,
    )
    prof_coef_df, prof_sign = _profile_coefficients(
        df=df,
        feature_names=FEATURE_COLS,
        ridge_alpha=ridge_alpha,
    )

    min_boot_sign = float(min(boot_sign.get(k, 0.0) for k in FEATURE_COLS))
    mean_boot_sign = float(np.mean([boot_sign.get(k, 0.0) for k in FEATURE_COLS]))
    min_prof_sign = float(
        np.nanmin([prof_sign.get(k, np.nan) for k in FEATURE_COLS]) if len(prof_coef_df) > 0 else np.nan
    )
    mean_prof_sign = float(
        np.nanmean([prof_sign.get(k, np.nan) for k in FEATURE_COLS]) if len(prof_coef_df) > 0 else np.nan
    )
    pass_prof_sign = bool(np.isnan(mean_prof_sign) or mean_prof_sign >= min_profile_sign_consistency)

    test_r2_median = float(np.median(test_r2_arr))
    test_r2_p25 = float(np.quantile(test_r2_arr, 0.25))
    test_r2_min = float(np.min(test_r2_arr))

    summary = pd.DataFrame(
        [
            {
                "seed": int(seed),
                "n_cases": int(n_cases),
                "train_frac": float(train_frac),
                "n_splits": int(n_splits),
                "n_perm": int(n_perm),
                "n_boot": int(n_boot),
                "g2_steps": int(g2_steps),
                "g2_traj": int(g2_traj),
                "g2_layers": int(g2_layers),
                "ridge_alpha": float(ridge_alpha),
                "train_r2_mean": float(np.mean(train_r2_arr)),
                "train_r2_median": float(np.median(train_r2_arr)),
                "test_r2_mean": float(np.mean(test_r2_arr)),
                "test_r2_median": float(test_r2_median),
                "test_r2_p25": float(test_r2_p25),
                "test_r2_min": float(test_r2_min),
                "full_r2": float(_r2(y, yhat_full)),
                "perm_p_value": float(p_perm),
                "perm_stat_p95": float(perm_df["stat_perm_median_test_r2"].quantile(0.95)),
                "boot_min_sign_consistency_key": float(min_boot_sign),
                "boot_mean_sign_consistency_key": float(mean_boot_sign),
                "profile_min_sign_consistency_key": float(min_prof_sign) if not np.isnan(min_prof_sign) else np.nan,
                "profile_mean_sign_consistency_key": float(mean_prof_sign) if not np.isnan(mean_prof_sign) else np.nan,
                "pass_test_r2": bool((test_r2_median >= min_test_r2_median) and (test_r2_p25 >= min_test_r2_p25)),
                "pass_perm_p": bool(p_perm <= max_perm_p),
                "pass_boot_sign": bool(mean_boot_sign >= min_sign_consistency),
                "pass_profile_sign": bool(pass_prof_sign),
                "pass_all": bool(
                    (test_r2_median >= min_test_r2_median)
                    and (test_r2_p25 >= min_test_r2_p25)
                    and (p_perm <= max_perm_p)
                    and (mean_boot_sign >= min_sign_consistency)
                    and pass_prof_sign
                ),
            }
        ]
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "experiment_K_dataset.csv", index=False)
    coef_df.to_csv(outdir / "experiment_K_coefficients.csv", index=False)
    splits_df.to_csv(outdir / "experiment_K_splits.csv", index=False)
    perm_df.to_csv(outdir / "experiment_K_permutation.csv", index=False)
    boot_df.to_csv(outdir / "experiment_K_bootstrap_coefficients.csv", index=False)
    prof_coef_df.to_csv(outdir / "experiment_K_profile_coefficients.csv", index=False)
    summary.to_csv(outdir / "experiment_K_summary.csv", index=False)

    return df, coef_df, splits_df, perm_df, boot_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/experiment_K_lambda_bridge", help="output directory")
    parser.add_argument("--seed", type=int, default=20260225)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--n-perm", type=int, default=140)
    parser.add_argument("--n-boot", type=int, default=220)
    parser.add_argument("--g2-steps", type=int, default=90)
    parser.add_argument("--g2-traj", type=int, default=360)
    parser.add_argument("--g2-layers", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=0.03)
    parser.add_argument("--min-test-r2-median", type=float, default=0.25)
    parser.add_argument("--min-test-r2-p25", type=float, default=0.00)
    parser.add_argument("--max-perm-p", type=float, default=0.05)
    parser.add_argument("--min-sign-consistency", type=float, default=0.65)
    parser.add_argument("--min-profile-sign-consistency", type=float, default=float(2.0 / 3.0))
    args = parser.parse_args()

    _, coef_df, splits_df, _, _, summary = run_experiment(
        outdir=Path(args.out),
        seed=args.seed,
        n_cases=args.cases,
        train_frac=args.train_frac,
        n_splits=args.splits,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        g2_steps=args.g2_steps,
        g2_traj=args.g2_traj,
        g2_layers=args.g2_layers,
        ridge_alpha=args.ridge_alpha,
        min_test_r2_median=args.min_test_r2_median,
        min_test_r2_p25=args.min_test_r2_p25,
        max_perm_p=args.max_perm_p,
        min_sign_consistency=args.min_sign_consistency,
        min_profile_sign_consistency=args.min_profile_sign_consistency,
    )

    print("Summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nCoefficients (full fit):")
    print(coef_df.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("\nSplit diagnostics:")
    print(
        splits_df[["split_id", "train_r2", "test_r2"]].to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
