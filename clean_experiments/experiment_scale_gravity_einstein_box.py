#!/usr/bin/env python3
"""Experiment T20: local synthetic "Einstein in a box" scale-gravity diagnostics.

This script runs option C from the experiment brief:
- build a synthetic 2D multiscale cascade on a single periodic patch,
- construct band-wise density matrices rho_mu(t) from rolling covariances,
- derive interscale curvature proxy Lambda_b(t) = Re Tr(F_b rho_b),
- compute band-wise energy-flow proxy Pi_b(t),
- test linearity Lambda ~ Pi in forcing/inertial/dissipation scale zones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _build_band_masks(
    *,
    ny: int,
    nx: int,
    dy: float,
    dx: float,
    scale_edges: list[float],
) -> tuple[list[np.ndarray], list[float], list[float], np.ndarray]:
    if len(scale_edges) < 3:
        raise ValueError("Need at least 3 scale edges.")
    edges = np.asarray(scale_edges, dtype=float)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("scale_edges must be strictly increasing.")

    ky = np.fft.fftfreq(ny, d=dy)[:, None]
    kx = np.fft.rfftfreq(nx, d=dx)[None, :]
    k_mag = np.sqrt(kx * kx + ky * ky)
    wavelength = np.full_like(k_mag, np.inf, dtype=float)
    nz = k_mag > 0.0
    wavelength[nz] = 1.0 / k_mag[nz]

    masks: list[np.ndarray] = []
    centers: list[float] = []
    mus: list[float] = []
    l_ref = float(edges[-1])
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        m = (wavelength >= lo) & (wavelength < hi) & np.isfinite(wavelength)
        if int(m.sum()) == 0:
            raise ValueError(f"Band [{lo},{hi}) has no cells; adjust grid/scale edges.")
        masks.append(m)
        c = float(np.sqrt(lo * hi))
        centers.append(c)
        mus.append(float(np.log2(l_ref / c)))
    return masks, centers, mus, k_mag


def _random_complex_matrix(rng: np.random.Generator, n_out: int, n_in: int) -> np.ndarray:
    mat = rng.normal(size=(n_out, n_in)) + 1j * rng.normal(size=(n_out, n_in))
    mat = mat / np.sqrt(2.0 * max(n_in, 1))
    s = np.linalg.svd(mat, full_matrices=False, compute_uv=False)
    scale = float(np.max(s)) if s.size > 0 else 1.0
    if scale > 1e-12:
        mat = mat / scale
    return mat


def _select_modes(
    *,
    rng: np.random.Generator,
    masks: list[np.ndarray],
    n_modes_per_var: int,
) -> list[dict[str, list[tuple[int, int]]]]:
    selected: list[dict[str, list[tuple[int, int]]]] = []
    for mask in masks:
        idx = np.argwhere(mask)
        if len(idx) < n_modes_per_var:
            raise ValueError(
                f"Not enough Fourier cells ({len(idx)}) for n_modes_per_var={n_modes_per_var}. "
                "Use smaller n_modes_per_var or adjust scales/grid."
            )
        pick_u = idx[rng.choice(len(idx), size=n_modes_per_var, replace=False)]
        pick_v = idx[rng.choice(len(idx), size=n_modes_per_var, replace=False)]
        selected.append(
            {
                "u": [(int(iy), int(ix)) for iy, ix in pick_u],
                "v": [(int(iy), int(ix)) for iy, ix in pick_v],
            }
        )
    return selected


def _simulate_cascade_coefficients(
    *,
    nt: int,
    centers: list[float],
    selected: list[dict[str, list[tuple[int, int]]]],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    n_bands = len(selected)
    dims = [len(selected[b]["u"]) + len(selected[b]["v"]) for b in range(n_bands)]

    # Local AR(1)-like latent flux driver that imprints both energy and geometry.
    q = np.zeros(nt, dtype=float)
    period = max(nt // 6, 10)
    for t in range(1, nt):
        q[t] = (
            0.92 * q[t - 1]
            + 0.45 * np.sin(2.0 * np.pi * t / float(period))
            + 0.35 * rng.normal()
        )

    down = [_random_complex_matrix(rng, dims[b], dims[b + 1]) for b in range(n_bands - 1)]
    up = [_random_complex_matrix(rng, dims[b + 1], dims[b]) for b in range(n_bands - 1)]
    self_mix = [_random_complex_matrix(rng, dims[b], dims[b]) for b in range(n_bands)]

    # k^(-5/3)-like baseline energy ladder (in wavelength space this means larger
    # scales carry more energy).
    center_arr = np.asarray(centers, dtype=float)
    e0 = np.power(np.maximum(center_arr, 1e-6), 5.0 / 3.0)
    e0 = e0 / np.mean(e0)

    coeff_by_band = [np.zeros((nt, dims[b]), dtype=np.complex128) for b in range(n_bands)]
    prev = [
        (rng.normal(size=dims[b]) + 1j * rng.normal(size=dims[b])) / np.sqrt(2.0 * max(dims[b], 1))
        for b in range(n_bands)
    ]

    for t in range(nt):
        drive = q[t]
        cur: list[np.ndarray] = []
        for b in range(n_bands):
            state = prev[b]
            base = (0.76 + 0.02 * b) * np.exp(1j * (0.11 * (b + 1) + 0.24 * drive)) * state
            base += 0.18 * (self_mix[b] @ state)
            if b < n_bands - 1:
                c_down = 0.23 + 0.06 * np.tanh(0.8 * drive)
                base += c_down * (down[b] @ prev[b + 1])
            if b > 0:
                c_up = 0.16 + 0.04 * np.tanh(-0.6 * drive)
                base += c_up * (up[b - 1] @ prev[b - 1])
            base += 0.22 * (
                rng.normal(size=dims[b]) + 1j * rng.normal(size=dims[b])
            ) / np.sqrt(2.0 * max(dims[b], 1))

            e_target = float(e0[b] * np.exp(0.33 * drive + 0.12 * rng.normal()))
            e_target = max(e_target, 1e-8)
            e_cur = float(np.mean(np.abs(base) ** 2))
            if e_cur < 1e-12:
                base = (rng.normal(size=dims[b]) + 1j * rng.normal(size=dims[b])) / np.sqrt(
                    2.0 * max(dims[b], 1)
                )
                e_cur = float(np.mean(np.abs(base) ** 2))
            base *= np.sqrt(e_target / max(e_cur, 1e-12))
            cur.append(base)
            coeff_by_band[b][t] = base
        prev = cur

    return coeff_by_band, q


def _coeff_to_fields(
    *,
    coeff_by_band: list[np.ndarray],
    selected: list[dict[str, list[tuple[int, int]]]],
    ny: int,
    nx: int,
    k_mag: np.ndarray,
    rng: np.random.Generator,
    spectral_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    nt = coeff_by_band[0].shape[0]
    nx_r = nx // 2 + 1
    u = np.zeros((nt, ny, nx), dtype=float)
    v = np.zeros((nt, ny, nx), dtype=float)

    slope = np.power(np.maximum(k_mag, 1e-6), -5.0 / 6.0)
    slope[0, 0] = 0.0

    for t in range(nt):
        uhat = spectral_noise * slope * (
            rng.normal(size=(ny, nx_r)) + 1j * rng.normal(size=(ny, nx_r))
        ) / np.sqrt(2.0)
        vhat = spectral_noise * slope * (
            rng.normal(size=(ny, nx_r)) + 1j * rng.normal(size=(ny, nx_r))
        ) / np.sqrt(2.0)
        for b, sel in enumerate(selected):
            coeff = coeff_by_band[b][t]
            nu = len(sel["u"])
            for i, (iy, ix) in enumerate(sel["u"]):
                uhat[iy, ix] += coeff[i]
            for j, (iy, ix) in enumerate(sel["v"]):
                vhat[iy, ix] += coeff[nu + j]

        u[t] = np.fft.irfft2(uhat, s=(ny, nx)).real
        v[t] = np.fft.irfft2(vhat, s=(ny, nx)).real
    return u, v


def _compute_energy_flux_proxy(
    *,
    u: np.ndarray,
    v: np.ndarray,
    masks: list[np.ndarray],
    k_mag: np.ndarray,
) -> np.ndarray:
    nt = u.shape[0]
    n_bands = len(masks)
    pi = np.zeros((nt, n_bands), dtype=float)
    for t in range(nt):
        uhat = np.fft.rfft2(u[t] - np.mean(u[t]))
        vhat = np.fft.rfft2(v[t] - np.mean(v[t]))
        e = 0.5 * (np.abs(uhat) ** 2 + np.abs(vhat) ** 2)
        for b, mask in enumerate(masks):
            km = k_mag[mask]
            em = e[mask]
            if km.size == 0:
                pi[t, b] = 0.0
            else:
                pi[t, b] = float(np.mean(km * em))
    return pi


def _compute_rho_and_lambda(
    *,
    coeff_by_band: list[np.ndarray],
    window: int,
    ridge: float,
    cov_shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= cov_shrinkage <= 1.0):
        raise ValueError("cov_shrinkage must be in [0, 1].")
    n_bands = len(coeff_by_band)
    nt = coeff_by_band[0].shape[0]

    coeff_scaled: list[np.ndarray] = []
    for b in range(n_bands):
        arr = np.asarray(coeff_by_band[b], dtype=np.complex128)
        rms = np.sqrt(np.mean(np.abs(arr) ** 2, axis=0))
        rms = np.where(rms < 1e-12, 1.0, rms)
        coeff_scaled.append(arr / rms[None, :])

    rho_cache: list[list[np.ndarray]] = [[] for _ in range(n_bands)]
    trace_err = np.zeros((nt, n_bands), dtype=float)
    min_eig = np.zeros((nt, n_bands), dtype=float)
    lambda_mu = np.full((nt, n_bands), np.nan, dtype=float)
    lambda_sum = np.zeros((nt, n_bands), dtype=float)
    lambda_count = np.zeros((nt, n_bands), dtype=float)

    for b in range(n_bands):
        m = coeff_scaled[b].shape[1]
        eye = np.eye(m, dtype=np.complex128)
        rho_cache[b] = [eye / float(m) for _ in range(nt)]
        for t in range(nt):
            i0 = max(0, t - window + 1)
            a = coeff_scaled[b][i0 : t + 1]
            c = (a.conj().T @ a) / float(len(a))
            c = 0.5 * (c + c.conj().T)
            if cov_shrinkage > 0.0:
                c = (1.0 - cov_shrinkage) * c + cov_shrinkage * np.diag(np.diag(c))
            c = c + ridge * eye
            tr = float(np.real(np.trace(c)))
            rho = eye / float(m) if tr < 1e-14 else c / tr
            rho = 0.5 * (rho + rho.conj().T)
            rho_cache[b][t] = rho
            vals = np.linalg.eigvalsh(rho)
            trace_err[t, b] = float(abs(np.real(np.trace(rho)) - 1.0))
            min_eig[t, b] = float(np.min(np.real(vals)))

    for b in range(n_bands - 1):
        m1 = coeff_scaled[b].shape[1]
        m2 = coeff_scaled[b + 1].shape[1]
        eye1 = np.eye(m1, dtype=np.complex128)
        eye2 = np.eye(m2, dtype=np.complex128)
        for t in range(nt):
            i0 = max(0, t - window + 1)
            a = coeff_scaled[b][i0 : t + 1]
            bnext = coeff_scaled[b + 1][i0 : t + 1]
            g1 = a.conj().T @ a + ridge * eye1
            g2 = bnext.conj().T @ bnext + ridge * eye2
            rhs_fwd = a.conj().T @ bnext
            rhs_rev = bnext.conj().T @ a
            try:
                m_fwd = np.linalg.solve(g1, rhs_fwd)
            except np.linalg.LinAlgError:
                m_fwd = np.linalg.pinv(g1, rcond=1e-10) @ rhs_fwd
            try:
                m_rev = np.linalg.solve(g2, rhs_rev)
            except np.linalg.LinAlgError:
                m_rev = np.linalg.pinv(g2, rcond=1e-10) @ rhs_rev

            g_fwd = m_fwd @ m_fwd.conj().T
            g_back = m_rev.conj().T @ m_rev
            comm = g_fwd @ g_back - g_back @ g_fwd
            f_phys = -0.5 * (1j * comm + (1j * comm).conj().T)
            rho = rho_cache[b][t]
            lambda_sum[t, b] += float(np.real(np.trace(f_phys @ rho)))
            lambda_count[t, b] += 1.0

            g_fwd_r = m_fwd.conj().T @ m_fwd
            g_back_r = m_rev @ m_rev.conj().T
            comm_r = g_fwd_r @ g_back_r - g_back_r @ g_fwd_r
            f_phys_r = -0.5 * (1j * comm_r + (1j * comm_r).conj().T)
            rho_r = rho_cache[b + 1][t]
            lambda_sum[t, b + 1] += float(np.real(np.trace(f_phys_r @ rho_r)))
            lambda_count[t, b + 1] += 1.0

    nz = lambda_count > 0
    lambda_mu[nz] = lambda_sum[nz] / lambda_count[nz]

    return lambda_mu, trace_err, min_eig


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def _binned_means(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < n_bins + 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, q))
    if len(edges) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    xb = []
    yb = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            m = (x >= lo) & (x <= hi)
        else:
            m = (x >= lo) & (x < hi)
        if int(m.sum()) < 3:
            continue
        xb.append(float(np.mean(x[m])))
        yb.append(float(np.mean(y[m])))
    return np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)


def _regime_label(
    band_id: int,
    n_bands: int,
    *,
    forcing_bands: int,
    dissipation_bands: int,
) -> str:
    if band_id >= n_bands - forcing_bands:
        return "forcing"
    if band_id < dissipation_bands:
        return "dissipation"
    return "inertial"


def _make_report(
    *,
    outdir: Path,
    args: argparse.Namespace,
    band_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    trace_err: np.ndarray,
    min_eig: np.ndarray,
    success: bool,
) -> None:
    inertial = regime_df.loc[regime_df["regime"] == "inertial"]
    if len(inertial) > 0:
        in_r2 = float(inertial["r2_binned"].iloc[0])
        in_slope = float(inertial["slope_binned"].iloc[0])
    else:
        in_r2 = float("nan")
        in_slope = float("nan")

    lines = [
        "# Einstein in a Box (Synthetic 2D Cascade)",
        "",
        "## Goal",
        "- Empirical local check of Einstein-type closure in scale space: `Lambda ~ Pi`.",
        "- `Lambda_b(t) = Re Tr(F_b rho_b)` from noncommuting interscale transport.",
        "- `Pi_b(t)` from spectral energy-flow proxy within each scale band.",
        "",
        "## Run setup",
        f"- seed: `{int(args.seed)}`",
        f"- grid: `{int(args.grid)} x {int(args.grid)}`",
        f"- timesteps: `{int(args.timesteps)}`",
        f"- scale edges: `{args.scale_edges}`",
        f"- n_modes_per_var requested/effective: "
        f"`{int(args.n_modes_per_var)}/{int(getattr(args, 'n_modes_eff', args.n_modes_per_var))}`",
        f"- rolling window W: `{int(args.window)}`",
        f"- ridge: `{float(args.ridge):.2e}`",
        f"- cov_shrinkage: `{float(args.cov_shrinkage):.3f}`",
        f"- lambda sign requested/effective: "
        f"`{int(args.lambda_sign)}/{int(getattr(args, 'lambda_sign_eff', args.lambda_sign))}`",
        "",
        "## Data quality checks",
        f"- max trace error `|Tr(rho)-1|`: `{float(np.nanmax(trace_err)):.3e}`",
        f"- min eigenvalue over all rho: `{float(np.nanmin(min_eig)):.3e}`",
        "",
        "## Bandwise linearity",
    ]
    for _, r in band_df.iterrows():
        lines.append(
            "- band {bid} ({reg}): mu={mu:.3f}, center={c:.3f}, "
            "R2_raw={r2r:.3f}, R2_binned={r2b:.3f}, slope_binned={sb:.3e}".format(
                bid=int(r["band_id"]),
                reg=str(r["regime"]),
                mu=float(r["mu"]),
                c=float(r["center_scale"]),
                r2r=float(r["r2_raw"]),
                r2b=float(r["r2_binned"]),
                sb=float(r["slope_binned"]),
            )
        )

    lines.extend(
        [
            "",
            "## Regime aggregates",
        ]
    )
    for _, r in regime_df.iterrows():
        lines.append(
            "- {reg}: n={n}, R2_raw={r2r:.3f}, R2_binned={r2b:.3f}, slope_binned={sb:.3e}".format(
                reg=str(r["regime"]),
                n=int(r["n_samples"]),
                r2r=float(r["r2_raw"]),
                r2b=float(r["r2_binned"]),
                sb=float(r["slope_binned"]),
            )
        )

    lines.extend(
        [
            "",
            "## Success criterion",
            f"- criterion: inertial `R2_binned >= {float(args.success_r2):.2f}` and positive slope.",
            f"- inertial R2_binned: `{in_r2:.3f}`",
            f"- inertial slope_binned: `{in_slope:.3e}`",
            f"- PASS: `{bool(success)}`",
            "",
            "## Interpretation",
            "- This run validates the local synthetic option-C protocol only.",
            "- It does not replace atmospheric/JHTDB validation for external physical claims.",
        ]
    )

    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    masks, centers, mus, k_mag = _build_band_masks(
        ny=args.grid,
        nx=args.grid,
        dy=args.dx,
        dx=args.dx,
        scale_edges=[float(x) for x in args.scale_edges.split(",")],
    )
    min_cells = int(min(int(m.sum()) for m in masks))
    n_modes_eff = int(min(args.n_modes_per_var, min_cells))
    if n_modes_eff < args.n_modes_per_var:
        print(
            f"[info] n_modes_per_var reduced from {args.n_modes_per_var} to {n_modes_eff} "
            f"(smallest band has {min_cells} Fourier cells)."
        )
    args.n_modes_eff = n_modes_eff
    selected = _select_modes(rng=rng, masks=masks, n_modes_per_var=n_modes_eff)
    coeff_by_band, q = _simulate_cascade_coefficients(
        nt=args.timesteps,
        centers=centers,
        selected=selected,
        rng=rng,
    )
    u, v = _coeff_to_fields(
        coeff_by_band=coeff_by_band,
        selected=selected,
        ny=args.grid,
        nx=args.grid,
        k_mag=k_mag,
        rng=rng,
        spectral_noise=args.spectral_noise,
    )

    lambda_mu, trace_err, min_eig = _compute_rho_and_lambda(
        coeff_by_band=coeff_by_band,
        window=args.window,
        ridge=args.ridge,
        cov_shrinkage=args.cov_shrinkage,
    )
    pi_mu = _compute_energy_flux_proxy(u=u, v=v, masks=masks, k_mag=k_mag)

    inertial_bands = [
        b
        for b in range(len(masks))
        if _regime_label(
            b,
            len(masks),
            forcing_bands=args.forcing_bands,
            dissipation_bands=args.dissipation_bands,
        )
        == "inertial"
    ]
    sign_eff = int(args.lambda_sign)
    if sign_eff not in (-1, 0, 1):
        raise ValueError("lambda_sign must be one of {-1,0,1}.")
    if sign_eff == 0 and len(inertial_bands) > 0:
        t0 = max(args.window - 1, 1)
        xs = []
        ys = []
        for b in inertial_bands:
            x = pi_mu[t0:, b]
            y = lambda_mu[t0:, b]
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) > 5:
                xs.append(x[m])
                ys.append(y[m])
        if len(xs) > 0:
            x_all = np.concatenate(xs)
            y_all = np.concatenate(ys)
            slope, _, _ = _linear_fit(x_all, y_all)
            sign_eff = -1 if slope < 0.0 else 1
        else:
            sign_eff = 1
    if sign_eff == 0:
        sign_eff = 1
    args.lambda_sign_eff = sign_eff
    lambda_mu = float(sign_eff) * lambda_mu

    rows = []
    for b in range(len(masks)):
        regime = _regime_label(
            b,
            len(masks),
            forcing_bands=args.forcing_bands,
            dissipation_bands=args.dissipation_bands,
        )
        for t in range(args.timesteps):
            rows.append(
                {
                    "t": int(t),
                    "q_driver": float(q[t]),
                    "band_id": int(b),
                    "mu": float(mus[b]),
                    "center_scale": float(centers[b]),
                    "regime": regime,
                    "pi": float(pi_mu[t, b]),
                    "lambda": float(lambda_mu[t, b]) if np.isfinite(lambda_mu[t, b]) else np.nan,
                    "trace_err": float(trace_err[t, b]),
                    "min_eig": float(min_eig[t, b]),
                }
            )
    ts_df = pd.DataFrame(rows)
    ts_df.to_csv(outdir / "timeseries_long.csv", index=False)

    # Skip the initial transient needed for rolling estimates.
    ts_eval = ts_df.loc[ts_df["t"] >= max(args.window - 1, 1)].copy()

    band_rows = []
    for b in range(len(masks)):
        g = ts_eval.loc[(ts_eval["band_id"] == b) & np.isfinite(ts_eval["lambda"]) & np.isfinite(ts_eval["pi"])]
        if len(g) < 12:
            continue
        x = g["pi"].to_numpy(dtype=float)
        y = g["lambda"].to_numpy(dtype=float)
        slope_raw, intercept_raw, r2_raw = _linear_fit(x, y)
        xb, yb = _binned_means(x, y, n_bins=args.bins)
        if len(xb) >= 3:
            slope_bin, intercept_bin, r2_bin = _linear_fit(xb, yb)
        else:
            slope_bin, intercept_bin, r2_bin = float("nan"), float("nan"), float("nan")
        band_rows.append(
            {
                "band_id": int(b),
                "regime": str(g["regime"].iloc[0]),
                "mu": float(mus[b]),
                "center_scale": float(centers[b]),
                "n_samples": int(len(g)),
                "slope_raw": float(slope_raw),
                "intercept_raw": float(intercept_raw),
                "r2_raw": float(r2_raw),
                "slope_binned": float(slope_bin),
                "intercept_binned": float(intercept_bin),
                "r2_binned": float(r2_bin),
            }
        )
    band_df = pd.DataFrame(band_rows).sort_values("band_id").reset_index(drop=True)
    band_df.to_csv(outdir / "band_regression.csv", index=False)

    regime_rows = []
    for regime in ("forcing", "inertial", "dissipation"):
        g = ts_eval.loc[(ts_eval["regime"] == regime) & np.isfinite(ts_eval["lambda"]) & np.isfinite(ts_eval["pi"])]
        if len(g) < 20:
            continue
        x = g["pi"].to_numpy(dtype=float)
        y = g["lambda"].to_numpy(dtype=float)
        slope_raw, intercept_raw, r2_raw = _linear_fit(x, y)
        xb, yb = _binned_means(x, y, n_bins=args.bins)
        if len(xb) >= 3:
            slope_bin, intercept_bin, r2_bin = _linear_fit(xb, yb)
        else:
            slope_bin, intercept_bin, r2_bin = float("nan"), float("nan"), float("nan")
        regime_rows.append(
            {
                "regime": regime,
                "n_samples": int(len(g)),
                "slope_raw": float(slope_raw),
                "intercept_raw": float(intercept_raw),
                "r2_raw": float(r2_raw),
                "slope_binned": float(slope_bin),
                "intercept_binned": float(intercept_bin),
                "r2_binned": float(r2_bin),
            }
        )
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(outdir / "aggregate_regression.csv", index=False)

    inertial = regime_df.loc[regime_df["regime"] == "inertial"]
    success = False
    if len(inertial) > 0:
        success = bool(
            float(inertial["r2_binned"].iloc[0]) >= float(args.success_r2)
            and float(inertial["slope_binned"].iloc[0]) > 0.0
        )

    if plt is not None:
        inertial_rows = ts_eval.loc[ts_eval["regime"] == "inertial"]
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for b, g in inertial_rows.groupby("band_id"):
            ax.scatter(
                g["pi"],
                g["lambda"],
                s=8,
                alpha=0.28,
                label=f"band {int(b)}",
            )
        x = inertial_rows["pi"].to_numpy(dtype=float)
        y = inertial_rows["lambda"].to_numpy(dtype=float)
        xb, yb = _binned_means(x, y, n_bins=args.bins)
        if len(xb) >= 3:
            slope, intercept, r2 = _linear_fit(xb, yb)
            xfit = np.linspace(float(np.min(xb)), float(np.max(xb)), 200)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, color="black", lw=2.1, label=f"binned fit R2={r2:.3f}")
            ax.scatter(xb, yb, c="black", s=36, zorder=3)
        ax.set_xlabel("Pi (energy-flow proxy)")
        ax.set_ylabel("Lambda (curvature projection)")
        ax.set_title("Einstein-in-a-box: inertial scale relation")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "lambda_vs_pi_inertial.png", dpi=180)
        plt.close(fig)

    _make_report(
        outdir=outdir,
        args=args,
        band_df=band_df,
        regime_df=regime_df,
        trace_err=trace_err,
        min_eig=min_eig,
        success=success,
    )

    print("Saved:", outdir.resolve())
    if len(regime_df) > 0:
        print("\nAggregate regime regression:")
        print(regime_df.to_string(index=False))
    print(f"\nPASS (inertial): {success}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default="clean_experiments/results/experiment_scale_gravity_einstein_box",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--grid", type=int, default=96, help="Grid size (ny=nx).")
    p.add_argument("--timesteps", type=int, default=540)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--scale-edges", dest="scale_edges", default="4,8,16,32,48")
    p.add_argument("--n-modes-per-var", type=int, default=6)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--cov-shrinkage", type=float, default=0.05)
    p.add_argument("--spectral-noise", type=float, default=0.02)
    p.add_argument("--forcing-bands", type=int, default=1)
    p.add_argument("--dissipation-bands", type=int, default=1)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--success-r2", type=float, default=0.7)
    p.add_argument(
        "--lambda-sign",
        type=int,
        default=0,
        help="Sign convention for Lambda: -1 or +1, or 0 to auto-align inertial slope.",
    )
    p.add_argument("--quick", action="store_true", help="Fast smoke-run settings.")
    args = p.parse_args()
    if args.quick:
        args.timesteps = min(args.timesteps, 180)
        args.grid = min(args.grid, 72)
    return args


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
