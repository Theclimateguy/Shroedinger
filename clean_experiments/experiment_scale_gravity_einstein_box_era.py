#!/usr/bin/env python3
"""Experiment A15: ERA5 box test for Einstein-type scale-gravity closure.

Protocol:
1) load a local atmospheric patch (u,v or ivt_u,ivt_v),
2) build scale bands in km,
3) compute band-wise Pi(t) from kinetic spectral energy proxy,
4) compute rho_mu(t), interscale transport curvature, and Lambda_mu(t),
5) evaluate binned conditional mean Lambda ~ Pi by regimes:
   forcing (>1000 km), inertial (100..1000 km), dissipation (<100 km).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from clean_experiments.experiment_M_cosmo_flow import _build_band_masks, _select_mode_indices, _xy_coordinates_m
except ImportError:
    from experiment_M_cosmo_flow import _build_band_masks, _select_mode_indices, _xy_coordinates_m  # type: ignore


VAR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "u": ("u", "u10", "u850", "u1000"),
    "v": ("v", "v10", "v850", "v1000"),
    "ivt_u": ("ivt_u", "viwve"),
    "ivt_v": ("ivt_v", "viwvn"),
}


def _infer_dim_name(dims: tuple[str, ...], candidates: tuple[str, ...], fallback: str | None = None) -> str:
    for c in candidates:
        if c in dims:
            return c
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not infer dimension name from candidates {candidates}; dims={dims}")


def _to_tyx_da(da):
    for d in list(da.dims):
        if da.sizes[d] == 1:
            da = da.isel({d: 0})

    dims = tuple(da.dims)
    if len(dims) != 3:
        raise ValueError(f"Expected 3D variable (time,lat,lon), got dims={dims}")

    t_dim = _infer_dim_name(dims, ("time", "valid_time", "datetime", "date"), fallback=dims[0])
    y_dim = _infer_dim_name(dims, ("lat", "latitude", "y", "rlat"), fallback=dims[-2])
    x_dim = _infer_dim_name(dims, ("lon", "longitude", "x", "rlon"), fallback=dims[-1])
    da = da.transpose(t_dim, y_dim, x_dim)
    return da, t_dim, y_dim, x_dim


def _find_var_name(available: list[str], role: str, override: str | None) -> str:
    if override is not None:
        if override not in available:
            raise KeyError(f"Variable '{override}' requested for role '{role}' not found.")
        return override
    for cand in VAR_CANDIDATES[role]:
        if cand in available:
            return cand
    raise KeyError(f"No variable found for role '{role}'. Tried {VAR_CANDIDATES[role]}.")


def _slice_tyx(
    arr: np.ndarray,
    *,
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    time_start: int,
    max_time: int | None,
) -> np.ndarray:
    arr = arr[::time_stride, ::lat_stride, ::lon_stride]
    if time_start > 0:
        arr = arr[time_start:]
    if max_time is not None:
        arr = arr[:max_time]
    return np.asarray(arr, dtype=float)


def _center_crop(arr: np.ndarray, ny: int | None, nx: int | None) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected (time,lat,lon), got {arr.shape}")
    _, y, x = arr.shape
    y_new = y if ny is None or ny <= 0 or ny >= y else int(ny)
    x_new = x if nx is None or nx <= 0 or nx >= x else int(nx)
    y0 = (y - y_new) // 2
    x0 = (x - x_new) // 2
    return arr[:, y0 : y0 + y_new, x0 : x0 + x_new]


def _center_crop_coords(vec: np.ndarray, n_new: int | None) -> np.ndarray:
    n = len(vec)
    if n_new is None or n_new <= 0 or n_new >= n:
        return np.asarray(vec, dtype=float)
    n_new = int(n_new)
    i0 = (n - n_new) // 2
    return np.asarray(vec[i0 : i0 + n_new], dtype=float)


def _load_vector_fields(
    *,
    input_path: Path,
    field_set: str,
    u_var: str | None,
    v_var: str | None,
    time_stride: int,
    lat_stride: int,
    lon_stride: int,
    time_start: int,
    max_time: int | None,
    crop_ny: int | None,
    crop_nx: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    if xr is None:
        raise ImportError("xarray is required for NetCDF input.")
    ds = xr.open_dataset(input_path)
    available = list(ds.data_vars)

    if field_set == "wind":
        role_u = "u"
        role_v = "v"
    elif field_set == "ivt":
        role_u = "ivt_u"
        role_v = "ivt_v"
    else:
        raise ValueError("field_set must be one of {'wind','ivt'}.")

    name_u = _find_var_name(available, role_u, u_var)
    name_v = _find_var_name(available, role_v, v_var)

    da_u, t_dim, y_dim, x_dim = _to_tyx_da(ds[name_u])
    da_v, _, _, _ = _to_tyx_da(ds[name_v])
    time_all = np.asarray(da_u[t_dim].values)
    lat_all = np.asarray(da_u[y_dim].values, dtype=float)
    lon_all = np.asarray(da_u[x_dim].values, dtype=float)

    u = _slice_tyx(
        da_u.values,
        time_stride=time_stride,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        time_start=time_start,
        max_time=max_time,
    )
    v = _slice_tyx(
        da_v.values,
        time_stride=time_stride,
        lat_stride=lat_stride,
        lon_stride=lon_stride,
        time_start=time_start,
        max_time=max_time,
    )

    time = time_all[::time_stride]
    lat = lat_all[::lat_stride]
    lon = lon_all[::lon_stride]
    if time_start > 0:
        time = time[time_start:]
    if max_time is not None:
        time = time[:max_time]

    u = _center_crop(u, crop_ny, crop_nx)
    v = _center_crop(v, crop_ny, crop_nx)
    lat = _center_crop_coords(lat, crop_ny)
    lon = _center_crop_coords(lon, crop_nx)

    nt, ny, nx = u.shape
    if v.shape != (nt, ny, nx):
        raise ValueError(f"Shape mismatch u={u.shape}, v={v.shape}")
    if len(time) != nt or len(lat) != ny or len(lon) != nx:
        raise ValueError("Coordinate lengths do not match field shapes after slicing/cropping.")

    ds.close()
    return (
        np.asarray(time),
        np.asarray(lat),
        np.asarray(lon),
        np.asarray(u, dtype=float),
        np.asarray(v, dtype=float),
        str(name_u),
        str(name_v),
    )


def _build_coefficients(
    *,
    mode_fields: dict[str, np.ndarray],
    selected: list[dict[str, list[tuple[int, int]]]],
) -> list[np.ndarray]:
    roles = list(mode_fields.keys())
    nt = next(iter(mode_fields.values())).shape[0]
    n_bands = len(selected)
    sizes = [sum(len(selected[b][r]) for r in roles) for b in range(n_bands)]
    coeff = [np.zeros((nt, m), dtype=np.complex128) for m in sizes]

    for t in range(nt):
        fft_cache: dict[str, np.ndarray] = {}
        for r in roles:
            field = np.nan_to_num(mode_fields[r][t], nan=0.0, posinf=0.0, neginf=0.0)
            fft_cache[r] = np.fft.rfft2(field - np.mean(field))

        for b in range(n_bands):
            vals: list[complex] = []
            for r in roles:
                fh = fft_cache[r]
                vals.extend(fh[iy, ix] for iy, ix in selected[b][r])
            if len(vals) > 0:
                coeff[b][t, :] = np.asarray(vals, dtype=np.complex128)
    return coeff


def _compute_pi_proxy(
    *,
    u: np.ndarray,
    v: np.ndarray,
    masks: list[np.ndarray],
    ky: np.ndarray,
    kx: np.ndarray,
) -> np.ndarray:
    nt = u.shape[0]
    n_bands = len(masks)
    k_mag = np.sqrt(ky[:, None] * ky[:, None] + kx[None, :] * kx[None, :])
    pi = np.zeros((nt, n_bands), dtype=float)
    for t in range(nt):
        uh = np.fft.rfft2(u[t] - np.mean(u[t]))
        vh = np.fft.rfft2(v[t] - np.mean(v[t]))
        e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
        for b, m in enumerate(masks):
            if int(m.sum()) == 0:
                continue
            pi[t, b] = float(np.mean(k_mag[m] * e[m]))
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
        if arr.shape[1] == 0:
            coeff_scaled.append(arr)
            continue
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
        if m == 0:
            rho_cache[b] = [np.eye(1, dtype=np.complex128) for _ in range(nt)]
            continue
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
        if m1 == 0 or m2 == 0:
            continue
        eye1 = np.eye(m1, dtype=np.complex128)
        eye2 = np.eye(m2, dtype=np.complex128)
        for t in range(nt):
            i0 = max(0, t - window + 1)
            a = coeff_scaled[b][i0 : t + 1]
            bnext = coeff_scaled[b + 1][i0 : t + 1]
            g1 = a.conj().T @ a + ridge * eye1
            g2 = bnext.conj().T @ bnext + ridge * eye2
            rhs_f = a.conj().T @ bnext
            rhs_r = bnext.conj().T @ a
            try:
                m_fwd = np.linalg.solve(g1, rhs_f)
            except np.linalg.LinAlgError:
                m_fwd = np.linalg.pinv(g1, rcond=1e-10) @ rhs_f
            try:
                m_rev = np.linalg.solve(g2, rhs_r)
            except np.linalg.LinAlgError:
                m_rev = np.linalg.pinv(g2, rcond=1e-10) @ rhs_r

            g_fwd = m_fwd @ m_fwd.conj().T
            g_back = m_rev.conj().T @ m_rev
            comm = g_fwd @ g_back - g_back @ g_fwd
            f_phys = 0.5 * (1j * comm + (1j * comm).conj().T)
            rho = rho_cache[b][t]
            lambda_sum[t, b] += float(np.real(np.trace(f_phys @ rho)))
            lambda_count[t, b] += 1.0

            g_fwd_r = m_fwd.conj().T @ m_fwd
            g_back_r = m_rev @ m_rev.conj().T
            comm_r = g_fwd_r @ g_back_r - g_back_r @ g_fwd_r
            f_phys_r = 0.5 * (1j * comm_r + (1j * comm_r).conj().T)
            rho_r = rho_cache[b + 1][t]
            lambda_sum[t, b + 1] += float(np.real(np.trace(f_phys_r @ rho_r)))
            lambda_count[t, b + 1] += 1.0

    nz = lambda_count > 0
    lambda_mu[nz] = lambda_sum[nz] / lambda_count[nz]
    return lambda_mu, trace_err, min_eig


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(d, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def _binned_means(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
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
        if int(np.sum(m)) < 3:
            continue
        xb.append(float(np.mean(x[m])))
        yb.append(float(np.mean(y[m])))
    return np.asarray(xb, dtype=float), np.asarray(yb, dtype=float)


def _slope_permutation_pvalue(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    if len(x) < 3:
        return float("nan")
    slope_obs, _, _ = _linear_fit(x, y)
    if not np.isfinite(slope_obs) or slope_obs <= 0.0:
        return 1.0
    c = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        slope_perm, _, _ = _linear_fit(x, yp)
        if slope_perm >= slope_obs:
            c += 1
    return float((c + 1) / (n_perm + 1))


def _regime_from_center(
    center_km: float,
    *,
    forcing_threshold_km: float,
    inertial_low_km: float,
) -> str:
    if center_km > forcing_threshold_km:
        return "forcing"
    if center_km < inertial_low_km:
        return "dissipation"
    return "inertial"


def _build_report(
    *,
    outdir: Path,
    args: argparse.Namespace,
    metadata: dict[str, float | int | str],
    band_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    trace_err: np.ndarray,
    min_eig: np.ndarray,
    pass_all: bool,
) -> None:
    inert = regime_df.loc[regime_df["regime"] == "inertial"]
    if len(inert) > 0:
        inert_r2 = float(inert["r2_binned"].iloc[0])
        inert_slope = float(inert["slope_binned"].iloc[0])
        inert_p = float(inert["slope_perm_p_binned"].iloc[0])
    else:
        inert_r2 = float("nan")
        inert_slope = float("nan")
        inert_p = float("nan")

    lines = [
        "# Einstein in the Atmospheric Column (ERA5 Box)",
        "",
        "## Goal",
        "- Test local Einstein-type scale-space closure on real atmosphere data:",
        "  `E[Lambda | Pi]` in forcing/inertial/dissipation ranges.",
        "",
        "## Input",
        f"- input: `{args.input}`",
        f"- field_set: `{args.field_set}`",
        f"- vector vars: `{metadata['var_u']}`, `{metadata['var_v']}`",
        f"- shape after slicing: `time={metadata['nt']}, lat={metadata['ny']}, lon={metadata['nx']}`",
        f"- median dx,dy (km): `{metadata['dx_km']:.2f}`, `{metadata['dy_km']:.2f}`",
        f"- time range: `{metadata['time_start']}` -> `{metadata['time_end']}`",
        "",
        "## Scale and model setup",
        f"- scale_edges_km: `{args.scale_edges_km}`",
        f"- regimes: dissipation `< {args.inertial_low_km}` km, inertial `["
        f"{args.inertial_low_km}, {args.forcing_threshold_km}]` km, forcing `> {args.forcing_threshold_km}` km",
        f"- n_modes_per_var requested/effective(min across bands): "
        f"`{args.n_modes_per_var}/{metadata['n_modes_eff']}`",
        f"- rolling window W: `{args.window}`",
        f"- ridge: `{args.ridge:.2e}`, cov_shrinkage: `{args.cov_shrinkage:.3f}`",
        f"- lambda sign requested/effective: `{args.lambda_sign}/{metadata['lambda_sign_eff']}`",
        "",
        "## Density-matrix checks",
        f"- max `|Tr(rho)-1|`: `{float(np.nanmax(trace_err)):.3e}`",
        f"- min eigenvalue(rho): `{float(np.nanmin(min_eig)):.3e}`",
        "",
        "## Bandwise results",
    ]
    for _, r in band_df.iterrows():
        lines.append(
            "- band {bid} ({reg}), center={c:.1f} km: R2_binned={r2:.3f}, "
            "slope_binned={s:.3e}, p_slope={p:.4f}".format(
                bid=int(r["band_id"]),
                reg=str(r["regime"]),
                c=float(r["center_scale_km"]),
                r2=float(r["r2_binned"]),
                s=float(r["slope_binned"]),
                p=float(r["slope_perm_p_binned"]),
            )
        )

    lines.extend(["", "## Regime aggregates"])
    for _, r in regime_df.iterrows():
        lines.append(
            "- {reg}: n={n}, R2_binned={r2:.3f}, slope_binned={s:.3e}, p_slope={p:.4f}".format(
                reg=str(r["regime"]),
                n=int(r["n_samples"]),
                r2=float(r["r2_binned"]),
                s=float(r["slope_binned"]),
                p=float(r["slope_perm_p_binned"]),
            )
        )

    lines.extend(
        [
            "",
            "## Pass criteria (ERA)",
            f"- inertial R2_binned >= {args.success_r2:.2f}: `{inert_r2:.3f}`",
            f"- inertial slope_binned > 0: `{inert_slope:.3e}`",
            f"- inertial slope significance p <= {args.success_p:.2f}: `{inert_p:.4f}`",
            "- forcing/dissipation breakdown: checked via lower R2 vs inertial or slope sign inversion.",
            f"- PASS_ALL: `{bool(pass_all)}`",
            "",
            "## Note",
            "- Atmospheric noise is expected to reduce raw fit quality.",
            "- Binned inertial relation is the primary criterion for this box-level closure test.",
        ]
    )

    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    time, lat, lon, u, v, var_u_name, var_v_name = _load_vector_fields(
        input_path=Path(args.input),
        field_set=args.field_set,
        u_var=args.u_var,
        v_var=args.v_var,
        time_stride=args.time_stride,
        lat_stride=args.lat_stride,
        lon_stride=args.lon_stride,
        time_start=args.time_start,
        max_time=args.max_time,
        crop_ny=args.crop_ny,
        crop_nx=args.crop_nx,
    )
    x_m, y_m = _xy_coordinates_m(lat, lon)
    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    dx_km = float(np.median(np.abs(np.diff(x_km))))
    dy_km = float(np.median(np.abs(np.diff(y_km))))

    scale_edges = [float(s) for s in args.scale_edges_km.split(",")]
    masks, ky, kx, centers, mus, wavelength_km = _build_band_masks(
        ny=u.shape[1],
        nx=u.shape[2],
        dy_km=dy_km,
        dx_km=dx_km,
        scale_edges_km=scale_edges,
    )

    mode_fields = {"u": u, "v": v}
    selected, mode_meta = _select_mode_indices(
        mode_fields=mode_fields,
        masks=masks,
        n_modes_per_var=args.n_modes_per_var,
        wavelength_km=wavelength_km,
    )
    coeff_by_band = _build_coefficients(mode_fields=mode_fields, selected=selected)
    lambda_mu, trace_err, min_eig = _compute_rho_and_lambda(
        coeff_by_band=coeff_by_band,
        window=args.window,
        ridge=args.ridge,
        cov_shrinkage=args.cov_shrinkage,
    )
    pi_mu = _compute_pi_proxy(u=u, v=v, masks=masks, ky=ky, kx=kx)

    # Optional sign convention alignment in inertial range.
    sign_eff = int(args.lambda_sign)
    if sign_eff not in (-1, 0, 1):
        raise ValueError("lambda_sign must be one of {-1, 0, +1}.")
    if sign_eff == 0:
        t0 = max(args.window - 1, 1)
        xs = []
        ys = []
        for b, c in enumerate(centers):
            reg = _regime_from_center(
                c,
                forcing_threshold_km=args.forcing_threshold_km,
                inertial_low_km=args.inertial_low_km,
            )
            if reg != "inertial":
                continue
            x = pi_mu[t0:, b]
            y = lambda_mu[t0:, b]
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) > 5:
                xs.append(x[m])
                ys.append(y[m])
        if len(xs) > 0:
            slope, _, _ = _linear_fit(np.concatenate(xs), np.concatenate(ys))
            sign_eff = 1 if slope >= 0.0 else -1
        else:
            sign_eff = 1
    if sign_eff == 0:
        sign_eff = 1
    lambda_mu = float(sign_eff) * lambda_mu

    rows = []
    for b in range(len(masks)):
        reg = _regime_from_center(
            centers[b],
            forcing_threshold_km=args.forcing_threshold_km,
            inertial_low_km=args.inertial_low_km,
        )
        for t in range(len(time)):
            rows.append(
                {
                    "t": int(t),
                    "time": str(pd.to_datetime(time[t])),
                    "band_id": int(b),
                    "mu": float(mus[b]),
                    "center_scale_km": float(centers[b]),
                    "regime": reg,
                    "pi": float(pi_mu[t, b]),
                    "lambda": float(lambda_mu[t, b]) if np.isfinite(lambda_mu[t, b]) else np.nan,
                    "trace_err": float(trace_err[t, b]),
                    "min_eig": float(min_eig[t, b]),
                }
            )
    ts_df = pd.DataFrame(rows)
    ts_df.to_csv(outdir / "timeseries_long.csv", index=False)
    mode_meta.to_csv(outdir / "mode_selection.csv", index=False)

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
            p_bin = _slope_permutation_pvalue(x=xb, y=yb, n_perm=args.n_perm_slope, rng=rng)
        else:
            slope_bin, intercept_bin, r2_bin, p_bin = float("nan"), float("nan"), float("nan"), float("nan")
        band_rows.append(
            {
                "band_id": int(b),
                "regime": str(g["regime"].iloc[0]),
                "mu": float(mus[b]),
                "center_scale_km": float(centers[b]),
                "n_samples": int(len(g)),
                "slope_raw": float(slope_raw),
                "intercept_raw": float(intercept_raw),
                "r2_raw": float(r2_raw),
                "slope_binned": float(slope_bin),
                "intercept_binned": float(intercept_bin),
                "r2_binned": float(r2_bin),
                "n_bins_eff": int(len(xb)),
                "slope_perm_p_binned": float(p_bin),
            }
        )
    band_df = pd.DataFrame(band_rows).sort_values("band_id").reset_index(drop=True)
    band_df.to_csv(outdir / "band_regression.csv", index=False)

    regime_rows = []
    for reg in ("forcing", "inertial", "dissipation"):
        g = ts_eval.loc[(ts_eval["regime"] == reg) & np.isfinite(ts_eval["lambda"]) & np.isfinite(ts_eval["pi"])]
        if len(g) < 20:
            continue
        x = g["pi"].to_numpy(dtype=float)
        y = g["lambda"].to_numpy(dtype=float)
        slope_raw, intercept_raw, r2_raw = _linear_fit(x, y)
        xb, yb = _binned_means(x, y, n_bins=args.bins)
        if len(xb) >= 3:
            slope_bin, intercept_bin, r2_bin = _linear_fit(xb, yb)
            p_bin = _slope_permutation_pvalue(x=xb, y=yb, n_perm=args.n_perm_slope, rng=rng)
        else:
            slope_bin, intercept_bin, r2_bin, p_bin = float("nan"), float("nan"), float("nan"), float("nan")
        regime_rows.append(
            {
                "regime": reg,
                "n_samples": int(len(g)),
                "slope_raw": float(slope_raw),
                "intercept_raw": float(intercept_raw),
                "r2_raw": float(r2_raw),
                "slope_binned": float(slope_bin),
                "intercept_binned": float(intercept_bin),
                "r2_binned": float(r2_bin),
                "n_bins_eff": int(len(xb)),
                "slope_perm_p_binned": float(p_bin),
            }
        )
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(outdir / "aggregate_regression.csv", index=False)

    inert = regime_df.loc[regime_df["regime"] == "inertial"]
    force = regime_df.loc[regime_df["regime"] == "forcing"]
    diss = regime_df.loc[regime_df["regime"] == "dissipation"]
    inert_ok = False
    forcing_break = False
    diss_break = False
    if len(inert) > 0:
        r2_i = float(inert["r2_binned"].iloc[0])
        s_i = float(inert["slope_binned"].iloc[0])
        p_i = float(inert["slope_perm_p_binned"].iloc[0])
        inert_ok = bool(r2_i >= args.success_r2 and s_i > 0.0 and p_i <= args.success_p)
        if len(force) > 0:
            r2_f = float(force["r2_binned"].iloc[0])
            s_f = float(force["slope_binned"].iloc[0])
            forcing_break = bool((r2_f <= r2_i - args.break_margin_r2) or (np.sign(s_f) != np.sign(s_i)))
        if len(diss) > 0:
            r2_d = float(diss["r2_binned"].iloc[0])
            s_d = float(diss["slope_binned"].iloc[0])
            diss_break = bool((r2_d <= r2_i - args.break_margin_r2) or (np.sign(s_d) != np.sign(s_i)))

    pass_all = bool(inert_ok and forcing_break and diss_break)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        colors = {"forcing": "#d04a1f", "inertial": "#1f77b4", "dissipation": "#2ca02c"}
        for reg, g in ts_eval.groupby("regime"):
            ax.scatter(g["pi"], g["lambda"], s=8, alpha=0.18, c=colors.get(reg, "gray"), label=reg)
            xb, yb = _binned_means(g["pi"].to_numpy(dtype=float), g["lambda"].to_numpy(dtype=float), n_bins=args.bins)
            if len(xb) >= 3:
                slope, intercept, r2 = _linear_fit(xb, yb)
                xx = np.linspace(float(np.min(xb)), float(np.max(xb)), 200)
                yy = slope * xx + intercept
                ax.plot(xx, yy, c=colors.get(reg, "gray"), lw=2.2, alpha=0.9, label=f"{reg} fit R2={r2:.2f}")
                ax.scatter(xb, yb, c=colors.get(reg, "gray"), s=30, zorder=3)
        ax.set_xlabel("Pi (spectral energy-flow proxy)")
        ax.set_ylabel("Lambda (curvature projection)")
        ax.set_title("ERA5 box: binned Lambda-Pi relation by scale regime")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "lambda_vs_pi_regimes.png", dpi=180)
        plt.close(fig)

    n_modes_eff = int(mode_meta.groupby("band_id")["mode_rank"].max().min() + 1) if len(mode_meta) > 0 else 0
    metadata = {
        "var_u": str(var_u_name),
        "var_v": str(var_v_name),
        "nt": int(u.shape[0]),
        "ny": int(u.shape[1]),
        "nx": int(u.shape[2]),
        "dx_km": float(dx_km),
        "dy_km": float(dy_km),
        "time_start": str(pd.to_datetime(time[0])),
        "time_end": str(pd.to_datetime(time[-1])),
        "n_modes_eff": int(n_modes_eff),
        "lambda_sign_eff": int(sign_eff),
    }
    _build_report(
        outdir=outdir,
        args=args,
        metadata=metadata,
        band_df=band_df,
        regime_df=regime_df,
        trace_err=trace_err,
        min_eig=min_eig,
        pass_all=pass_all,
    )

    print("Saved:", outdir.resolve())
    if len(regime_df) > 0:
        print("\nAggregate regime regression:")
        print(regime_df.to_string(index=False))
    print(
        "\nPASS components:",
        {
            "inertial_ok": inert_ok,
            "forcing_break": forcing_break,
            "diss_break": diss_break,
            "PASS_ALL": pass_all,
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/processed/wpwp_era5_2017_2019_experiment_M_input.nc",
        help="ERA5 NetCDF input.",
    )
    p.add_argument(
        "--outdir",
        default="clean_experiments/results/experiment_scale_gravity_einstein_box_era",
    )
    p.add_argument("--field-set", choices=("wind", "ivt"), default="wind")
    p.add_argument("--u-var", default=None, help="Override variable name for u/ivt_u channel.")
    p.add_argument("--v-var", default=None, help="Override variable name for v/ivt_v channel.")
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--lat-stride", type=int, default=1)
    p.add_argument("--lon-stride", type=int, default=1)
    p.add_argument("--time-start", type=int, default=0)
    p.add_argument("--max-time", type=int, default=360, help="Use ~3 months by default (6-hourly data).")
    p.add_argument("--crop-ny", type=int, default=0, help="Center crop latitude count; 0 keeps full.")
    p.add_argument("--crop-nx", type=int, default=0, help="Center crop longitude count; 0 keeps full.")
    p.add_argument("--scale-edges-km", default="50,100,200,400,800,1600,3200")
    p.add_argument("--n-modes-per-var", type=int, default=6)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--cov-shrinkage", type=float, default=0.05)
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--forcing-threshold-km", type=float, default=1000.0)
    p.add_argument("--inertial-low-km", type=float, default=100.0)
    p.add_argument("--success-r2", type=float, default=0.40)
    p.add_argument("--success-p", type=float, default=0.05)
    p.add_argument("--break-margin-r2", type=float, default=0.10)
    p.add_argument("--n-perm-slope", type=int, default=999)
    p.add_argument(
        "--lambda-sign",
        type=int,
        default=0,
        help="Sign convention for Lambda: -1 or +1, or 0 to auto-align inertial slope.",
    )
    p.add_argument("--seed", type=int, default=20260309)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.max_time = min(args.max_time, 360)
        args.n_perm_slope = min(args.n_perm_slope, 199)
    return args


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
