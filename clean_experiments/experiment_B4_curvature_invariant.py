#!/usr/bin/env python3
"""Phase-4: curvature strength as a physical regional invariant.

Preregistered protocol: docs/PROTOCOL_PHASE4_CURVATURE_INVARIANT.md
(frozen 2026-08-12, pre-computation revision of the confirmation set logged).

Confirmation data: curvature-naive on-disk region-windows
(data/b1: R1-R4 x 2017-2019; data/b2b: R5-R12 x 2021-2022).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import clean_experiments.download_b1_era5_wind as dl_b1
    from clean_experiments.download_b2_era5_wind import REGIONS_B2
    from clean_experiments.download_b2b_era5_wind import REGIONS_B2B
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        SCALE_EDGES_KM,
        compute_vorticity,
        envelope_rho_profile,
        spectral_features,
    )
    from clean_experiments.experiment_B2b_heldout_invariants import (
        _perm_test,
        _zscore_cols,
    )
    from clean_experiments.experiment_B3_scattering_benchmark import (
        loo_pca_residuals,
    )
    from clean_experiments.experiment_M_cosmo_flow import (
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _build_coefficients,
        _load_vector_fields,
    )
except ImportError:
    import download_b1_era5_wind as dl_b1  # type: ignore
    from download_b2_era5_wind import REGIONS_B2  # type: ignore
    from download_b2b_era5_wind import REGIONS_B2B  # type: ignore
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        SCALE_EDGES_KM,
        compute_vorticity,
        envelope_rho_profile,
        spectral_features,
    )
    from experiment_B2b_heldout_invariants import (  # type: ignore
        _perm_test,
        _zscore_cols,
    )
    from experiment_B3_scattering_benchmark import (  # type: ignore
        loo_pca_residuals,
    )
    from experiment_M_cosmo_flow import (  # type: ignore
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _build_coefficients,
        _load_vector_fields,
    )

SEED = 20260811
WINDOW, RIDGE, SHRINK, NMODES, WARMUP = 20, 1e-6, 0.05, 6, 19
REGIONS_ALL = {**dl_b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}

B1_WINDOWS = ["W1_2017JFM", "W2_2017JAS", "W3_2018JFM", "W4_2019JAS"]
B2B_WINDOWS = ["W5_2021JFM", "W6_2021JAS", "W7_2022JFM", "W8_2022JAS"]


def confirmation_files(root: Path) -> list[Path]:
    files = []
    for r in ("R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA"):
        for w in B1_WINDOWS:
            files.append(root / "data/b1" / f"era5_wind850_{r}__{w}.nc")
    for r in ("R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS",
              "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"):
        for w in B2B_WINDOWS:
            files.append(root / "data/b2b" / f"era5_wind850_{r}__{w}.nc")
    return [f for f in files if f.exists()]


def curvature_profiles(u, v, lat, lon) -> dict:
    x_m, y_m = _xy_coordinates_m(lat, lon)
    dx = float(np.median(np.abs(np.diff(x_m / 1000.0))))
    dy = float(np.median(np.abs(np.diff(y_m / 1000.0))))
    masks, _, _, _, _, wl = _build_band_masks(
        ny=u.shape[1], nx=u.shape[2], dy_km=dy, dx_km=dx, scale_edges_km=SCALE_EDGES_KM
    )
    fields = {"u": u, "v": v}
    selected, _ = _select_mode_indices(
        mode_fields=fields, masks=masks, n_modes_per_var=NMODES, wavelength_km=wl
    )
    coeff = _build_coefficients(mode_fields=fields, selected=selected)

    nb = len(coeff)
    nt = coeff[0].shape[0]
    scaled = []
    for b in range(nb):
        a = np.asarray(coeff[b], dtype=np.complex128)
        rms = np.sqrt(np.mean(np.abs(a) ** 2, axis=0))
        scaled.append(a / np.where(rms < 1e-12, 1.0, rms)[None, :])

    rho = [[None] * nt for _ in range(nb)]
    for b in range(nb):
        m = scaled[b].shape[1]
        eye = np.eye(m, dtype=np.complex128)
        for t in range(nt):
            a = scaled[b][max(0, t - WINDOW + 1): t + 1]
            c = (a.conj().T @ a) / len(a)
            c = 0.5 * (c + c.conj().T)
            c = (1 - SHRINK) * c + SHRINK * np.diag(np.diag(c)) + RIDGE * eye
            tr = float(np.real(np.trace(c)))
            rho[b][t] = eye / m if tr < 1e-14 else c / tr

    fnorm = np.zeros((nt, nb))
    lam = np.zeros((nt, nb))
    cnt = np.zeros((nt, nb))
    for b in range(nb - 1):
        m1, m2 = scaled[b].shape[1], scaled[b + 1].shape[1]
        e1 = np.eye(m1, dtype=np.complex128)
        e2 = np.eye(m2, dtype=np.complex128)
        for t in range(nt):
            a = scaled[b][max(0, t - WINDOW + 1): t + 1]
            bb = scaled[b + 1][max(0, t - WINDOW + 1): t + 1]
            g1 = a.conj().T @ a + RIDGE * e1
            g2 = bb.conj().T @ bb + RIDGE * e2
            mf = np.linalg.solve(g1, a.conj().T @ bb)
            mr = np.linalg.solve(g2, bb.conj().T @ a)
            for (gf, gb, r, bi) in (
                (mf @ mf.conj().T, mr.conj().T @ mr, rho[b][t], b),
                (mf.conj().T @ mf, mr @ mr.conj().T, rho[b + 1][t], b + 1),
            ):
                comm = gf @ gb - gb @ gf
                f = 0.5 * (1j * comm + (1j * comm).conj().T)
                fnorm[t, bi] += float(np.linalg.norm(f))
                lam[t, bi] += float(np.real(np.trace(f @ r)))
                cnt[t, bi] += 1
    nz = cnt > 0
    fnorm[nz] /= cnt[nz]
    lam[nz] /= cnt[nz]
    return {
        "Fnorm_profile": [float(x) for x in np.median(fnorm[WARMUP:], axis=0)],
        "Lam_profile": [float(x) for x in np.median(lam[WARMUP:], axis=0)],
        "LamAbs_profile": [float(x) for x in np.median(np.abs(lam[WARMUP:]), axis=0)],
    }


def run_region_window(nc_path: Path, out_dir: Path) -> dict:
    tag = nc_path.stem.replace("era5_wind850_", "")
    cached = out_dir / f"{tag}.json"
    if cached.exists():
        print(f"[{tag}] cached", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))
    print(f"[{tag}] computing", flush=True)
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], 1600.0, dx_km, dy_km)
    omega = compute_vorticity(u, v, lat, lon)
    rho = envelope_rho_profile(omega, dx_km, dy_km, mask)
    result = {
        "tag": tag,
        "region": tag.split("__")[0],
        "window": tag.split("__")[1],
        **curvature_profiles(u, v, lat, lon),
        "P_real": [float(x) for x in np.median(rho, axis=0)],
        "F_spec": spectral_features(omega, dx_km, dy_km, mask),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def region_covariates(cov_dir: Path) -> dict[str, dict[str, float]]:
    import xarray as xr
    inv = xr.open_dataset(cov_dir / "era5_invariants_global.nc")
    z_name = "z" if "z" in inv else list(inv.data_vars)[0]
    lsm_name = "lsm" if "lsm" in inv else list(inv.data_vars)[-1]
    z = inv[z_name].squeeze()
    lsm = inv[lsm_name].squeeze()
    lat_name = "latitude" if "latitude" in z.dims else "lat"
    lon_name = "longitude" if "longitude" in z.dims else "lon"

    out = {}
    for region, (n, w, s, e) in REGIONS_ALL.items():
        lon_sel = ((w + 360) % 360, (e + 360) % 360)
        zz = z.sel({lat_name: slice(n, s)})
        ll = lsm.sel({lat_name: slice(n, s)})
        lon_vals = zz[lon_name].values
        lon_mod = (lon_vals + 360) % 360
        if lon_sel[0] <= lon_sel[1]:
            m = (lon_mod >= lon_sel[0]) & (lon_mod <= lon_sel[1])
        else:
            m = (lon_mod >= lon_sel[0]) | (lon_mod <= lon_sel[1])
        zz = zz.isel({lon_name: np.where(m)[0]})
        ll = ll.isel({lon_name: np.where(m)[0]})
        cape_ds = xr.open_dataset(cov_dir / f"era5_cape_monthly_{region}.nc")
        cape_name = next(v for v in cape_ds.data_vars)
        out[region] = {
            "orog_std": float((zz / 9.80665).std()),
            "land_frac": float(ll.mean()),
            "cape_mean": float(cape_ds[cape_name].mean()),
        }
        cape_ds.close()
    inv.close()
    return out


def consolidate(results: list[dict], out_dir: Path, cov_dir: Path, b3_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    labels = np.asarray([r["region"] for r in results])
    fn = np.asarray([r["Fnorm_profile"] for r in results])
    la = np.asarray([r["LamAbs_profile"] for r in results])
    p_mat = np.asarray([r["P_real"] for r in results])
    spec_keys = sorted(results[0]["F_spec"].keys())
    spec = np.asarray([[r["F_spec"][k] for k in spec_keys] for r in results])

    h4a = _perm_test(_zscore_cols(fn), labels, rng)
    res_spec = loo_pca_residuals(spec, fn, 6)
    h4b_spec = _perm_test(_zscore_cols(res_spec), labels, rng)
    res_p = loo_pca_residuals(p_mat, fn, 5)
    h4b_p = _perm_test(_zscore_cols(res_p), labels, rng)
    h4b_pass = bool(h4b_spec["pass"] and h4b_p["pass"])

    # H4c: pooled region medians (Phase-3 + Phase-4) of fine-band Fnorm
    pooled: dict[str, list[float]] = {}
    for r in results:
        pooled.setdefault(r["region"], []).append(float(np.mean(r["Fnorm_profile"][:2])))
    for f in sorted(b3_dir.glob("*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "E2_curvature" in r:
            pooled.setdefault(r["region"], []).append(
                float(np.mean(r["E2_curvature"]["Fnorm_profile"][:2]))
            )
    regions = sorted(pooled.keys())
    y = np.log(np.asarray([np.median(pooled[r]) for r in regions]))
    cov = region_covariates(cov_dir)
    x = np.asarray([[cov[r]["orog_std"], cov[r]["land_frac"], cov[r]["cape_mean"]]
                    for r in regions])
    xz = _zscore_cols(x)

    def loo_r2(xm: np.ndarray, yv: np.ndarray) -> float:
        n = len(yv)
        pred = np.zeros(n)
        for k in range(n):
            tr = np.arange(n) != k
            a = np.hstack([np.ones((tr.sum(), 1)), xm[tr]])
            coef, *_ = np.linalg.lstsq(a, yv[tr], rcond=None)
            pred[k] = np.hstack([[1.0], xm[k]]) @ coef
        ss_res = float(np.sum((yv - pred) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    r2_obs = loo_r2(xz, y)
    perm = [loo_r2(xz, y[rng.permutation(len(y))]) for _ in range(999)]
    p_c = float((1 + np.sum(np.asarray(perm) >= r2_obs)) / 1000)
    h4c = {"loo_r2": r2_obs, "p": p_c, "pass": bool(r2_obs > 0 and p_c < 0.05),
           "covariates": cov, "region_median_log_fnorm_fine": dict(zip(regions, y.tolist()))}

    if h4a["pass"] and h4b_pass:
        verdict = "CONFIRMED_PHYSICAL" if h4c["pass"] else "CONFIRMED_DESCRIPTIVE"
    else:
        verdict = "NEGATIVE"

    la_sig = _perm_test(_zscore_cols(la), labels, rng)
    summary = {
        "n_region_windows": len(results),
        "H4a_signature": h4a,
        "H4b_beyond_spectrum": h4b_spec,
        "H4b_beyond_P": h4b_p,
        "H4c_physical": h4c,
        "PHASE4_VERDICT": verdict,
        "descriptive": {
            "LamAbs_signature": la_sig,
            "mean_Fnorm": {
                reg: [float(v) for v in fn[labels == reg].mean(0)]
                for reg in sorted(set(labels))
            },
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Experiment B4: curvature strength as regional invariant", ""]
    lines.append(f"Region-windows: {len(results)} (curvature-naive years).")
    lines.append(f"H4a signature: diff={h4a['diff']:+.3f} p={h4a['p']:.3f} pass={h4a['pass']}")
    lines.append(f"H4b beyond spectrum: diff={h4b_spec['diff']:+.3f} p={h4b_spec['p']:.3f}; "
                 f"beyond P: diff={h4b_p['diff']:+.3f} p={h4b_p['p']:.3f}; pass={h4b_pass}")
    lines.append(f"H4c physical (orog_std, land_frac, cape_mean -> log fine-band Fnorm): "
                 f"LOO R^2={r2_obs:.3f} p={p_c:.3f} pass={h4c['pass']}")
    lines.append("")
    lines.append(f"## PHASE4_VERDICT: {verdict}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("H4a_signature", "H4b_beyond_spectrum", "H4b_beyond_P",
                       "H4c_physical", "PHASE4_VERDICT")}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_HERE.parent)
    parser.add_argument("--out-dir", type=Path,
                        default=_HERE / "results" / "experiment_B4_curvature_invariant")
    parser.add_argument("--cov-dir", type=Path, default=None)
    parser.add_argument("--skip-consolidate", action="store_true")
    args = parser.parse_args()

    root = args.repo_root
    cov_dir = args.cov_dir or (root / "data/b4cov")
    b3_dir = _HERE / "results" / "experiment_B3_scattering_benchmark"

    files = confirmation_files(root)
    print(f"{len(files)} confirmation region-window files found", flush=True)
    results = [run_region_window(f, args.out_dir) for f in files]

    if not args.skip_consolidate:
        consolidate(results, args.out_dir, cov_dir, b3_dir)


if __name__ == "__main__":
    main()
