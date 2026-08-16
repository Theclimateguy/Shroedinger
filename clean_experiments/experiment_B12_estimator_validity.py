#!/usr/bin/env python3
"""Phase-12: is A a valid estimator of mesoscale irrecoverability?

Preregistered protocol: docs/PROTOCOL_PHASE12_ESTIMATOR_VALIDITY.md
(frozen 2026-08-16).

  --stage synthetic : H12a, the controlled positive test
  --stage real      : u(irrecoverability) on the 39 programme domains
  --stage tests     : H12b-H12d and the verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM, N_STEPS, gaussian_bar_grouped,
    )
    from clean_experiments.experiment_B3_scattering_benchmark import loo_pca_residuals
    from clean_experiments.experiment_B4_curvature_invariant import curvature_profiles
    from clean_experiments.experiment_B9_predictability import (
        BANDS_RESOLVED, perm_spearman, vort,
    )
    from clean_experiments.download_b11_wb2 import DOMAINS
except ImportError:  # pragma: no cover
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM, N_STEPS, gaussian_bar_grouped,
    )
    from experiment_B3_scattering_benchmark import loo_pca_residuals  # type: ignore
    from experiment_B4_curvature_invariant import curvature_profiles  # type: ignore
    from experiment_B9_predictability import BANDS_RESOLVED, perm_spearman, vort  # type: ignore
    from download_b11_wb2 import DOMAINS  # type: ignore

SEED = 20260816
NBINS = 16
OUT = Path("clean_experiments/results/experiment_B12_estimator_validity")
ERA5_DIRS = {"prog": Path("data/b3"), "lattice": Path("data/b10era5")}
EPS = 1e-12

# Envelope pairs whose finer member spans 200-800 km, matching BANDS_RESOLVED
PAIRS_RESOLVED = (3, 4)


# --------------------------------------------------------------------------
# information-theoretic irrecoverability
# --------------------------------------------------------------------------

def envelopes(omega: np.ndarray, dx_km: np.ndarray, dy_km: float) -> list[np.ndarray]:
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ELLS_KM]
    envs = [np.log(gaussian_bar_grouped(np.abs(w - bars[0]),
                                        ELLS_KM[0] / np.sqrt(12.0), dx_km, dy_km) + EPS)]
    for i in range(N_STEPS):
        d = bars[i] - bars[i + 1]
        envs.append(np.log(gaussian_bar_grouped(np.abs(d), ELLS_KM[i + 1] / np.sqrt(12.0),
                                                dx_km, dy_km) + EPS))
    return envs


def _rank_bins(x: np.ndarray, nbins: int) -> np.ndarray:
    """Equal-frequency binning of ranks: invariant to any monotone transform."""
    order = np.argsort(np.argsort(x))
    return np.minimum((order * nbins) // len(x), nbins - 1)


def norm_mi(x: np.ndarray, y: np.ndarray, nbins: int = NBINS) -> float:
    bx, by = _rank_bins(x, nbins), _rank_bins(y, nbins)
    joint = np.zeros((nbins, nbins), dtype=float)
    np.add.at(joint, (bx, by), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    return mi / hx if hx > 0 else 0.0


def irrecoverability(omega: np.ndarray, dx_km: np.ndarray, dy_km: float,
                     mask: np.ndarray) -> np.ndarray:
    """u_i(t) for the five adjacent-envelope pairs: 1 - normalised MI."""
    envs = envelopes(omega, dx_km, dy_km)
    nt = omega.shape[0]
    u = np.zeros((nt, N_STEPS))
    for i in range(N_STEPS):
        a, b = envs[i][:, mask], envs[i + 1][:, mask]
        for t in range(nt):
            u[t, i] = 1.0 - norm_mi(a[t], b[t])
    return u


# --------------------------------------------------------------------------
# H12a: synthetic ladder with known irrecoverability
# --------------------------------------------------------------------------

def _uv_from_vorticity(omega: np.ndarray, dx_km: float, dy_km: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Invert vorticity to a non-divergent wind, so that curl(u, v) == omega."""
    nt, ny, nx = omega.shape
    ky = np.fft.fftfreq(ny, d=dy_km)[:, None] * 2 * np.pi
    kx = np.fft.rfftfreq(nx, d=dx_km)[None, :] * 2 * np.pi
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    u = np.empty_like(omega)
    v = np.empty_like(omega)
    for t in range(nt):
        w = np.fft.rfft2(omega[t])
        psi = -w / k2
        psi[0, 0] = 0.0
        u[t] = np.fft.irfft2(-1j * ky * psi, s=(ny, nx))
        v[t] = np.fft.irfft2(1j * kx * psi, s=(ny, nx))
    return u.astype(np.float32), v.astype(np.float32)


def synthetic_pair(rng: np.random.Generator, indep: float, ny: int, nx: int,
                   nt: int, sig_coarse: float, sig_fine: float,
                   dx_km: float = 27.7, dy_km: float = 27.8,
                   rho_t: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """u, v whose vorticity has a fine band `1-indep` addressed by the coarse.

    The fine-scale activity is modulated in space by the coarse-band envelope
    with weight (1-indep) and by an independent envelope with weight `indep`,
    so the irrecoverability of fine placement given the coarse field is set by
    construction. The two bands are centred on the two octaves that the
    confirmatory envelope pair spans (200-400 and 400-800 km).
    """
    def ar1(sig: float) -> np.ndarray:
        out = np.empty((nt, ny, nx), dtype=np.float32)
        prev = gaussian_filter(rng.standard_normal((ny, nx)), sig, mode="wrap")
        for t in range(nt):
            new = gaussian_filter(rng.standard_normal((ny, nx)), sig, mode="wrap")
            prev = rho_t * prev + np.sqrt(1 - rho_t ** 2) * new
            out[t] = prev
        return out / (out.std() + EPS)

    coarse = ar1(sig_coarse)
    carrier = ar1(sig_fine)
    alt = np.abs(ar1(sig_coarse))
    cenv = np.abs(gaussian_filter(coarse, (0, sig_coarse, sig_coarse), mode="wrap"))
    cenv /= cenv.std() + EPS
    alt /= alt.std() + EPS
    modul = (1.0 - indep) * cenv + indep * alt
    fine = carrier * modul
    omega = coarse + 0.8 * fine / (fine.std() + EPS)
    return _uv_from_vorticity(omega.astype(np.float32), dx_km, dy_km)


def stage_synthetic() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "synthetic_ladder.json"
    if cache.exists():
        print("cached", flush=True)
        return
    rng = np.random.default_rng(SEED)
    ny, nx, nt = 81, 161, 60
    lat = np.arange(10.0, -10.01, -0.25)
    lon = np.arange(130.0, 170.01, 0.25)
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(ny, nx, 1600.0, dx_km, dy_km)
    # band centres of the two octaves the confirmatory pair spans
    dxm, dym = float(np.mean(dx_km)), float(dy_km)
    sig_coarse = 550.0 / dxm / np.sqrt(12.0)
    sig_fine = 280.0 / dxm / np.sqrt(12.0)

    ladder = np.linspace(0.0, 1.0, 9)
    rows = []
    for indep in ladder:
        for rep in range(20):
            u, v = synthetic_pair(rng, float(indep), ny, nx, nt,
                                  sig_coarse, sig_fine, dxm, dym)
            cur = curvature_profiles(u, v, lat, lon)
            fn = np.asarray(cur["Fnorm_profile"], dtype=float)
            A = float(np.mean([np.log10(max(fn[b], 1e-12)) for b in BANDS_RESOLVED]))
            om = vort(u, v, lat, lon)
            uu = irrecoverability(om, dx_km, dy_km, mask)
            uval = float(np.median(np.mean(uu[:, list(PAIRS_RESOLVED)], axis=1)))
            rows.append({"indep": float(indep), "rep": rep, "A": A, "u": uval})
        print(f"  indep={indep:.3f} done "
              f"A={np.mean([r['A'] for r in rows if r['indep']==indep]):.3f} "
              f"u={np.mean([r['u'] for r in rows if r['indep']==indep]):.3f}",
              flush=True)
    cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# real domains
# --------------------------------------------------------------------------

def stage_real() -> None:
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    files = [(p, "prog") for p in sorted(ERA5_DIRS["prog"].glob("era5_wind850_*.nc"))]
    files += [(p, "lattice") for p in sorted(ERA5_DIRS["lattice"].glob("era5_wind850_*.nc"))]
    for path, kind in files:
        tag = path.stem.replace("era5_wind850_", "")
        cache = OUT / f"u_{tag}.json"
        if cache.exists():
            continue
        ds = xr.open_dataset(path)
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
        v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
        ds.close()
        dx_km, dy_km = _grid_geometry(lat, lon)
        mask = _interior_mask(u.shape[1], u.shape[2], 1600.0, dx_km, dy_km)
        om = vort(u, v, lat, lon)
        uu = irrecoverability(om, dx_km, dy_km, mask)
        rec = {"tag": tag, "region": tag.split("__")[0], "window": tag.split("__")[1],
               "set": kind,
               "u_profile": [float(x) for x in np.median(uu, axis=0)],
               "u": float(np.median(np.mean(uu[:, list(PAIRS_RESOLVED)], axis=1)))}
        cache.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[{tag}] u={rec['u']:.3f}", flush=True)


# --------------------------------------------------------------------------

def _descriptors() -> dict[str, dict]:
    try:
        from clean_experiments.experiment_B11_learned_refinement import descriptors
    except ImportError:  # pragma: no cover
        from experiment_B11_learned_refinement import descriptors  # type: ignore
    return descriptors()


def _profiles_P() -> dict[str, list[float]]:
    out: dict[str, list] = {}
    for f in sorted(Path("clean_experiments/results/experiment_B3_scattering_benchmark")
                    .glob("R*__W*.json")):
        d = json.loads(f.read_text())
        out.setdefault(d["region"], []).append(d["P_real"])
    for f in sorted(Path("clean_experiments/results/experiment_B10_irrecoverability")
                    .glob("desc_*.json")):
        d = json.loads(f.read_text())
        out.setdefault(d["region"], []).append(d["P_real"])
    return {k: list(np.median(np.asarray(v, dtype=float), axis=0)) for k, v in out.items()}


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE12_ESTIMATOR_VALIDITY.md"}

    # ---- H12a ------------------------------------------------------------
    lad = OUT / "synthetic_ladder.json"
    if lad.exists():
        rows = json.loads(lad.read_text())
        ind = np.array([r["indep"] for r in rows])
        Asy = np.array([r["A"] for r in rows])
        usy = np.array([r["u"] for r in rows])
        levels = sorted(set(ind))
        mA = np.array([np.mean(Asy[ind == l]) for l in levels])
        mu = np.array([np.mean(usy[ind == l]) for l in levels])
        res["H12a_synthetic"] = {
            "ladder": [float(x) for x in levels],
            "mean_A": [float(x) for x in mA],
            "mean_u": [float(x) for x in mu],
            "rho_A_vs_independent_fraction": float(spearmanr(levels, mA).statistic),
            "rho_u_vs_independent_fraction": float(spearmanr(levels, mu).statistic),
            "rho_A_all_reps": float(spearmanr(ind, Asy).statistic),
            "rho_u_all_reps": float(spearmanr(ind, usy).statistic),
        }
        res["H12a_pass_A"] = bool(res["H12a_synthetic"]
                                  ["rho_A_vs_independent_fraction"] >= 0.90)
        res["H12a_pass_u"] = bool(res["H12a_synthetic"]
                                  ["rho_u_vs_independent_fraction"] >= 0.90)
    else:
        res["H12a_synthetic"] = None
        res["H12a_pass_A"] = res["H12a_pass_u"] = None

    # ---- real domains ----------------------------------------------------
    per: dict[str, list[float]] = {}
    for f in sorted(OUT.glob("u_*.json")):
        d = json.loads(f.read_text())
        per.setdefault(d["region"], []).append(d["u"])
    desc = _descriptors()
    Pprof = _profiles_P()
    names = sorted(set(per) & set(desc))
    res["n_domains"] = len(names)
    if len(names) >= 8:
        U = np.array([np.median(per[n]) for n in names])
        A = np.array([desc[n]["A"] for n in names])
        LV = np.array([desc[n]["logvar"] for n in names])
        res["domain_values"] = {n: {"A": float(A[i]), "u": float(U[i])}
                                for i, n in enumerate(names)}
        res["H12b_A_vs_u"] = perm_spearman(A, U, rng)
        res["H12c_placebo_logvar_vs_u"] = perm_spearman(LV, U, rng,
                                                        alternative="two-sided")
        res["H12c_pass"] = bool(abs(res["H12b_A_vs_u"]["rho"])
                                > abs(res["H12c_placebo_logvar_vs_u"]["rho"]))
        res["H12b_pass"] = bool(res["H12b_A_vs_u"]["rho"] > 0
                                and res["H12b_A_vs_u"]["p_perm"] < 0.05)
        Pm = np.array([Pprof[n] for n in names if n in Pprof])
        if len(Pm) == len(names):
            ra = loo_pca_residuals(Pm, A[:, None], 2)[:, 0]
            ru = loo_pca_residuals(Pm, U[:, None], 2)[:, 0]
            res["H12d_A_vs_u_beyond_P"] = perm_spearman(ra, ru, rng)
            res["H12d_pass"] = bool(res["H12d_A_vs_u_beyond_P"]["rho"] > 0
                                    and res["H12d_A_vs_u_beyond_P"]["p_perm"] < 0.05)
        res["descriptive_u_vs_prior_targets"] = _u_vs_targets(names, U)

    # ---- verdict ---------------------------------------------------------
    if res.get("H12a_pass_A") is None or res.get("H12b_pass") is None:
        res["PHASE12_VERDICT"] = "INCOMPLETE"
    elif not res["H12a_pass_A"]:
        res["PHASE12_VERDICT"] = "ESTIMATOR_INVALID"
    elif res["H12b_pass"] and res["H12c_pass"]:
        res["PHASE12_VERDICT"] = "ESTIMATOR_VALID"
    else:
        res["PHASE12_VERDICT"] = "ESTIMATOR_WEAK"
    return res


def _u_vs_targets(names: list[str], U: np.ndarray) -> dict:
    """Descriptive only: how u relates to the targets of Phases 9-11."""
    out = {}
    try:
        from clean_experiments.experiment_B11_learned_refinement import load_deficits
    except ImportError:  # pragma: no cover
        from experiment_B11_learned_refinement import load_deficits  # type: ignore
    for s in ("graphcast", "pangu", "hres"):
        d = load_deficits(s, 120)
        y = np.array([np.nanmedian(list(d[n].values())) if n in d else np.nan
                      for n in names])
        m = np.isfinite(y)
        if m.sum() >= 8:
            out[f"deficit_{s}"] = float(spearmanr(U[m], y[m]).statistic)
    p10 = Path("clean_experiments/results/experiment_B10_irrecoverability")
    for key in ("D", "r12"):
        acc: dict[str, list] = {}
        for f in p10.glob("metrics_*.json"):
            r = json.loads(f.read_text())
            acc.setdefault(r["region"], []).append(float(np.mean(r[key][:2])))
        y = np.array([np.median(acc[n]) if n in acc else np.nan for n in names])
        m = np.isfinite(y)
        if m.sum() >= 8:
            out[key] = float(spearmanr(U[m], y[m]).statistic)
    out["_note"] = ("descriptive, inherits the multiplicity of every target "
                    "already tried; cannot be claimed as a positive result")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synthetic", "real", "tests"],
                    default="synthetic")
    args = ap.parse_args()
    if args.stage == "synthetic":
        stage_synthetic()
    elif args.stage == "real":
        stage_real()
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":
    main()
