#!/usr/bin/env python3
"""Shared math/physics utilities for clean experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def kronN(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def op_on_site(op: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    ops = [I2] * n_sites
    ops[site] = op
    return kronN(ops)


def normalize_vec(v: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    return arr / (np.linalg.norm(arr) + eps)


def su2_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    nx, ny, nz = axis
    gen = nx * SX + ny * SY + nz * SZ
    return expm(-0.5j * angle * gen)


def random_su2(rng: np.random.Generator) -> np.ndarray:
    axis = normalize_vec(rng.normal(size=3))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    return su2_from_axis_angle(axis, angle)


def axis_from_unitary(u: np.ndarray) -> np.ndarray:
    anti = (u - u.conj().T) / (2j)
    comps = np.array(
        [
            0.5 * np.real(np.trace(anti @ SX)),
            0.5 * np.real(np.trace(anti @ SY)),
            0.5 * np.real(np.trace(anti @ SZ)),
        ],
        dtype=float,
    )
    if np.linalg.norm(comps) < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return normalize_vec(comps)


def axis_from_hermitian(a: np.ndarray) -> np.ndarray:
    comps = np.array(
        [
            0.5 * np.real(np.trace(a @ SX)),
            0.5 * np.real(np.trace(a @ SY)),
            0.5 * np.real(np.trace(a @ SZ)),
        ],
        dtype=float,
    )
    if np.linalg.norm(comps) < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return normalize_vec(comps)


def projectors_axis(axis: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = axis
    herm = nx * SX + ny * SY + nz * SZ
    _, vecs = np.linalg.eigh(herm)
    p0 = vecs[:, 0:1] @ vecs[:, 0:1].conj().T
    p1 = vecs[:, 1:2] @ vecs[:, 1:2].conj().T
    return p0, p1, vecs


def plaquette(
    ux: dict[tuple[int, int], np.ndarray],
    umu: dict[tuple[int, int], np.ndarray],
    x: int,
    k: int,
    n_x: int,
    n_k: int,
) -> np.ndarray:
    xp = (x + 1) % n_x
    kp = (k + 1) % n_k
    return ux[(x, k)] @ umu[(xp, k)] @ ux[(x, kp)].conj().T @ umu[(x, k)].conj().T


def f_phys_from_plaquette(p: np.ndarray) -> np.ndarray:
    return (p - p.conj().T) / (2j)


def hermitian_part(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def build_u2_connection(beta: float, theta: float, phi: float) -> np.ndarray:
    return beta * I2 + theta * SY + phi * SX


def transport_step(a: np.ndarray, dmu: float) -> np.ndarray:
    return expm(-1j * a * dmu)


def curvature_discrete(a0: np.ndarray, a1: np.ndarray, dmu: float) -> np.ndarray:
    d_a = (a1 - a0) / dmu
    return d_a + comm(a0, a1)


def lambda_matter(f_phys: np.ndarray, rho: np.ndarray) -> float:
    return float(np.real(np.trace(f_phys @ rho)))


def qubit_state_density(alpha: float) -> np.ndarray:
    psi = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)
    return np.outer(psi, psi.conj())


def normalize_rho(rho: np.ndarray) -> np.ndarray:
    tr = np.trace(rho)
    if abs(tr) < 1e-15:
        raise ValueError("Cannot normalize density matrix with zero trace")
    return rho / tr


def lindblad_dissipator(rho: np.ndarray, jump_ops: Sequence[np.ndarray]) -> np.ndarray:
    dr = np.zeros_like(rho)
    for op in jump_ops:
        dr += op @ rho @ op.conj().T - 0.5 * (op.conj().T @ op @ rho + rho @ op.conj().T @ op)
    return dr


def expect(op: np.ndarray, rho: np.ndarray) -> float:
    return float(np.real(np.trace(op @ rho)))


def vn_entropy(rho: np.ndarray, tol: float = 1e-14) -> float:
    herm = hermitian_part(rho)
    vals = np.linalg.eigvalsh(herm)
    vals = vals[vals > tol]
    if vals.size == 0:
        return 0.0
    return float(-np.sum(vals * np.log(vals)))


def coherence_offdiag(rho_1q: np.ndarray) -> float:
    return float(abs(rho_1q[0, 1]))


def insert_bit(rest_idx: int, pos_lsb: int, bit: int) -> int:
    low_mask = (1 << pos_lsb) - 1
    low = rest_idx & low_mask
    high = rest_idx >> pos_lsb
    return (high << (pos_lsb + 1)) | (bit << pos_lsb) | low


def partial_trace_site(rho: np.ndarray, keep_site: int, n_sites: int) -> np.ndarray:
    pos = n_sites - 1 - keep_site
    red = np.zeros((2, 2), dtype=complex)
    dim_rest = 1 << (n_sites - 1)
    for a in (0, 1):
        for b in (0, 1):
            acc = 0.0 + 0.0j
            for r in range(dim_rest):
                i = insert_bit(r, pos, a)
                j = insert_bit(r, pos, b)
                acc += rho[i, j]
            red[a, b] = acc
    return normalize_rho(hermitian_part(red))


def bond_xx_yy(i: int, j: int, n_sites: int, j_coupling: float = 1.0) -> np.ndarray:
    return 0.5 * j_coupling * (
        op_on_site(SX, i, n_sites) @ op_on_site(SX, j, n_sites)
        + op_on_site(SY, i, n_sites) @ op_on_site(SY, j, n_sites)
    )


def random_pure_qubit_density(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    vec = vec / np.linalg.norm(vec)
    return np.outer(vec, vec.conj())


def random_product_state_density(rng: np.random.Generator, n_sites: int) -> np.ndarray:
    return kronN([random_pure_qubit_density(rng) for _ in range(n_sites)])


def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def train_test_r2(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    seed: int = 0,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(train_frac * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    if test_idx.size == 0:
        test_idx = train_idx

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_train_1 = np.column_stack([np.ones(len(x_train)), x_train])
    x_test_1 = np.column_stack([np.ones(len(x_test)), x_test])
    beta = np.linalg.lstsq(x_train_1, y_train, rcond=None)[0]
    pred = x_test_1 @ beta

    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-15))
