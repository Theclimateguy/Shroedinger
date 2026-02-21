#!/usr/bin/env python3
"""
Эксперимент F: сравнение семейств вертикальных операторов
(pure transport, pointer из Uμ, pointer из F_{xμ}) при одинаковом правиле η(coh).

Соответствует описанию в "Numerical evidence (experiments).md".
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm, norm
from pathlib import Path
import argparse

# -------------------------
# Базовые матрицы и утилиты
# -------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

def comm(A, B):
    return A @ B - B @ A

def kronN(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out

def op_on_site(op, site, L):
    ops = [I2] * L
    ops[site] = op
    return kronN(ops)

def normalize_vec(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    m = np.linalg.norm(v)
    return v / (m + eps)

def su2_from_axis_angle(n, angle):
    nx, ny, nz = n
    A = nx * SX + ny * SY + nz * SZ
    return expm(-1j * 0.5 * angle * A)

def projectors_axis(n):
    nx, ny, nz = n
    A = nx * SX + ny * SY + nz * SZ
    w, V = np.linalg.eigh(A)
    P0 = V[:, 0:1] @ V[:, 0:1].conj().T
    P1 = V[:, 1:2] @ V[:, 1:2].conj().T
    return P0, P1

def axis_from_U(U):
    A = (U - U.conj().T) / (2j)
    ax = 0.5 * np.real(np.trace(A @ SX))
    ay = 0.5 * np.real(np.trace(A @ SY))
    az = 0.5 * np.real(np.trace(A @ SZ))
    v = np.array([ax, ay, az], dtype=float)
    if np.linalg.norm(v) < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return normalize_vec(v)

def axis_from_Herm(A):
    ax = 0.5 * np.real(np.trace(A @ SX))
    ay = 0.5 * np.real(np.trace(A @ SY))
    az = 0.5 * np.real(np.trace(A @ SZ))
    v = np.array([ax, ay, az], dtype=float)
    if np.linalg.norm(v) < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return normalize_vec(v)

def vn_entropy(rho, tol=1e-14):
    rho = (rho + rho.conj().T) / 2
    w = np.linalg.eigvalsh(rho)
    w = w[w > tol]
    return float(-np.sum(w * np.log(w)))

def coherence_offdiag(rho):
    return float(abs(rho[0, 1]))

def normalize_rho(rho):
    return rho / np.trace(rho)

def lindblad_dissipator(rho, Ls):
    dr = np.zeros_like(rho)
    for L in Ls:
        dr += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return dr

def expect(op, rho):
    return float(np.real(np.trace(op @ rho)))

# -------------------------
# Partial trace (single site)
# -------------------------
def insert_bit(r, pos_lsb, bit):
    low_mask = (1 << pos_lsb) - 1
    low = r & low_mask
    high = r >> pos_lsb
    return (high << (pos_lsb + 1)) | (bit << pos_lsb) | low

def partial_trace_site(rho, keep, L):
    pos = (L - 1 - keep)
    red = np.zeros((2, 2), dtype=complex)
    dim_rest = 1 << (L - 1)
    for a in (0, 1):
        for b in (0, 1):
            s = 0.0 + 0.0j
            for r in range(dim_rest):
                i = insert_bit(r, pos, a)
                j = insert_bit(r, pos, b)
                s += rho[i, j]
            red[a, b] = s
    red = (red + red.conj().T) / 2
    red = red / np.trace(red)
    return red

# -------------------------
# Спин-цепочка
# -------------------------
def bond_xx_yy(i, j, L, J=1.0):
    return 0.5 * J * (op_on_site(SX, i, L) @ op_on_site(SX, j, L) +
                      op_on_site(SY, i, L) @ op_on_site(SY, j, L))

# -------------------------
# Решёточная калибровка
# -------------------------
def plaquette(Ux, Umu, x, k, Lx, K):
    xp = (x + 1) % Lx
    kp = (k + 1) % K
    return Ux[(x, k)] @ Umu[(xp, k)] @ Ux[(x, kp)].conj().T @ Umu[(x, k)].conj().T

def F_phys_from_P(P):
    return (P - P.conj().T) / (2j)

# -------------------------
# Основная функция эксперимента F
# -------------------------
def run_experiment_F(outdir: Path, seed: int = 20260218):
    rng = np.random.default_rng(seed)

    L = 4                     # число сайтов
    K = 4                     # число масштабных слоёв
    xs = np.arange(L)
    ks = np.arange(K)

    # Параметры для генерации SU(2) линков
    def axis_mu(x, k):
        v = [
            np.sin(0.7*x + 0.9*k) + 0.25*np.cos(0.3*x - 0.4*k),
            np.cos(0.6*x - 0.8*k) + 0.15*np.sin(0.9*x + 0.2*k),
            np.sin(0.5*x + 0.5*k) + 0.35,
        ]
        return normalize_vec(v)

    def axis_x(x, k):
        v = [
            np.cos(0.8*x + 0.4*k) + 0.1,
            np.sin(0.5*x - 0.7*k) + 0.15*np.cos(0.9*x + 0.3*k),
            np.cos(0.4*x + 0.6*k) - 0.25,
        ]
        return normalize_vec(v)

    omega = 0.25 + 0.45*np.exp(-0.5*((ks-(K-1)/2)/(0.35*K))**2)
    angle_mu = 0.9 * omega / omega.max()
    angle_x = 0.55 * (0.75 + 0.25*np.cos(2*np.pi*ks/K))

    Ux = {}
    Umu = {}
    for x in xs:
        for k in ks:
            Umu[(x, k)] = su2_from_axis_angle(axis_mu(x, k), float(angle_mu[k]))
            Ux[(x, k)] = su2_from_axis_angle(axis_x(x, k), float(angle_x[k]))

    def P_xmu(x, k):
        xp = (x+1) % L
        kp = (k+1) % K
        return Ux[(x, k)] @ Umu[(xp, k)] @ Ux[(x, kp)].conj().T @ Umu[(x, k)].conj().T

    Fsite = {(x, k): F_phys_from_P(P_xmu(x, k)) for x in xs for k in ks}
    F2site = {(x, k): Fsite[(x, k)] @ Fsite[(x, k)] for x in xs for k in ks}
    Fnorm = {(x, k): float(norm(Fsite[(x, k)])) for x in xs for k in ks}

    # Динамика внутри слоя
    d = 2**L
    D = K * d
    J = 1.0
    h_bonds = [bond_xx_yy(i, i+1, L, J) for i in range(L-1)]
    H_chain = sum(h_bonds)
    j_bonds = [1j * comm(h_bonds[i], h_bonds[i+1]) for i in range(L-2)]

    Pk = [np.eye(K, dtype=complex)[[k]].T @ np.eye(K, dtype=complex)[[k]] for k in range(K)]

    def lift_to_layer(op_chain, k):
        return np.kron(Pk[k], op_chain)

    H_tot = sum(lift_to_layer(H_chain, k) for k in range(K))

    gamma = 0.35
    Ls_deph = [np.sqrt(gamma) * lift_to_layer(op_on_site(SZ, site, L), k)
               for k in range(K) for site in range(L)]

    U_chain = {k: kronN([Umu[(x, k)] for x in xs]) for k in range(K-1)}
    comm_norm_U = float(norm(U_chain[0] @ U_chain[1] - U_chain[1] @ U_chain[0]))

    # Функции для построения вертикальных операторов
    def build_Ls_mu(family, eta):
        Ls = []
        for k in range(K-1):
            shift = np.zeros((K, K), dtype=complex)
            shift[k+1, k] = 1.0
            if family == 'pure_transport':
                Ls.append(np.sqrt(eta[k]) * np.kron(shift, U_chain[k]))
            elif family == 'pointer_from_U':
                Ps = []
                for x in xs:
                    n = axis_from_U(Umu[(x, k)])
                    Ps.append(projectors_axis(n))
                for bits in range(2**L):
                    ops = []
                    for x in range(L):
                        b = (bits >> (L-1-x)) & 1
                        ops.append(Ps[x][b])
                    Pglob = kronN(ops)
                    Ls.append(np.sqrt(eta[k]) * np.kron(shift, Pglob @ U_chain[k]))
            elif family == 'pointer_from_F':
                Ps = []
                for x in xs:
                    n = axis_from_Herm(Fsite[(x, k)])
                    Ps.append(projectors_axis(n))
                for bits in range(2**L):
                    ops = []
                    for x in range(L):
                        b = (bits >> (L-1-x)) & 1
                        ops.append(Ps[x][b])
                    Pglob = kronN(ops)
                    Ls.append(np.sqrt(eta[k]) * np.kron(shift, Pglob @ U_chain[k]))
            else:
                raise ValueError('unknown family')
        return Ls

    families = ['pure_transport', 'pointer_from_U', 'pointer_from_F']
    Ns = 140
    eta0 = 0.9

    def random_pure_qubit_local(rr):
        v = rr.normal(size=2) + 1j * rr.normal(size=2)
        v = v / np.linalg.norm(v)
        return np.outer(v, v.conj())

    def random_product_state_local(rr):
        return kronN([random_pure_qubit_local(rr) for _ in range(L)])

    def linreg_r2(X, y, seed2=0, train_frac=0.7):
        rr = np.random.default_rng(seed2)
        n = len(y)
        idx = np.arange(n)
        rr.shuffle(idx)
        ntr = int(train_frac * n)
        tr = idx[:ntr]
        te = idx[ntr:]
        Xtr = X[tr]
        ytr = y[tr]
        Xte = X[te]
        yte = y[te]
        Xtr1 = np.column_stack([np.ones(len(tr)), Xtr])
        Xte1 = np.column_stack([np.ones(len(te)), Xte])
        beta, *_ = np.linalg.lstsq(Xtr1, ytr, rcond=None)
        pred = Xte1 @ beta
        ss_res = np.sum((yte - pred)**2)
        ss_tot = np.sum((yte - np.mean(yte))**2) + 1e-15
        return float(1 - ss_res / ss_tot)

    summary_rows = []
    for fam in families:
        print(f"Processing family: {fam}")
        rows = []
        max_res = 0.0
        for s in range(Ns):
            rr = np.random.default_rng(10000 + s)
            w = rr.dirichlet(alpha=np.ones(K))
            rho_layers = [random_product_state_local(rr) for _ in range(K)]

            rho_tot = np.zeros((D, D), dtype=complex)
            for k in range(K):
                rho_tot[k*d:(k+1)*d, k*d:(k+1)*d] = w[k] * rho_layers[k]
            rho_tot = normalize_rho(rho_tot)

            coh_layer = [0.0] * K
            for k in range(K):
                rho_k = rho_tot[k*d:(k+1)*d, k*d:(k+1)*d]
                pk_layer = float(np.real(np.trace(rho_k)))
                if pk_layer < 1e-14:
                    continue
                rho_kc = rho_k / pk_layer
                cohs = [coherence_offdiag(partial_trace_site(rho_kc, x, L)) for x in xs]
                coh_layer[k] = float(np.mean(cohs))

            eta = [eta0 * min(1.0, coh_layer[k] / 0.5) for k in range(K-1)]
            Ls_mu = build_Ls_mu(fam, eta)

            unitary_part = -1j * comm(H_tot, rho_tot)
            D_deph = lindblad_dissipator(rho_tot, Ls_deph)
            D_mu = lindblad_dissipator(rho_tot, Ls_mu)
            dr = unitary_part + D_deph + D_mu

            for k in range(K):
                rho_k = rho_tot[k*d:(k+1)*d, k*d:(k+1)*d]
                pk_layer = float(np.real(np.trace(rho_k)))
                if pk_layer < 1e-14:
                    continue
                rho_kc = rho_k / pk_layer

                site_red = {x: partial_trace_site(rho_kc, x, L) for x in xs}
                site_coh = {x: coherence_offdiag(site_red[x]) for x in xs}
                site_L = {x: float(np.real(np.trace(Fsite[(x, k)] @ site_red[x]))) for x in xs}
                site_e = {x: float(0.5 * np.real(np.trace(site_red[x] @ F2site[(x, k)]))) for x in xs}
                site_fn = {x: Fnorm[(x, k)] for x in xs}

                jexp = [expect(lift_to_layer(j_bonds[i], k), rho_tot) for i in range(L-2)]

                for b in range(L-1):
                    O = lift_to_layer(h_bonds[b], k)
                    dO_dt = expect(O, dr)
                    j_i = jexp[b] if b <= L-3 else 0.0
                    j_im1 = jexp[b-1] if b-1 >= 0 else 0.0
                    divJ = j_i - j_im1
                    src_deph = expect(O, D_deph)
                    src_mu = expect(O, D_mu)
                    residual = dO_dt + divJ - (src_deph + src_mu)
                    max_res = max(max_res, abs(residual))

                    x0, x1 = b, b+1
                    rows.append({
                        'family': fam,
                        'sample': s,
                        'k': k,
                        'bond': b,
                        'abs_src_mu': abs(src_mu),
                        'coh_x0': site_coh[x0],
                        'coh_x1': site_coh[x1],
                        'coh_layer_mean': coh_layer[k],
                        'eta': (eta[k] if k < K-1 else 0.0),
                        'absLambda_x0': abs(site_L[x0]),
                        'absLambda_x1': abs(site_L[x1]),
                        'Fnorm_x0': site_fn[x0],
                        'Fnorm_x1': site_fn[x1],
                        'e_x0': site_e[x0],
                        'e_x1': site_e[x1],
                        'residual': residual,
                    })

        df = pd.DataFrame(rows)
        df.to_csv(outdir / f'experiment_F_{fam}_dataset.csv', index=False)

        y = df['abs_src_mu'].values
        X_coh = df[['coh_layer_mean', 'coh_x0', 'coh_x1', 'eta']].values
        X_plus = df[['coh_layer_mean', 'coh_x0', 'coh_x1', 'eta',
                     'absLambda_x0', 'absLambda_x1', 'Fnorm_x0', 'Fnorm_x1',
                     'e_x0', 'e_x1']].values
        r2_coh = linreg_r2(X_coh, y, seed2=1)
        r2_plus = linreg_r2(X_plus, y, seed2=2)

        corr_lam = float(np.corrcoef((df['absLambda_x0'] + df['absLambda_x1']).values, y)[0, 1])
        corr_coh = float(np.corrcoef((df['coh_x0'] + df['coh_x1']).values, y)[0, 1])

        n_ops = (K-1) * (1 if fam == 'pure_transport' else 2**L)
        summary_rows.append({
            'family': fam,
            'max_abs_balance_residual': float(max_res),
            'mean_abs_src_mu': float(np.mean(y)),
            'corr(|Lambda|sum, |src_mu|)': corr_lam,
            'corr(coh_sum, |src_mu|)': corr_coh,
            'R2_coh_only': r2_coh,
            'R2_plus_Lambda_like': r2_plus,
            'n_Lmu_ops_total': int(n_ops),
        })

    summary = pd.DataFrame(summary_rows)
    summary.insert(0, 'noncomm_norm_[U0,U1]', comm_norm_U)
    summary.insert(1, 'gamma_deph', gamma)
    summary.insert(2, 'eta0', eta0)
    summary.to_csv(outdir / 'experiment_F_summary.csv', index=False)

    print("\n=== Эксперимент F завершён ===")
    print(f"CSV файлы сохранены в {outdir.resolve()}")
    print(summary.to_string(index=False))

# -------------------------
# Запуск
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out', help='директория для вывода CSV')
    parser.add_argument('--seed', type=int, default=20260218, help='seed')
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_experiment_F(outdir, seed=args.seed)