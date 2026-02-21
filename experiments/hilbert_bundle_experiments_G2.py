#!/usr/bin/env python3
"""
==========================================================================
Численные эксперименты: Гильбертово расслоение и эмерджентная гравитация
==========================================================================
Структура:
    - Эксперимент 1: Непрерывный предел (K слоёв), проверка Lambda ~ sin(omega*Delta_mu)
    - Эксперимент 2: Декогеренция через уравнение Линдблада
    - Эксперимент 3: Закон сохранения (ковариантный и обычный)
    - Эксперимент 4: Профили угловой скорости omega(mu)

Зависимости: numpy, scipy, pandas
"""

import numpy as np
from scipy.linalg import expm
import pandas as pd

# ==========================================================================
# 0. БАЗОВЫЕ ОБЪЕКТЫ
# ==========================================================================

def sigma_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def sigma_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def sigma_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def build_A(beta, theta, phi_angle):
    """
    U(2)-связность: A = beta*I + theta*sigma_y + phi_angle*sigma_x
    beta       — U(1)-компонента (дилатон)
    theta      — SU(2) компонента вдоль sigma_y
    phi_angle  — SU(2) компонента вдоль sigma_x
    """
    return (beta * np.eye(2, dtype=complex)
            + theta * sigma_y()
            + phi_angle * sigma_x())

def transport_step(A, dmu):
    """Унитарный перенос: U = exp(-i * A * dmu)"""
    return expm(-1j * A * dmu)

def curvature_F(A1, A2, dmu):
    """
    Дискретная кривизна на шаге dmu:
        F ~ (A2 - A1)/dmu + [A1, A2]
    """
    dA   = (A2 - A1) / dmu
    comm = A1 @ A2 - A2 @ A1
    return dA + comm

def F_phys(F):
    """Эрмитова часть кривизны: F_phys = (F + F†)/2"""
    return (F + F.conj().T) / 2

def Lambda_matter(F_p, rho):
    """
    Гравитационный источник: Tr(F_phys * rho)
    Скаляр — аналог компоненты T_ab^eff
    """
    return np.real(np.trace(F_p @ rho))

def state_rho(alpha):
    """
    Чистое состояние: |psi> = cos(alpha)|0> + sin(alpha)|1>
    rho = |psi><psi|
    """
    psi = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)
    return np.outer(psi, psi.conj())


# ==========================================================================
# 1. НЕПРЕРЫВНЫЙ ПРЕДЕЛ: K слоёв, постоянная угловая скорость omega
# ==========================================================================

def experiment_1_continuous_limit(omega=0.3, Delta_mu=2.0, beta=0.1, r=0.4,
                                   alpha_state=np.pi/4,
                                   K_values=None):
    """
    Проверяем: Lambda_matter = r * <sigma_x>_rho * sin(omega * Delta_mu)
    при произвольном числе слоёв K.

    Параметры
    ----------
    omega       : угловая скорость вращения A_mu в su(2)
    Delta_mu    : полный масштабный интервал
    beta        : U(1)-компонента (дилатон)
    r           : радиус в su(2) (амплитуда SU(2)-части)
    alpha_state : угол, задающий начальное когерентное состояние
    K_values    : список значений K для проверки сходимости
    """
    if K_values is None:
        K_values = [2, 4, 8, 16, 32, 64, 128]

    rho0 = state_rho(alpha_state)
    sx   = np.real(np.trace(sigma_x() @ rho0))  # <sigma_x>_rho

    rows = []
    for K in K_values:
        dmu    = Delta_mu / K
        mu_arr = np.linspace(0, Delta_mu, K + 1)

        # Строим цепочку связностей
        A_list = []
        for mu in mu_arr:
            angle = omega * mu
            A_list.append(build_A(beta, r * np.cos(angle), r * np.sin(angle)))

        # Накапливаем интеграл кривизны
        Lambda_sum = 0.0
        for k in range(K):
            F  = curvature_F(A_list[k], A_list[k + 1], dmu)
            Fp = F_phys(F)
            Lambda_sum += Lambda_matter(Fp, rho0) * dmu

        theory    = np.sin(omega * Delta_mu) * r * sx
        rel_error = abs(Lambda_sum - theory) / (abs(theory) + 1e-15) * 100

        rows.append({
            'K'               : K,
            'dmu'             : dmu,
            'Lambda_numerical': Lambda_sum,
            'Lambda_theory'   : theory,
            'rel_error_%'     : rel_error
        })

    df = pd.DataFrame(rows)
    df.to_csv('exp1_continuous_limit.csv', index=False)
    print("=== ЭКСПЕРИМЕНТ 1: Непрерывный предел ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print(f"\nТеоретический закон: Lambda = r*<sx>*sin(omega*Delta_mu)"
          f" = {r*sx*np.sin(omega*Delta_mu):.8f}")
    print("Сохранено: exp1_continuous_limit.csv\n")
    return df


# ==========================================================================
# 2. ДЕКОГЕРЕНЦИЯ: уравнение Линдблада
# ==========================================================================

def lindblad_rhs(rho, H, gamma):
    """
    Правая часть уравнения Линдблада:
        drho/dt = -i[H, rho] + gamma*(L rho L† - ½{L†L, rho})
    Оператор дефазирования: L = sqrt(gamma)*sigma_z
    """
    L        = np.sqrt(gamma) * sigma_z()
    comm     = -1j * (H @ rho - rho @ H)
    dissip   = (L @ rho @ L.conj().T
                - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))
    return comm + dissip

def rk4_step(rho, H, gamma, dt):
    """Один шаг метода Рунге–Кутта 4-го порядка для уравнения Линдблада"""
    k1 = lindblad_rhs(rho, H, gamma)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, gamma)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, gamma)
    k4 = lindblad_rhs(rho + dt * k3, H, gamma)
    return rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def experiment_2_decoherence(omega=0.3, r=0.4, beta=0.1,
                              mu0=1.0, dmu_ref=0.5,
                              gamma=0.5, t_max=10.0, n_steps=500):
    """
    Моделируем декогеренцию когерентного состояния.
    Отслеживаем: <sigma_x>, <sigma_y>, Lambda_matter, Tr(rho²) во времени.

    Параметры
    ----------
    mu0     : масштаб, на котором фиксируем связность
    dmu_ref : шаг для вычисления кривизны
    gamma   : скорость дефазирования
    """
    # Фиксированная кривизна на масштабе mu0
    angle0 = omega * mu0
    angle1 = omega * (mu0 + dmu_ref)
    A0 = build_A(beta, r * np.cos(angle0), r * np.sin(angle0))
    A1 = build_A(beta, r * np.cos(angle1), r * np.sin(angle1))
    F0 = curvature_F(A0, A1, dmu_ref)
    Fp0 = F_phys(F0)

    # Гамильтониан = связность на mu0
    H = A0

    rho_t = state_rho(np.pi / 4).copy()
    t_arr = np.linspace(0, t_max, n_steps)
    dt    = t_arr[1] - t_arr[0]

    rows = []
    for t in t_arr:
        sx     = np.real(np.trace(sigma_x() @ rho_t))
        sy     = np.real(np.trace(sigma_y() @ rho_t))
        Lm     = Lambda_matter(Fp0, rho_t)
        purity = np.real(np.trace(rho_t @ rho_t))
        rows.append({'t': t, 'sx': sx, 'sy': sy,
                     'Lambda_matter': Lm, 'purity': purity})
        rho_t = rk4_step(rho_t, H, gamma, dt)

    df = pd.DataFrame(rows)
    df.to_csv('exp2_decoherence.csv', index=False)

    print("=== ЭКСПЕРИМЕНТ 2: Декогеренция (Линдблад) ===")
    for t_target in [0, 2, 5, 10]:
        idx = (df.t - t_target).abs().idxmin()
        row = df.loc[idx]
        print(f"  t={t_target:4.0f}: <sx>={row.sx:.4e}  <sy>={row.sy:.4e}"
              f"  Lambda={row.Lambda_matter:.4e}  purity={row.purity:.6f}")

    ratio = df.Lambda_matter.iloc[-1] / df.Lambda_matter.iloc[0]
    print(f"\nLambda(t_max)/Lambda(0) = {ratio:.3e}  "
          f"(подавление в {1/abs(ratio):.0f} раз)")
    print("Сохранено: exp2_decoherence.csv\n")
    return df


# ==========================================================================
# 3. ЗАКОН СОХРАНЕНИЯ
# ==========================================================================

def build_T_eff_grid(x_grid, rho_x_func, omega, r, beta, k_spatial,
                     K_cont=32, Delta_mu=2.0):
    """
    Вычисляет T_eff(x) = \int d\mu Tr(F_phys(x,\mu) * rho(x))
    на пространственной сетке x_grid.
    """
    dmu_cont = Delta_mu / K_cont
    mu_arr   = np.linspace(0, Delta_mu, K_cont + 1)
    T_vals   = np.zeros(len(x_grid))

    for ix, x in enumerate(x_grid):
        rho_loc = rho_x_func(x)
        Lambda_sum = 0.0
        for k in range(K_cont):
            angle_k  = omega * mu_arr[k]   + k_spatial * x
            angle_k1 = omega * mu_arr[k+1] + k_spatial * x
            A_k  = build_A(beta, r * np.cos(angle_k),  r * np.sin(angle_k))
            A_k1 = build_A(beta, r * np.cos(angle_k1), r * np.sin(angle_k1))
            F    = curvature_F(A_k, A_k1, dmu_cont)
            Fp   = F_phys(F)
            Lambda_sum += Lambda_matter(Fp, rho_loc) * dmu_cont
        T_vals[ix] = Lambda_sum
    return T_vals

def experiment_3_conservation(omega=0.3, r=0.4, beta=0.1,
                               k_spatial=1.0, Nx=50, Delta_mu=2.0, K_cont=32):
    """
    Проверка ковариантного закона сохранения:
        d/dx T_eff + i*Tr([A_x, F_phys]*rho) = 0

    Тестируем два случая:
        (a) Смешанное состояние rho=I/2: T_eff = 0 точно
        (b) Когерентное состояние с параллельным переносом вдоль x
    """
    L_spatial = 2 * np.pi
    x_grid    = np.linspace(0, L_spatial, Nx, endpoint=False)
    dx        = x_grid[1] - x_grid[0]
    dmu_cont  = Delta_mu / K_cont
    mu_arr    = np.linspace(0, Delta_mu, K_cont + 1)

    print("=== ЭКСПЕРИМЕНТ 3: Закон сохранения ===")

    # --- (a) Смешанное состояние ---
    rho_mixed = np.eye(2, dtype=complex) / 2.0
    T_mixed   = build_T_eff_grid(x_grid, lambda x: rho_mixed,
                                  omega, r, beta, k_spatial, K_cont, Delta_mu)
    dT_mixed  = np.gradient(T_mixed, dx)
    print(f"(a) rho=I/2:  max|T_eff|={np.max(np.abs(T_mixed)):.2e}  "
          f"max|dT/dx|={np.max(np.abs(dT_mixed)):.2e}  "
          f"(T_eff=0 точно — классический предел)")

    # --- (b) Когерентное состояние с параллельным переносом ---
    T_cov      = np.zeros(Nx)
    comm_arr   = np.zeros(Nx)
    rho_cov    = state_rho(np.pi / 4).copy()

    for ix, x in enumerate(x_grid):
        # T_eff(x)
        Lambda_sum = 0.0
        F_int = np.zeros((2, 2), dtype=complex)
        for k in range(K_cont):
            angle_k  = omega * mu_arr[k]   + k_spatial * x
            angle_k1 = omega * mu_arr[k+1] + k_spatial * x
            A_k  = build_A(beta, r * np.cos(angle_k),  r * np.sin(angle_k))
            A_k1 = build_A(beta, r * np.cos(angle_k1), r * np.sin(angle_k1))
            F    = curvature_F(A_k, A_k1, dmu_cont)
            Fp   = F_phys(F)
            Lambda_sum += Lambda_matter(Fp, rho_cov) * dmu_cont
            F_int += Fp * dmu_cont
        T_cov[ix] = Lambda_sum

        # Коммутаторный член
        A_x      = build_A(0, r * k_spatial * np.cos(k_spatial * x),
                              r * k_spatial * np.sin(k_spatial * x))
        comm_F   = A_x @ F_int - F_int @ A_x
        comm_arr[ix] = np.real(np.trace(1j * comm_F @ rho_cov))

        # Параллельный перенос rho вдоль x
        U_step = expm(-1j * A_x * dx)
        rho_cov = U_step @ rho_cov @ U_step.conj().T

    dT_cov     = np.gradient(T_cov, dx)
    cov_resid  = dT_cov + comm_arr

    print(f"(b) Когерентное (параллельный перенос):")
    print(f"    max|dT/dx|           = {np.max(np.abs(dT_cov)):.4e}")
    print(f"    max|[A,F]-член|      = {np.max(np.abs(comm_arr)):.4e}")
    print(f"    max|ковар. невязка|  = {np.max(np.abs(cov_resid)):.4e}")
    print(f"    => Ковар. невязка != 0: нужно уравнение Линдблада для rho(x)")

    df = pd.DataFrame({'x': x_grid, 'T_eff': T_cov,
                       'dT_dx': dT_cov, 'comm_term': comm_arr,
                       'cov_residual': cov_resid})
    df.to_csv('exp3_conservation.csv', index=False)
    print("Сохранено: exp3_conservation.csv\n")
    return df


# ==========================================================================
# 4. ПРОФИЛИ УГЛОВОЙ СКОРОСТИ omega(mu)
# ==========================================================================

def experiment_4_omega_profiles(Delta_mu=2.0, K=64, r=0.4, beta=0.1,
                                  alpha_state=np.pi/4):
    """
    Проверяем: Lambda_matter \propto sin(\int omega(\mu) d\mu) * <sigma_x>
    для трёх профилей omega(\mu):
        1. Константа:   omega(\mu) = omega_0
        2. Гауссов пакет: omega(\mu) = omega_0 * exp(-(\mu-\mu_c)^2 / (2*sigma^2))
        3. Осцилляции:  omega(\mu) = omega_0 * cos(k_rg * \mu)
    """
    dmu    = Delta_mu / K
    mu_arr = np.linspace(0, Delta_mu, K + 1)
    rho0   = state_rho(alpha_state)
    sx     = np.real(np.trace(sigma_x() @ rho0))

    # Профили
    omega0  = 0.5
    mu_c    = Delta_mu / 2
    sigma_g = 0.4
    k_rg    = 2.0

    profiles = {
        'constant' : lambda mu: omega0 * np.ones_like(mu),
        'gaussian' : lambda mu: omega0 * np.exp(-(mu - mu_c)**2 / (2 * sigma_g**2)),
        'oscillating': lambda mu: omega0 * np.cos(k_rg * mu)
    }

    rows = []
    for name, omega_func in profiles.items():
        omega_vals = omega_func(mu_arr)

        # Суммарный угол поворота
        phi_total = np.trapz(omega_vals, mu_arr)

        # Строим связности
        A_list = []
        cum_angle = 0.0
        for k_idx, mu in enumerate(mu_arr):
            A_list.append(build_A(beta,
                                   r * np.cos(cum_angle),
                                   r * np.sin(cum_angle)))
            if k_idx < K:
                cum_angle += omega_vals[k_idx] * dmu

        # Накапливаем Lambda
        Lambda_sum = 0.0
        for k in range(K):
            F  = curvature_F(A_list[k], A_list[k+1], dmu)
            Fp = F_phys(F)
            Lambda_sum += Lambda_matter(Fp, rho0) * dmu

        theory    = r * sx * np.sin(phi_total)
        rel_error = abs(Lambda_sum - theory) / (abs(theory) + 1e-15) * 100

        rows.append({
            'profile'         : name,
            'phi_total'       : phi_total,
            'Lambda_numerical': Lambda_sum,
            'Lambda_theory'   : theory,
            'rel_error_%'     : rel_error
        })

    df = pd.DataFrame(rows)
    df.to_csv('exp4_omega_profiles.csv', index=False)
    print("=== ЭКСПЕРИМЕНТ 4: Профили omega(mu) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print("\nПодтверждение: Lambda = r*<sx>*sin(integral omega dmu) для всех профилей")
    print("Сохранено: exp4_omega_profiles.csv\n")
    return df




# ==========================================================================
# 5. G2(μ): НЕОБРАТИМЫЕ ПРЫЖКИ ПО МАСШТАБУ + POINTER ИЗ F_phys
# ==========================================================================

def experiment_5_G2_pointerF(
    # Геометрия/связность по μ
    Delta_mu=2.0,
    K_layers=64,
    omega0=0.5,
    omega_profile='constant',   # 'constant' | 'gaussian' | 'oscillating'
    mu_c=None,
    sigma_g=0.4,
    k_rg=2.0,
    r=0.4,
    beta=0.1,
    # MCWF по μ
    dmu_mcwf=0.03,
    Gamma=0.9,
    Nmu_steps=200,
    Ntraj=800,
    # Начальное состояние
    alpha_state=np.pi/4,
    # Горизонтная энтропия по слоям (микроструктура)
    DeltaS_hor=0.12,
    # Оператор «буст-энергии» для δQ
    K_op='sigma_x',             # 'sigma_x' | 'sigma_y' | 'sigma_z' | 'A_mu0'
    mu0_for_K=1.0,
    seed=20260219,
    out_prefix='exp5_G2_pointerF'
):
    """
    G2(μ) в постановке:
      - μ дискретизуется слоями k=0..K_layers-1
      - F_phys(k) строится из A_k, A_{k+1}
      - pointer-базис = собственный базис F_phys(k)
      - необратимые MCWF-прыжки k->k+1, выбор pointer по Born
      - δQ = Δ<K>, dS_hor = ΔS_hor(k), логируем Λ_matter-показатели

    Сохраняет:
      - {out_prefix}_timeseries.csv
      - {out_prefix}_fit.csv
    """

    rng = np.random.default_rng(seed)

    # --- μ-сетка для геометрии (слои k) ---
    dmu_layer = Delta_mu / K_layers
    mu_arr = np.linspace(0.0, Delta_mu, K_layers + 1)

    if mu_c is None:
        mu_c = Delta_mu / 2

    # --- ω(μ) профиль (как в exp4) ---
    if omega_profile == 'constant':
        omega_vals = omega0 * np.ones_like(mu_arr)
    elif omega_profile == 'gaussian':
        omega_vals = omega0 * np.exp(-(mu_arr - mu_c)**2 / (2 * sigma_g**2))
    elif omega_profile == 'oscillating':
        omega_vals = omega0 * np.cos(k_rg * mu_arr)
    else:
        raise ValueError("omega_profile must be 'constant'|'gaussian'|'oscillating'")

    # --- Строим A_k по cum_angle, как в exp4 ---
    A_list = []
    cum_angle = 0.0
    for k_idx, mu in enumerate(mu_arr):
        A_list.append(build_A(beta, r * np.cos(cum_angle), r * np.sin(cum_angle)))
        if k_idx < K_layers:
            cum_angle += omega_vals[k_idx] * dmu_layer

    # --- Строим F_phys(k) и pointer базисы (evecs[k]) ---
    Fp_list = []
    Evecs = []
    for k in range(K_layers):
        Fk = curvature_F(A_list[k], A_list[k+1], dmu_layer)
        Fp = F_phys(Fk)
        Fp = 0.5 * (Fp + Fp.conj().T)
        _, U = np.linalg.eigh(Fp)
        Fp_list.append(Fp)
        Evecs.append(U)

    # --- Определяем K-оператор (фиксированный для δQ) ---
    if K_op == 'sigma_x':
        Kmat = sigma_x()
    elif K_op == 'sigma_y':
        Kmat = sigma_y()
    elif K_op == 'sigma_z':
        Kmat = sigma_z()
    elif K_op == 'A_mu0':
        k0 = int(np.clip(round(mu0_for_K / dmu_layer), 0, K_layers))
        Kmat = 0.5 * (A_list[k0] + A_list[k0].conj().T)
    else:
        raise ValueError("K_op must be 'sigma_x'|'sigma_y'|'sigma_z'|'A_mu0'")

    # --- Горизонтная энтропия по слоям ---
    S_hor = np.array([k * DeltaS_hor for k in range(K_layers + 1)], dtype=float)
    w_layers = np.exp(-S_hor[:-1])
    w_layers = w_layers / np.sum(w_layers)

    # --- Jump probability per MCWF step ---
    p_jump = min(Gamma * dmu_mcwf, 0.8)

    # --- Инициализация ансамбля чистых состояний ---
    psi0 = np.array([np.cos(alpha_state), np.sin(alpha_state)], dtype=complex)
    psi0 = psi0 / np.linalg.norm(psi0)
    phases = np.exp(1j * rng.uniform(0.0, 2*np.pi, size=Ntraj))
    psis = phases[:, None] * psi0[None, :]
    layer = np.zeros(Ntraj, dtype=int)

    def expvals_many(psis_local, A):
        X = (A @ psis_local.T).T
        return np.real(np.sum(psis_local.conj() * X, axis=1))

    def blocks_and_entropy(psis_local, layer_local):
        blocks = []
        for kk in range(K_layers + 1):
            idx = np.where(layer_local == kk)[0]
            if len(idx) == 0:
                blocks.append(None)
                continue
            Psi = psis_local[idx]
            rho_k = (Psi.conj().T @ Psi) / Ntraj
            blocks.append(rho_k)
        S = 0.0
        eps = 1e-15
        for B in blocks:
            if B is None:
                continue
            Bh = 0.5 * (B + B.conj().T)
            vals = np.real_if_close(np.linalg.eigvalsh(Bh))
            vals = np.maximum(vals, 0.0)
            if np.sum(vals) <= 0:
                continue
            S -= float(np.sum(vals * np.log(vals + eps)))
        return blocks, S

    blocks, S_ens = blocks_and_entropy(psis, layer)

    def l1_coh_from_coeffs(C):
        s = np.sum(np.abs(C), axis=1)
        return (s*s - 1.0).astype(float)

    def lambda_features(psis_local, layer_local):
        p_k = np.array([np.mean(layer_local == kk) for kk in range(K_layers + 1)], float)
        Lam_cond = np.zeros(K_layers)
        Coh_cond = np.zeros(K_layers)
        for kk in range(K_layers):
            idx = np.where(layer_local == kk)[0]
            if len(idx) == 0:
                continue
            Psi = psis_local[idx]
            rho_k_cond = (Psi.conj().T @ Psi) / len(idx)
            Lam_cond[kk] = Lambda_matter(Fp_list[kk], rho_k_cond)
            U = Evecs[kk]
            C = (U.conj().T @ Psi.T).T
            Coh_cond[kk] = float(np.mean(l1_coh_from_coeffs(C)))

        Lambda_w = float(np.sum(w_layers * Lam_cond))
        Lambda_unw = float(np.sum(p_k[:K_layers] * Lam_cond))
        Coh_ptr_w = float(np.sum(w_layers * Coh_cond))
        out = {
            'Lambda_w': Lambda_w,
            'Lambda_unw': Lambda_unw,
            'Coh_ptr_w': Coh_ptr_w,
            'layer_mean': float(np.mean(layer_local)),
        }
        for kk in range(min(10, K_layers + 1)):
            out[f'p_layer{kk}'] = float(p_k[kk])
        return out

    rows = []
    for n in range(Nmu_steps):
        K_before = expvals_many(psis, Kmat)
        layer_before = layer.copy()

        do_jump = rng.random(Ntraj) < p_jump

        for kk in range(K_layers):
            idx = np.where((layer == kk) & do_jump)[0]
            if len(idx) == 0:
                continue
            U = Evecs[kk]
            C = (U.conj().T @ psis[idx].T).T
            probs = np.abs(C)**2
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            rr = rng.random(len(idx))
            cdf = np.cumsum(probs, axis=1)
            choice = (cdf < rr[:, None]).sum(axis=1)
            ci = C[np.arange(len(idx)), choice]
            phase = ci / (np.abs(ci) + 1e-15)
            evec = U[:, choice].T
            psi_proj = phase[:, None] * evec
            U_step = transport_step(A_list[kk], dmu_layer)
            psi_new = (U_step @ psi_proj.T).T
            psi_new = psi_new / np.linalg.norm(psi_new, axis=1)[:, None]
            psis[idx] = psi_new
            layer[idx] = kk + 1

        K_after = expvals_many(psis, Kmat)
        blocks, S_ens_new = blocks_and_entropy(psis, layer)

        dS_ens = S_ens_new - S_ens
        S_ens = S_ens_new

        dQ_avg = float(np.mean(K_after - K_before))
        dQ_in = -dQ_avg

        dS_hor_avg = float(np.mean(S_hor[layer] - S_hor[layer_before]))
        feats = lambda_features(psis, layer)

        rows.append({
            'n': int(n),
            'dmu_mcwf': float(dmu_mcwf),
            'Delta_mu': float(Delta_mu),
            'K_layers': int(K_layers),
            'omega_profile': str(omega_profile),
            'omega0': float(omega0),
            'r': float(r),
            'beta': float(beta),
            'Gamma': float(Gamma),
            'p_jump': float(p_jump),
            'Ntraj': int(Ntraj),
            'alpha_state': float(alpha_state),
            'DeltaS_hor': float(DeltaS_hor),
            'K_op': str(K_op),
            'dS_hor_avg': float(dS_hor_avg),
            'dQ_avg': float(dQ_avg),
            'dQ_in': float(dQ_in),
            'S_ens': float(S_ens),
            'dS_ens': float(dS_ens),
            'jump_rate_emp': float(np.mean(layer != layer_before)),
            **feats
        })

    df = pd.DataFrame(rows)

    # --- Регрессия: dS_hor ~ (1/Teff) dQ_in + b ---
    x = df['dQ_in'].values
    y = df['dS_hor_avg'].values
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = a*x + b
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    fit = pd.DataFrame([{
        'a_1_over_Teff': float(a),
        'b': float(b),
        'r2': float(r2),
        'Teff': float(1.0/a) if abs(a) > 1e-12 else np.nan,
        'mean_jump_rate': float(df.jump_rate_emp.mean()),
        'mean_Lambda_w': float(df.Lambda_w.mean()),
        'mean_Coh_ptr_w': float(df.Coh_ptr_w.mean()),
    }])

    df.to_csv(f'{out_prefix}_timeseries.csv', index=False)
    fit.to_csv(f'{out_prefix}_fit.csv', index=False)

    return df, fit

# ==========================================================================
# MAIN: запуск всех экспериментов
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ГИЛЬБЕРТОВО РАССЛОЕНИЕ: Численные эксперименты")
    print("=" * 70 + "\n")

    df1 = experiment_1_continuous_limit()
    df2 = experiment_2_decoherence()
    df3 = experiment_3_conservation()
    df4 = experiment_4_omega_profiles()

    print("=" * 70)
    print("Все эксперименты завершены. Файлы CSV сохранены.")
    print("=" * 70)
