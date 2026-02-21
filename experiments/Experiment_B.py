#!/usr/bin/env python3
"""
Эксперимент B: x‑локальная диссипация и пространственный баланс энергии
в одномерной спиновой цепочке (без учёта масштабной координаты μ).

Соответствует описанию в "Numerical evidence (experiments).md".
"""

import numpy as np
from scipy.linalg import expm  # не используется напрямую, но может пригодиться

# Фиксируем seed для воспроизводимости
np.random.seed(20260218)

# ------------------------------------------------------------
# Базовые объекты (повтор из эксперимента A для автономности)
# ------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def kronN(ops):
    """Тензорное произведение списка операторов."""
    result = np.array([[1.0 + 0.0j]])
    for op in ops:
        result = np.kron(result, op)
    return result

def op_on_site(op, site, L):
    """Оператор, действующий только на одном сайте (0..L-1)."""
    ops = [I2] * L
    ops[site] = op
    return kronN(ops)

def comm(A, B):
    """Коммутатор матриц."""
    return A @ B - B @ A

def lindblad_dissipator(rho, Ls):
    """Диссипативная часть уравнения Линдблада: сумма по операторам L."""
    dr = np.zeros_like(rho)
    for L in Ls:
        dr += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return dr

def expect(op, rho):
    """Среднее значение оператора."""
    return np.real(np.trace(op @ rho))

# ------------------------------------------------------------
# Параметры цепочки
# ------------------------------------------------------------
L = 4                      # число сайтов
J = 1.0                    # константа связи (например, для XX-модели)

# Строим гамильтониан как сумму связей (bond terms)
# Используем XX-взаимодействие: 0.5 * J * (σ^x_i σ^x_{i+1} + σ^y_i σ^y_{i+1})
# Это стандартный выбор для модели с сохраняющейся намагниченностью по z.
def bond_xx(i, j, L, J=1.0):
    """Оператор связи между сайтами i и j (предполагается j = i+1)."""
    return 0.5 * J * (
        op_on_site(sigma_x, i, L) @ op_on_site(sigma_x, j, L) +
        op_on_site(sigma_y, i, L) @ op_on_site(sigma_y, j, L)
    )

# Список операторов энергии на каждой связи (локальные плотности)
h_bonds = [bond_xx(i, i+1, L, J) for i in range(L-1)]
H = sum(h_bonds)            # полный гамильтониан

# Операторы тока: для каждой внутренней связи ток j_i строится как коммутатор
# плотностей соседних связей: j_i = i [h_i, h_{i+1}] (стандартная формула).
# (Для крайних связей ток не определён, но мы используем их только внутри цепочки)
j_ops = [1j * comm(h_bonds[i], h_bonds[i+1]) for i in range(L-2)]

# ------------------------------------------------------------
# Случайное начальное состояние (матрица плотности)
# ------------------------------------------------------------
dim = 2**L
# Случайная эрмитова положительная матрица
X = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
rho = X @ X.conj().T
rho = rho / np.trace(rho)   # нормировка

# ------------------------------------------------------------
# Диссипация: локальный дефазинг на каждом сайте (операторы σ^z)
# ------------------------------------------------------------
gamma = 0.6                 # скорость дефазинга
Ls_local = [np.sqrt(gamma) * op_on_site(sigma_z, site, L) for site in range(L)]

# Вычисляем полную производную ρ (правая часть Линдблада)
D_rho = lindblad_dissipator(rho, Ls_local)
drho_dt = -1j * comm(H, rho) + D_rho

# ------------------------------------------------------------
# Проверка баланса для каждой связи
# ------------------------------------------------------------
print("=== Эксперимент B: локальный баланс энергии ===")
print(f"L={L}, gamma={gamma}\n")

rows = []
for i in range(L-1):
    h_i = h_bonds[i]
    # Производная среднего от h_i
    dh_dt = expect(h_i, drho_dt)

    # Ток слева и справа
    j_left  = expect(j_ops[i-1], rho) if i-1 >= 0 else 0.0
    j_right = expect(j_ops[i],   rho) if i   <= L-3 else 0.0
    divJ = j_right - j_left

    # Диссипативный источник на этой связи (вклад от дефазинга)
    source = expect(h_i, D_rho)

    # Остаток баланса: d<h_i>/dt + divJ - source должен быть близок к нулю
    residual = dh_dt + divJ - source

    print(f"Связь {i}:")
    print(f"  d<h_i>/dt = {dh_dt:.3e}")
    print(f"  div J     = {divJ:.3e}")
    print(f"  source    = {source:.3e}")
    print(f"  residual  = {residual:.3e}")
    print()

    rows.append({
        "bond": i,
        "dh_dt": dh_dt,
        "divJ": divJ,
        "source": source,
        "residual": residual,
        "abs_residual": abs(residual)
    })

# Также проверим полную энергию
dE_dt = expect(H, drho_dt)
source_total = expect(H, D_rho)
print(f"Полная энергия: d<E>/dt = {dE_dt:.3e}, источник (Tr(H D[ρ])) = {source_total:.3e}")
print(f"Невязка для полной энергии: {dE_dt - source_total:.3e} (должна быть мала)\n")

# Сохраняем результаты
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("expB_balance.csv", index=False)
summary = pd.DataFrame([{
    "gamma": gamma,
    "dE_dt_total": dE_dt,
    "Tr_H_D": source_total,
    "dE_residual": dE_dt - source_total,
    "max_abs_residual": df["abs_residual"].max()
}])
summary.to_csv("expB_summary.csv", index=False)
print("Результаты сохранены в expB_balance.csv и expB_summary.csv")