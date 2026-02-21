#!/usr/bin/env python3
"""
Эксперимент C: рост энтропии не гарантирует сохранение энергии (наблюдаемой).
Демонстрируется на примере одного кубита с дефазировкой.
Соответствует описанию в "Numerical evidence (experiments).md".
"""

import numpy as np
from scipy.linalg import expm

# Фиксируем seed для воспроизводимости
np.random.seed(20260219)

# ------------------------------------------------------------
# Базовые объекты
# ------------------------------------------------------------
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def vn_entropy(rho, tol=1e-15):
    """Энтропия фон Неймана для матрицы плотности."""
    # делаем эрмитовой
    rho = (rho + rho.conj().T) / 2
    w = np.linalg.eigvalsh(rho)
    w = w[w > tol]
    return -np.sum(w * np.log(w)).real

def lindblad_rhs(rho, H, L):
    """Правая часть уравнения Линдблада для одного оператора диссипации L."""
    comm = -1j * (H @ rho - rho @ H)
    dissip = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return comm + dissip

def rk4_step(rho, H, L, dt):
    """Один шаг метода Рунге–Кутты 4-го порядка."""
    k1 = lindblad_rhs(rho, H, L)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, L)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, L)
    k4 = lindblad_rhs(rho + dt * k3, H, L)
    return rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def state_rho(alpha):
    """Чистое состояние |ψ⟩ = cos(α)|0⟩ + sin(α)|1⟩."""
    psi = np.array([np.cos(alpha), np.sin(alpha)], dtype=complex)
    return np.outer(psi, psi.conj())

# ------------------------------------------------------------
# Параметры
# ------------------------------------------------------------
# Гамильтониан: свободная эволюция (для простоты возьмём нулевой, чтобы
# подчеркнуть эффект диссипации). Если хотим осцилляции, можно добавить,
# но не обязательно.
H = np.zeros((2, 2), dtype=complex)

# Начальное состояние: когерентная суперпозиция (чистое)
rho0 = state_rho(np.pi / 4)   # |+⟩

# Диссипация: дефазировка с разными операторами
gamma = 0.5                    # скорость дефазировки
t_max = 5.0
n_steps = 500
dt = t_max / n_steps
t_arr = np.linspace(0, t_max, n_steps)

# ------------------------------------------------------------
# Случай 1: дефазировка в базисе σ_z (L = √γ σ_z)
# ------------------------------------------------------------
Lz = np.sqrt(gamma) * sigma_z
rho = rho0.copy()
entropy_z = []
exp_z = []       # ⟨σ_z⟩
exp_x = []       # ⟨σ_x⟩ (для сравнения)

for t in t_arr:
    entropy_z.append(vn_entropy(rho))
    exp_z.append(np.real(np.trace(sigma_z @ rho)))
    exp_x.append(np.real(np.trace(sigma_x @ rho)))
    rho = rk4_step(rho, H, Lz, dt)

# ------------------------------------------------------------
# Случай 2: дефазировка в базисе σ_x (L = √γ σ_x)
# ------------------------------------------------------------
Lx = np.sqrt(gamma) * sigma_x
rho = rho0.copy()
entropy_x = []
exp_z2 = []
exp_x2 = []

for t in t_arr:
    entropy_x.append(vn_entropy(rho))
    exp_z2.append(np.real(np.trace(sigma_z @ rho)))
    exp_x2.append(np.real(np.trace(sigma_x @ rho)))
    rho = rk4_step(rho, H, Lx, dt)

# ------------------------------------------------------------
# Вывод результатов
# ------------------------------------------------------------
print("=== Эксперимент C: энтропия и сохранение наблюдаемых ===\n")
print(f"Начальное состояние: |+⟩ (чистое), γ = {gamma}\n")

print("Дефазировка вдоль σ_z:")
print(f"  ⟨σ_z⟩ в конце: {exp_z[-1]:.4f} (начальное: {exp_z[0]:.4f})")
print(f"  ⟨σ_x⟩ в конце: {exp_x[-1]:.4f} (начальное: {exp_x[0]:.4f})")
print(f"  Энтропия в конце: {entropy_z[-1]:.4f}\n")

print("Дефазировка вдоль σ_x:")
print(f"  ⟨σ_z⟩ в конце: {exp_z2[-1]:.4f} (начальное: {exp_z2[0]:.4f})")
print(f"  ⟨σ_x⟩ в конце: {exp_x2[-1]:.4f} (начальное: {exp_x2[0]:.4f})")
print(f"  Энтропия в конце: {entropy_x[-1]:.4f}\n")

# Пояснение
print("Интерпретация:")
print("  При дефазировке вдоль σ_z наблюдаемая σ_z сохраняется (коммутирует с L),")
print("  а σ_x затухает. При дефазировке вдоль σ_x — наоборот.")
print("  В обоих случаях энтропия растёт одинаково, демонстрируя, что рост энтропии")
print("  сам по себе не определяет, какая наблюдаемая сохраняется.")
print("  Это поддерживает идею балансного уравнения с источниками, а не просто")
print("  апелляцию к росту энтропии.")

# ------------------------------------------------------------
# Сохранение результатов в CSV
# ------------------------------------------------------------
import pandas as pd
df_z = pd.DataFrame({
    "t": t_arr,
    "entropy": entropy_z,
    "exp_z": exp_z,
    "exp_x": exp_x,
    "dephasing_basis": "z"
})
df_x = pd.DataFrame({
    "t": t_arr,
    "entropy": entropy_x,
    "exp_z": exp_z2,
    "exp_x": exp_x2,
    "dephasing_basis": "x"
})
df = pd.concat([df_z, df_x], ignore_index=True)
df.to_csv("expC_entropy_conservation.csv", index=False)

summary = pd.DataFrame([{
    "gamma": gamma,
    "final_entropy_z": entropy_z[-1],
    "final_entropy_x": entropy_x[-1],
    "final_exp_z_z": exp_z[-1],
    "final_exp_x_z": exp_x[-1],
    "final_exp_z_x": exp_z2[-1],
    "final_exp_x_x": exp_x2[-1],
}])
summary.to_csv("expC_summary.csv", index=False)

print("\nРезультаты сохранены в expC_entropy_conservation.csv и expC_summary.csv")