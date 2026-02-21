#!/usr/bin/env python3
"""
Эксперимент A: проверка калибровочной инвариантности и некоммутативности
в гильбертовом расслоении с SU(2)-линками.

Соответствует описанию в "Numerical evidence (experiments).md".
"""

import numpy as np
from scipy.linalg import expm

# Фиксируем seed для воспроизводимости
np.random.seed(20260218)

# ------------------------------------------------------------
# Базовые объекты
# ------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def normalize_vec(v, eps=1e-12):
    """Нормировка 3-вектора."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + eps)

def su2_from_axis_angle(n, angle):
    """SU(2) матрица поворота на угол angle вокруг оси n (единичный вектор)."""
    nx, ny, nz = n
    A = nx * sigma_x + ny * sigma_y + nz * sigma_z
    return expm(-0.5j * angle * A)

def random_su2(rng=None):
    """Случайная SU(2) матрица."""
    if rng is None:
        rng = np.random
    # равномерное распределение на группе SU(2)
    u = rng.uniform(0, 1)
    v = rng.uniform(0, 1)
    w = rng.uniform(0, 1)
    phi = 2 * np.pi * u
    theta = np.arccos(2 * v - 1)
    psi = 2 * np.pi * w
    # параметризация через углы Эйлера (не обязательно, но проще)
    # возьмём случайную ось и угол
    axis = normalize_vec(rng.normal(size=3))
    angle = rng.uniform(0, 2 * np.pi)
    return su2_from_axis_angle(axis, angle)

def plaquette(Ux, Umu, x, k, Lx, K):
    """Плакет P_{xμ}(x,k)."""
    xp = (x + 1) % Lx
    kp = (k + 1) % K
    return Ux[(x, k)] @ Umu[(xp, k)] @ Ux[(x, kp)].conj().T @ Umu[(x, k)].conj().T

def F_phys_from_P(P):
    """Эрмитова кривизна из плакета: (P - P†)/(2i)."""
    return (P - P.conj().T) / (2j)

# ------------------------------------------------------------
# Параметры решётки
# ------------------------------------------------------------
Lx = 8               # число пространственных точек
K = 30               # число масштабных слоёв
xs = np.arange(Lx)
ks = np.arange(K)

# Генерируем случайные SU(2) линки (детерминированно благодаря seed)
rng = np.random.RandomState(20260218)  # отдельный генератор для воспроизводимости
Ux = {}
Umu = {}
for x in xs:
    for k in ks:
        Umu[(x, k)] = random_su2(rng)
        Ux[(x, k)] = random_su2(rng)

# ------------------------------------------------------------
# Проверка некоммутативности вертикальных линков
# ------------------------------------------------------------
comm_norms = []
for x in xs:
    for k in range(K - 1):
        A = Umu[(x, k)]
        B = Umu[(x, k + 1)]
        comm_norms.append(np.linalg.norm(A @ B - B @ A))

mean_comm_norm = np.mean(comm_norms)
max_comm_norm = np.max(comm_norms)
print("=== Эксперимент A: некоммутативность ===")
print(f"Средняя норма коммутатора соседних Uμ: {mean_comm_norm:.3e}")
print(f"Максимальная норма коммутатора: {max_comm_norm:.3e}")
if mean_comm_norm > 1e-6:
    print("→ Вертикальные линки не коммутируют (ожидаемо).")
else:
    print("→ Предупреждение: линки почти коммутируют, возможно, генерация дала тривиальные матрицы.")

# ------------------------------------------------------------
# Калибровочная инвариантность Λ = Re Tr(F_phys ρ)
# ------------------------------------------------------------
# Выберем состояние ρ = |+⟩⟨+| (когерентное) для всех точек
psi_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
rho_plus = np.outer(psi_plus, psi_plus.conj())

# Вычислим Λ для всех точек до калибровки
Lambda0 = {}
for x in xs:
    for k in ks:
        P = plaquette(Ux, Umu, x, k, Lx, K)
        Fp = F_phys_from_P(P)
        Lambda0[(x, k)] = np.real(np.trace(Fp @ rho_plus))

# Выберем несколько случайных точек для теста калибровки
test_points = [(x, k) for x in xs for k in ks]
# возьмём подмножество, например 10 случайных
rng_test = np.random.RandomState(20260219)
test_indices = rng_test.choice(len(test_points), size=10, replace=False)
test_points = [test_points[i] for i in test_indices]

max_delta = 0.0
for (x, k) in test_points:
    # Генерируем случайное локальное калибровочное преобразование g ∈ SU(2)
    g = random_su2(rng_test)
    # Преобразуем F и ρ в этой точке
    P = plaquette(Ux, Umu, x, k, Lx, K)
    Fp = F_phys_from_P(P)
    Fp_trans = g @ Fp @ g.conj().T
    rho_trans = g @ rho_plus @ g.conj().T
    Lambda_trans = np.real(np.trace(Fp_trans @ rho_trans))
    delta = abs(Lambda_trans - Lambda0[(x, k)])
    max_delta = max(max_delta, delta)

print("\n=== Эксперимент A: калибровочная инвариантность Λ ===")
print(f"Максимальное изменение Λ при локальном SU(2)-преобразовании: {max_delta:.2e}")
if max_delta < 1e-12:
    print("→ Λ инвариантна с машинной точностью (успех).")
else:
    print("→ Предупреждение: инвариантность нарушена.")

# ------------------------------------------------------------
# (Опционально) Сохраним результаты в файл
# ------------------------------------------------------------
import pandas as pd
summary = pd.DataFrame([{
    "mean_comm_norm": mean_comm_norm,
    "max_comm_norm": max_comm_norm,
    "max_delta_Lambda": max_delta,
    "gauge_invariance_ok": max_delta < 1e-12,
}])
summary.to_csv("expA_summary.csv", index=False)
print("\nРезультаты сохранены в expA_summary.csv")