#!/usr/bin/env python3
"""
Эксперимент D: полный баланс на (t,x,μ) с прямой суммой гильбертовых пространств
и вертикальными переходами (RG-потоком).

Соответствует описанию в "Numerical evidence (experiments).md" и "Numerical evidence 2.md".
"""

import numpy as np
from scipy.linalg import expm

# Фиксируем seed для воспроизводимости
np.random.seed(20260220)

# ------------------------------------------------------------
# Базовые объекты (повтор из предыдущих экспериментов)
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
    """Оператор, действующий на одном сайте цепочки длины L."""
    ops = [I2] * L
    ops[site] = op
    return kronN(ops)

def comm(A, B):
    return A @ B - B @ A

def lindblad_dissipator(rho, Ls):
    """Диссипативная часть уравнения Линдблада."""
    dr = np.zeros_like(rho)
    for L in Ls:
        dr += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return dr

def expect(op, rho):
    return np.real(np.trace(op @ rho))

def normalize_vec(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + eps)

def su2_from_axis_angle(n, angle):
    """SU(2) матрица поворота на угол angle вокруг оси n."""
    nx, ny, nz = n
    A = nx * sigma_x + ny * sigma_y + nz * sigma_z
    return expm(-0.5j * angle * A)

def random_su2(rng):
    """Случайная SU(2) матрица."""
    axis = normalize_vec(rng.normal(size=3))
    angle = rng.uniform(0, 2 * np.pi)
    return su2_from_axis_angle(axis, angle)

# ------------------------------------------------------------
# Параметры решётки
# ------------------------------------------------------------
Lx = 4                     # число сайтов в цепочке (пространство)
K = 4                      # число масштабных слоёв
J = 1.0                    # константа связи для внутрислойной динамики
gamma = 0.35                # скорость дефазинга внутри слоя
eta0 = 0.9                  # базовая скорость вертикальных переходов

# ------------------------------------------------------------
# Построение SU(2) линков (связности) для всех (x,k)
# ------------------------------------------------------------
xs = np.arange(Lx)
ks = np.arange(K)
rng_links = np.random.RandomState(20260220)

Ux = {}
Umu = {}
for x in xs:
    for k in ks:
        Umu[(x, k)] = random_su2(rng_links)
        Ux[(x, k)] = random_su2(rng_links)

# ------------------------------------------------------------
# Функции для работы с прямой суммой
# ------------------------------------------------------------
def lift_to_layer(op_chain, k, Lx, K):
    """Поднимает оператор цепочки в слой k (блок K×K)."""
    # op_chain имеет размерность 2^{Lx} × 2^{Lx}
    # Создаём оператор на всём пространстве прямой суммы
    dim_layer = 2**Lx
    dim_tot = K * dim_layer
    # Единичная матрица на слоях, но нужен проектор на слой k
    # Проще: построить оператор как блочно-диагональный, где на блоке k стоит op_chain,
    # а на остальных — нули. Но для унитарных операторов перехода между слоями
    # мы будем использовать явные матрицы перехода, а здесь для наблюдаемых:
    # мы можем просто работать со срезом rho_tot[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer]
    # Поэтому эта функция не нужна для expect, мы будем извлекать подматрицу.
    # Оставим заглушку.
    raise NotImplementedError("Используйте прямое индексирование.")

# ------------------------------------------------------------
# Построение гамильтониана и токов внутри слоя
# ------------------------------------------------------------
def bond_xx(i, j, L, J=1.0):
    """Оператор связи между сайтами i и j (XX-взаимодействие)."""
    return 0.5 * J * (
        op_on_site(sigma_x, i, L) @ op_on_site(sigma_x, j, L) +
        op_on_site(sigma_y, i, L) @ op_on_site(sigma_y, j, L)
    )

h_bonds = [bond_xx(i, i+1, Lx, J) for i in range(Lx-1)]
H_chain = sum(h_bonds)
# Токи между связями (стандартная формула)
j_ops = [1j * comm(h_bonds[i], h_bonds[i+1]) for i in range(Lx-2)]

dim_layer = 2**Lx
dim_tot = K * dim_layer

# ------------------------------------------------------------
# Построение вертикальных операторов (jump-операторы между слоями)
# ------------------------------------------------------------
# Для каждого слоя k (от 0 до K-2) строим унитарный оператор V_k,
# который переводит состояние из слоя k в слой k+1 с учётом связности Umu.
# V_k действует как тензорное произведение Umu[(x,k)] по всем x.
V_ops = []
for k in range(K-1):
    # строим унитарную матрицу на dim_layer, как тензорное произведение по x
    V_list = [Umu[(x, k)] for x in xs]
    Vk = kronN(V_list)   # размерность dim_layer × dim_layer
    # теперь нужно вставить эту матрицу в полное пространство как оператор перехода
    # из слоя k в слой k+1. Для этого создадим матрицу размера dim_tot × dim_tot
    # с единственным блоком (k+1, k) равным Vk.
    V_full = np.zeros((dim_tot, dim_tot), dtype=complex)
    V_full[(k+1)*dim_layer:(k+2)*dim_layer, k*dim_layer:(k+1)*dim_layer] = Vk
    V_ops.append(V_full)

# Вертикальный jump-оператор (неэрмитов) будет sqrt(eta) * V_full.
# Позже мы определим eta(k) на основе когерентности, но здесь для простоты
# возьмём постоянную eta = eta0.

# ------------------------------------------------------------
# Построение диссипативных операторов внутри слоя (локальный дефазинг)
# ------------------------------------------------------------
Ls_deph = []
for k in range(K):
    for site in range(Lx):
        L = np.sqrt(gamma) * op_on_site(sigma_z, site, Lx)
        # помещаем в блок k
        L_full = np.zeros((dim_tot, dim_tot), dtype=complex)
        L_full[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer] = L
        Ls_deph.append(L_full)

# ------------------------------------------------------------
# Функция для генерации случайного состояния (прямая сумма)
# ------------------------------------------------------------
def random_product_state(rng):
    """Случайное чистое произведение состояний на каждом сайте."""
    psi_list = []
    for _ in range(Lx):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v = v / np.linalg.norm(v)
        psi_list.append(v.reshape(2,1))
    psi = kronN([v.flatten() for v in psi_list])  # вектор-столбец
    return psi

def random_density_layer(rng):
    """Случайная матрица плотности на одном слое (цепочка)."""
    psi = random_product_state(rng)
    return np.outer(psi, psi.conj())

# ------------------------------------------------------------
# Инициализация состояния: случайная прямая сумма с весами
# ------------------------------------------------------------
rng_state = np.random.RandomState(20260221)
w = rng_state.dirichlet(alpha=np.ones(K))   # веса слоёв
rho_tot = np.zeros((dim_tot, dim_tot), dtype=complex)
for k in range(K):
    rho_layer = random_density_layer(rng_state)
    rho_tot[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer] = w[k] * rho_layer
# нормировка (уже должна быть нормирована, но проверим)
rho_tot = rho_tot / np.trace(rho_tot)

# ------------------------------------------------------------
# Вычисление баланса
# ------------------------------------------------------------
# Полный гамильтониан (блочно-диагональный)
H_tot = np.zeros((dim_tot, dim_tot), dtype=complex)
for k in range(K):
    H_tot[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer] = H_chain

# Вычисляем правую часть уравнения Линдблада
unitary_part = -1j * comm(H_tot, rho_tot)
D_deph = lindblad_dissipator(rho_tot, Ls_deph)

# Определим eta на основе когерентности (как в эксперименте E)
# Для простоты возьмём eta постоянную
eta = eta0
Ls_mu = [np.sqrt(eta) * V for V in V_ops]
D_mu = lindblad_dissipator(rho_tot, Ls_mu)

drho_dt = unitary_part + D_deph + D_mu

# ------------------------------------------------------------
# Проверка баланса для каждой связи в каждом слое
# ------------------------------------------------------------
print("=== Эксперимент D: полный баланс на (t,x,μ) ===\n")
max_residual = 0.0
rows = []
for k in range(K):
    # извлекаем блок для слоя k
    rho_k = rho_tot[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer]
    pk = np.trace(rho_k).real
    if pk < 1e-12:
        continue
    rho_kc = rho_k / pk   # условная матрица плотности

    # токи на этом слое (вычисляем как средние по полной матрице, но они локализованы в блоке)
    j_vals = [expect(j_ops[i], rho_kc) for i in range(Lx-2)]

    for b in range(Lx-1):
        h_b = h_bonds[b]
        # среднее и производная
        dh_dt = expect(h_b, drho_dt[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer])  # не совсем точно, надо брать проекцию drho_dt на блок?
        # правильнее: expect на полном пространстве, но h_b поднято?
        # Упростим: будем использовать expect для подматрицы, но учтём, что drho_dt имеет недиагональные блоки,
        # которые не вносят вклад в expect диагонального оператора. Так что можно взять диагональный блок.
        # Создадим оператор h_b на полном пространстве
        h_full = np.zeros((dim_tot, dim_tot), dtype=complex)
        h_full[k*dim_layer:(k+1)*dim_layer, k*dim_layer:(k+1)*dim_layer] = h_b
        dh_dt_full = expect(h_full, drho_dt)
        # divJ
        j_left  = j_vals[b-1] if b-1 >= 0 else 0.0
        j_right = j_vals[b]   if b   <= Lx-3 else 0.0
        divJ = j_right - j_left

        # источники
        src_deph = expect(h_full, D_deph)
        src_mu   = expect(h_full, D_mu)
        residual = dh_dt_full + divJ - (src_deph + src_mu)

        max_residual = max(max_residual, abs(residual))

        rows.append({
            "k": k,
            "bond": b,
            "dh_dt": dh_dt_full,
            "divJ": divJ,
            "src_deph": src_deph,
            "src_mu": src_mu,
            "residual": residual,
            "abs_residual": abs(residual)
        })

        print(f"k={k}, bond={b}: residual = {residual:.3e}")

print(f"\nМаксимальный остаток баланса: {max_residual:.3e}")
if max_residual < 1e-12:
    print("→ Баланс выполняется с машинной точностью (успех).")
else:
    print("→ Предупреждение: баланс нарушен, невязка велика.")

# ------------------------------------------------------------
# Сохранение результатов
# ------------------------------------------------------------
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("expD_balance.csv", index=False)

summary = pd.DataFrame([{
    "Lx": Lx,
    "K": K,
    "gamma": gamma,
    "eta": eta,
    "max_residual": max_residual,
    "balance_ok": max_residual < 1e-12
}])
summary.to_csv("expD_summary.csv", index=False)

print("\nРезультаты сохранены в expD_balance.csv и expD_summary.csv")