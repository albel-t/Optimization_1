import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ========================================================
# 1. КЛАСС СИМПЛЕКС-ТАБЛИЦЫ И ДВУХФАЗНОГО СИМПЛЕКС-МЕТОДА
# ========================================================
class SimplexMethod:
    def __init__(self, c, A_ub, b_ub, A_eq=None, b_eq=None, var_names=None):
        self.c = np.array(c, dtype=float)
        self.A_ub = np.array(A_ub, dtype=float)
        self.b_ub = np.array(b_ub, dtype=float)
        self.trajectory = []  # Траектория вершин (x1, x2)

    def print_table(self, table, basis, col_names, phase_name, iteration):
        print(f"\n--- {phase_name}: Итерация {iteration} ---")
        header = f"{'Базис':<8} | {'Значение':<10} | " + " | ".join([f"{name:<8}" for name in col_names])
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for i in range(len(basis)):
            row_str = f"{basis[i]:<8} | {table[i, 0]:<10.4f} | " + " | ".join(
                [f"{table[i, j + 1]:<8.4f}" for j in range(len(col_names))])
            print(row_str)

        print("-" * len(header))
        # Строка -f0
        row_f0 = f"{'-f0(X)':<8} | {table[-2, 0]:<10.4f} | " + " | ".join(
            [f"{table[-2, j + 1]:<8.4f}" for j in range(len(col_names))])
        print(row_f0)

        # Если Фаза 1, выводим строку -W
        if table.shape[0] - len(basis) == 2:
            row_W = f"{'-W':<8} | {table[-1, 0]:<10.4f} | " + " | ".join(
                [f"{table[-1, j + 1]:<8.4f}" for j in range(len(col_names))])
            print(row_W)
        print("-" * len(header))

    def solve(self):
        # 1. Построение начальной симплекс-таблицы (Вариант 2)
        # Ограничения:
        # 2.3*x1 + 13.2*x2 >= 23.1 -> 2.3*x1 + 13.2*x2 - x3 + x7 = 23.1 (искусственная x7)
        # 8.2*x1 + 23.1*x2 <= 134.2 -> 8.2*x1 + 23.1*x2 + x4 = 134.2
        # -3.1*x1 + 8.3*x2 <= 2.5 -> -3.1*x1 + 8.3*x2 + x5 = 2.5
        # 8.3*x1 - 13.1*x2 <= 34.2 -> 8.3*x1 - 13.1*x2 + x6 = 34.2
        # Базис: [x7, x4, x5, x6]. x7 — искусственная переменная.

        col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
        basis = ['x7', 'x4', 'x5', 'x6']

        # Строки: [x7, x4, x5, x6, -f0, -W]
        # Столбцы: [Значение, x1, x2, x3, x4, x5, x6, x7]
        table = np.zeros((6, 8), dtype=float)

        # Строка x7 (искусственная)
        table[0, :] = [23.1, 2.3, 13.2, -1.0, 0.0, 0.0, 0.0, 1.0]
        # Строка x4
        table[1, :] = [134.2, 8.2, 23.1, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Строка x5
        table[2, :] = [2.5, -3.1, 8.3, 0.0, 0.0, 1.0, 0.0, 0.0]
        # Строка x6
        table[3, :] = [34.2, 8.3, -13.1, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Строка -f0: c = [-0.9, -0.3] -> -f0 = -0.9*x1 - 0.3*x2
        table[4, :] = [0.0, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Строка -W: W = x7 = 23.1 - 2.3*x1 - 13.2*x2 + x3 -> -W = -23.1 + 2.3*x1 + 13.2*x2 - x3
        table[5, :] = [-23.1, 2.3, 13.2, -1.0, 0.0, 0.0, 0.0, 0.0]

        # ==========================================
        # ФАЗА 1: Минимизация W
        # ==========================================
        iter_p1 = 1
        while True:
            self.print_table(table, basis, col_names, "Фаза 1", iter_p1)

            # Проверка критерия оптимальности Фазы 1 (коэффициенты при переменных в строке -W >= 0)
            row_W_vars = table[-1, 1:len(col_names) + 1]
            if np.all(row_W_vars <= 1e-7):  # Для минимизации -W все коэффициенты должны быть <= 0
                print("\n>>> Фаза 1 завершена: W = 0, найдено допустимое базисное решение!")
                break

            # Ведущий столбец (наибольший положительный для -W)
            pivot_col = np.argmax(row_W_vars) + 1

            # Симплекс-отношения
            ratios = []
            for i in range(len(basis)):
                a_ij = table[i, pivot_col]
                b_i = table[i, 0]
                if a_ij > 1e-7:
                    ratios.append(b_i / a_ij)
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                raise ValueError("Задача не ограничена!")

            pivot_row = np.argmin(ratios)
            print(
                f"Ведущий элемент: строка {basis[pivot_row]}, столбец {col_names[pivot_col - 1]} (значение = {table[pivot_row, pivot_col]:.4f})")

            # Ввод новой переменной в базис
            basis[pivot_row] = col_names[pivot_col - 1]

            # Жорданово исключение
            pivot_val = table[pivot_row, pivot_col]
            table[pivot_row, :] /= pivot_val
            for i in range(table.shape[0]):
                if i != pivot_row:
                    table[i, :] -= table[i, pivot_col] * table[pivot_row, :]

            iter_p1 += 1

        # Извлечение точки (x1, x2) после Фазы 1
        pt = self.extract_point(table, basis, col_names)
        self.trajectory.append(pt)

        # Удаление искусственной переменной x7 и строки -W
        table = table[:-1, :-1]
        col_names = col_names[:-1]

        # ==========================================
        # ФАЗА 2: Минимизация f0(X)
        # ==========================================
        iter_p2 = 1
        while True:
            self.print_table(table, basis, col_names, "Фаза 2", iter_p2)

            pt = self.extract_point(table, basis, col_names)
            if len(self.trajectory) == 0 or np.linalg.norm(np.array(pt) - np.array(self.trajectory[-1])) > 1e-5:
                self.trajectory.append(pt)

            # Проверка критерия оптимальности Фазы 2 (коэффициенты в строке -f0 >= 0 для минимизации)
            row_f0_vars = table[-1, 1:len(col_names) + 1]
            if np.all(row_f0_vars >= -1e-7):
                print("\n>>> Оптимальное решение найдено!")
                break

            pivot_col = np.argmin(row_f0_vars) + 1

            ratios = []
            for i in range(len(basis)):
                a_ij = table[i, pivot_col]
                b_i = table[i, 0]
                if a_ij > 1e-7:
                    ratios.append(b_i / a_ij)
                else:
                    ratios.append(np.inf)

            if all(r == np.inf for r in ratios):
                raise ValueError("Целевая функция не ограничена снизу!")

            pivot_row = np.argmin(ratios)
            print(
                f"Ведущий элемент: строка {basis[pivot_row]}, столбец {col_names[pivot_col - 1]} (значение = {table[pivot_row, pivot_col]:.4f})")

            basis[pivot_row] = col_names[pivot_col - 1]

            pivot_val = table[pivot_row, pivot_col]
            table[pivot_row, :] /= pivot_val
            for i in range(table.shape[0]):
                if i != pivot_row:
                    table[i, :] -= table[i, pivot_col] * table[pivot_row, :]

            iter_p2 += 1

        opt_pt = self.extract_point(table, basis, col_names)
        opt_val = -table[-1, 0]
        return opt_pt, opt_val, self.trajectory

    def extract_point(self, table, basis, col_names):
        x1, x2 = 0.0, 0.0
        if 'x1' in basis:
            x1 = table[basis.index('x1'), 0]
        if 'x2' in basis:
            x2 = table[basis.index('x2'), 0]
        return [x1, x2]


# ========================================================
# 2. ГЕОМЕТРИЧЕСКАЯ ИЛЛЮСТРАЦИЯ ЗАДАЧИ ЛП
# ========================================================
def plot_linear_programming(trajectory, opt_pt, opt_val):
    fig, ax = plt.subplots(figsize=(10, 8))

    x1_vals = np.linspace(-0.5, 12.0, 400)

    # 1. Границы ограничений:
    # Ограничение 1: 2.3*x1 + 13.2*x2 = 23.1  =>  x2 = (23.1 - 2.3*x1) / 13.2
    line1 = (23.1 - 2.3 * x1_vals) / 13.2
    # Ограничение 2: 8.2*x1 + 23.1*x2 = 134.2 =>  x2 = (134.2 - 8.2*x1) / 23.1
    line2 = (134.2 - 8.2 * x1_vals) / 23.1
    # Ограничение 3: -3.1*x1 + 8.3*x2 = 2.5  =>  x2 = (2.5 + 3.1*x1) / 8.3
    line3 = (2.5 + 3.1 * x1_vals) / 8.3
    # Ограничение 4: 8.3*x1 - 13.1*x2 = 34.2 =>  x2 = (8.3*x1 - 34.2) / 13.1
    line4 = (8.3 * x1_vals - 34.2) / 13.1

    ax.plot(x1_vals, line1, color='crimson', linewidth=2, label=r'$2.3x_1 + 13.2x_2 = 23.1$')
    ax.plot(x1_vals, line2, color='navy', linewidth=2, label=r'$8.2x_1 + 23.1x_2 = 134.2$')
    ax.plot(x1_vals, line3, color='darkorange', linewidth=2, label=r'$-3.1x_1 + 8.3x_2 = 2.5$')
    ax.plot(x1_vals, line4, color='green', linewidth=2, label=r'$8.3x_1 - 13.1x_2 = 34.2$')

    # 2. Построение допустимого многоугольника (вершины):
    # Найдем точки пересечения прямых, образующих ОДР
    # V1: пересечение L1 и L3 (оптимум)
    # V2: пересечение L1 и L2
    # V3: пересечение L2 и L4
    # V4: пересечение L3 и L4 (может быть вне ОДР, проверим)
    # V5: пересечение L2 с осью x2 (x1=0)
    # V6: пересечение L4 с осью x1 (x2=0)

    # Решаем системы уравнений для вершин
    def intersect(line1_coeffs, line2_coeffs):
        a1, b1, c1 = line1_coeffs
        a2, b2, c2 = line2_coeffs
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            return None
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return [x, y]

    # L1: 2.3x1 + 13.2x2 = 23.1
    # L2: 8.2x1 + 23.1x2 = 134.2
    # L3: -3.1x1 + 8.3x2 = 2.5
    # L4: 8.3x1 - 13.1x2 = 34.2

    vertices = []
    # V1: L1 & L3 (оптимум)
    v1 = intersect((2.3, 13.2, 23.1), (-3.1, 8.3, 2.5))
    if v1: vertices.append(v1)
    # V2: L1 & L2
    v2 = intersect((2.3, 13.2, 23.1), (8.2, 23.1, 134.2))
    if v2: vertices.append(v2)
    # V3: L2 & L4
    v3 = intersect((8.2, 23.1, 134.2), (8.3, -13.1, 34.2))
    if v3: vertices.append(v3)
    # V4: L3 & L4
    v4 = intersect((-3.1, 8.3, 2.5), (8.3, -13.1, 34.2))
    if v4: vertices.append(v4)
    # V5: L2 & x2-axis (x1=0)
    v5 = [0, 134.2 / 23.1]
    if v5[1] >= 0: vertices.append(v5)
    # V6: L4 & x1-axis (x2=0)
    v6 = [34.2 / 8.3, 0]
    if v6[0] >= 0: vertices.append(v6)

    # Фильтруем вершины, проверяя их на допустимость (все ограничения)
    def is_feasible(pt):
        x1, x2 = pt
        if x1 < -1e-5 or x2 < -1e-5: return False
        if 2.3 * x1 + 13.2 * x2 < 23.1 - 1e-5: return False
        if 8.2 * x1 + 23.1 * x2 > 134.2 + 1e-5: return False
        if -3.1 * x1 + 8.3 * x2 > 2.5 + 1e-5: return False
        if 8.3 * x1 - 13.1 * x2 > 34.2 + 1e-5: return False
        return True

    vertices = [v for v in vertices if is_feasible(v)]
    # Упорядочим вершины по полярному углу для построения полигона
    if vertices:
        cx = sum(v[0] for v in vertices) / len(vertices)
        cy = sum(v[1] for v in vertices) / len(vertices)
        vertices.sort(key=lambda v: np.arctan2(v[1] - cy, v[0] - cx))
        vertices_np = np.array(vertices)
        poly = Polygon(vertices_np, closed=True, facecolor='#a8e6cf', edgecolor='green', alpha=0.45, linewidth=2,
                       label='Допустимая область (ОДР)')
        ax.add_patch(poly)

    # 3. Линии уровня целевой функции f0(X) = -0.9*x1 - 0.3*x2
    X1, X2 = np.meshgrid(np.linspace(0, 12, 200), np.linspace(0, 6, 200))
    Z = -0.9 * X1 - 0.3 * X2
    cs = ax.contour(X1, X2, Z, levels=12, cmap='viridis', linestyles='--', alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # 4. Вектор антиградиента (направление скорейшего убывания f0 -> min)
    # grad f0 = [-0.9, -0.3], направление спуска = [0.9, 0.3]
    grad_dir = np.array([0.9, 0.3])
    grad_dir = grad_dir / np.linalg.norm(grad_dir) * 2.5
    ax.arrow(4.0, 3.0, grad_dir[0], grad_dir[1], head_width=0.25, head_length=0.35, fc='purple', ec='purple',
             linewidth=2, label=r'Направление $-\nabla f_0$ (к минимуму)')

    # 5. Траектория симплекс-метода
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], 'bo-', linewidth=2.5, markersize=8, label='Путь симплекс-метода')
    for idx, pt in enumerate(traj):
        ax.annotate(f" Шаг {idx + 1}: ({pt[0]:.2f}, {pt[1]:.2f})", (pt[0], pt[1]), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, fontweight='bold')

    # 6. Оптимальная точка
    ax.scatter(opt_pt[0], opt_pt[1], color='red', marker='*', s=260, edgecolors='black', zorder=6,
               label=f'Оптимум X* = [{opt_pt[0]:.3f}, {opt_pt[1]:.3f}]\nf0(X*) = {opt_val:.3f}')

    ax.set_title('Геометрическая иллюстрация решения задачи ЛП симплекс-методом\n(Вариант 2)', fontsize=13,
                 fontweight='bold')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_xlim([-0.5, 12.0])
    ax.set_ylim([-0.5, 6.5])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()


# ========================================================
# 3. ТОЧКА ВХОДА
# ========================================================
def main():
    print("=" * 80)
    print(" ЛАБОРАТОРНАЯ РАБОТА 2.6: СИМПЛЕКС-МЕТОД (ВАРИАНТ 2)")
    print("=" * 80)

    # Параметры задачи
    c = [-0.9, -0.3]
    A_ub = [[-2.3, -13.2], [8.2, 23.1], [-3.1, 8.3], [8.3, -13.1]]
    b_ub = [-23.1, 134.2, 2.5, 34.2]

    solver = SimplexMethod(c, A_ub, b_ub)
    opt_pt, opt_val, trajectory = solver.solve()

    print("\n" + "=" * 80)
    print(" РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 80)
    print(f"Оптимальный вектор X* = [{opt_pt[0]:.4f}, {opt_pt[1]:.4f}]")
    print(f"Минимальное значение целевой функции f0(X*) = {opt_val:.4f}")
    print(f"Последовательность пройденных вершин:")
    for i, pt in enumerate(trajectory):
        print(f"  Вершина {i + 1}: x1 = {pt[0]:.4f}, x2 = {pt[1]:.4f}")
    print("=" * 80)

    # Геометрическая визуализация
    plot_linear_programming(trajectory, opt_pt, opt_val)


if __name__ == "__main__":
    main()