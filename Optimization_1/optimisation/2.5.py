import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings

warnings.filterwarnings('ignore')


# ========================================================
# 1. ОБЁРТКА ДЛЯ ПОДСЧЕТА ВЫЗОВОВ ФУНКЦИИ
# ========================================================
class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.func(np.array(x, dtype=float))

    def reset(self):
        self.count = 0


# ========================================================
# 2. ЦЕЛЕВАЯ ФУНКЦИЯ И ОГРАНИЧЕНИЯ (ВАРИАНТ 2)
# ========================================================
# Таблица 2.4: a=2, b=-1, c=2, d=3, alpha=125
a_param, b_param, c_param, d_param, alpha_deg = 2.0, -1.0, 2.0, 3.0, 125.0
alpha = np.radians(alpha_deg)


def f0(X):
    x1, x2 = X[0], X[1]
    t1 = ((x1 - a_param) * np.cos(alpha) + (x2 - b_param) * np.sin(alpha)) / c_param
    t2 = ((x2 - b_param) * np.cos(alpha) - (x1 - a_param) * np.sin(alpha)) / d_param
    return t1 ** 2 + t2 ** 2


# Ограничение-неравенство: x1^2 + 3x2^2 - 2x1 - 5 <= 0
# Приводим к виду f1(X) >= 0:
# f1(X) = 5 + 2x1 - x1^2 - 3x2^2 >= 0
def f1_ineq(X):
    return 5.0 + 2.0 * X[0] - X[0] ** 2 - 3.0 * X[1] ** 2


def grad_f1_ineq(X):
    df_dx1 = 2.0 - 2.0 * X[0]
    df_dx2 = -6.0 * X[1]
    return np.array([df_dx1, df_dx2])


# Ограничение-равенство: x1^2 - x2 = 0
def f2_eq(X):
    return X[0] ** 2 - X[1]


def is_strictly_feasible(X, tol=1e-4):
    return f1_ineq(X) > tol


# ========================================================
# 3. АЛГОРИТМ ПЕРЕМЕЩЕНИЯ НАЧАЛЬНОЙ ТОЧКИ
# ========================================================
def move_to_feasible_region(X0, h=0.05, max_iter=1000):
    X = np.array(X0, dtype=float)
    trajectory = [X.copy()]
    if is_strictly_feasible(X):
        return X, trajectory

    for _ in range(max_iter):
        grad = grad_f1_ineq(X)
        norm_g = np.linalg.norm(grad)
        if norm_g < 1e-12:
            break
        X = X + h * (grad / norm_g)
        trajectory.append(X.copy())
        if is_strictly_feasible(X, tol=1e-3):
            return X, trajectory
    return X, trajectory


# ========================================================
# 4. КОМБИНИРОВАННАЯ ФУНКЦИЯ ШТРАФА (ФОРМУЛА 2.15)
# ========================================================
def combined_penalty_R(X, K):
    val_ineq = f1_ineq(X)
    if val_ineq <= 1e-12:
        return 1e20

    barrier = (1.0 / K) * np.log(val_ineq)
    penalty_eq = K * (f2_eq(X) ** 2)
    return f0(X) - barrier + penalty_eq


# ========================================================
# 5. БЕЗУСЛОВНАЯ ОПТИМИЗАЦИЯ (НЕЛДЕР — МИД)
# ========================================================
def nelder_mead_solver(f, x_init, k_step=0.2, eps=1e-5, max_iter=1500):
    x_init = np.array(x_init, dtype=float)
    n = len(x_init)
    step = k_step
    simplex = [x_init.copy()]
    for j in range(n):
        pt = x_init.copy()
        pt[j] += step
        simplex.append(pt)

    attempts = 0
    while any(f(pt) >= 1e19 for pt in simplex[1:]) and attempts < 25:
        step *= 0.5
        simplex = [x_init.copy()]
        for j in range(n):
            pt = x_init.copy()
            pt[j] += step
            simplex.append(pt)
        attempts += 1
    simplex = np.array(simplex)

    for _ in range(max_iter):
        f_vals = np.array([f(pt) for pt in simplex])
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]

        f_l, f_g, f_h = f_vals[0], f_vals[-2], f_vals[-1]
        x_h = simplex[-1]
        f_mean = np.mean(f_vals)
        sigma = np.sqrt(np.sum((f_vals - f_mean) ** 2) / (n + 1))
        if sigma < eps:
            return simplex[0]

        x_c = np.mean(simplex[:-1], axis=0)
        x_0 = 2.0 * x_c - x_h
        f_0 = f(x_0)

        if f_0 < f_l:
            x_r = x_c + 2.0 * (x_0 - x_c)
            f_r = f(x_r)
            simplex[-1] = x_r if f_r < f_l else x_0
        elif f_l <= f_0 <= f_g:
            simplex[-1] = x_0
        else:
            if f_0 < f_h:
                x_s = 0.5 * x_0 + 0.5 * x_c
            else:
                x_s = 0.5 * x_h + 0.5 * x_c
            f_s = f(x_s)
            if f_s < f_h:
                simplex[-1] = x_s
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    return simplex[0]


# ========================================================
# 6. РЕШЕНИЕ КОМБИНИРОВАННЫМ МЕТОДОМ
# ========================================================
def solve_combined_penalty(x0, K1=1.0, C=4.0, eps=1e-3, max_outer_iter=15):
    history_shifts = []
    if not is_strictly_feasible(x0):
        x_curr, shift_traj = move_to_feasible_region(x0)
        history_shifts = shift_traj
    else:
        x_curr = np.array(x0, dtype=float)

    K = K1
    history = [(0, K, x_curr.copy(), f0(x_curr), f1_ineq(x_curr), f2_eq(x_curr), 0.0)]
    X_star_prev = x_curr.copy()

    for i in range(1, max_outer_iter + 1):
        counter = FunctionCounter(lambda X: combined_penalty_R(X, K))
        k_step = 0.25 / np.sqrt(K)
        X_star_i = nelder_mead_solver(counter, x_curr, k_step=k_step, eps=1e-5)

        dist = np.linalg.norm(X_star_i - X_star_prev)
        history.append((i, K, X_star_i.copy(), f0(X_star_i), f1_ineq(X_star_i), f2_eq(X_star_i), dist))

        if dist <= eps and abs(f2_eq(X_star_i)) <= eps:
            break
        K *= C
        x_curr = X_star_i.copy()
        X_star_prev = X_star_i.copy()

    return X_star_i, history, history_shifts


# ========================================================
# 7. ИНТЕРФЕЙС Tkinter
# ========================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Лабораторная работа 2.5: Вариант 2")
        root.geometry("1600x900")

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, root)
        toolbar.update()
        self.canvas.toolbar = toolbar

        self.plot()
        self.print_tables()

    def plot(self):
        eps = 1e-3
        # Вариант 2: безусловный минимум (2, -1).
        # Проверим неравенство: 5 + 2*2 - 4 - 3*1 = 2 > 0 (внутри)
        # Равенство: 4 - (-1) = 5 != 0.

        # Точка 1: Внутри (по неравенству)
        X0_inside = np.array([2.0, -1.0])
        # Точка 2: Вне области
        X0_outside = np.array([-3.0, 2.0])

        # Решаем
        self.res_in, self.hist_in, _ = solve_combined_penalty(X0_inside, eps=eps)
        self.res_out, self.hist_out, self.shifts_out = solve_combined_penalty(X0_outside, eps=eps)

        x1_vals = np.linspace(-4.0, 4.0, 300)
        x2_vals = np.linspace(-4.0, 4.0, 300)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        Z = np.zeros_like(X1)
        for r in range(X1.shape[0]):
            for c in range(X1.shape[1]):
                Z[r, c] = f0([X1[r, c], X2[r, c]])

        # Неравенство: 5 + 2x1 - x1^2 - 3x2^2 >= 0
        ineq_vals = 5.0 + 2.0 * X1 - X1 ** 2 - 3.0 * X2 ** 2
        feas_mask = ineq_vals >= 0

        titles = [
            "Эксперимент 1: Старт из точки внутри f1 > 0",
            "Эксперимент 2: Старт вне допустимой области"
        ]
        runs = [
            (self.hist_in, [], "darkorange"),
            (self.hist_out, self.shifts_out, "blue")
        ]

        for ax, title, (hist, shifts, color) in zip(self.axes, titles, runs):
            ax.clear()

            # Линии уровня
            cs = ax.contour(X1, X2, Z, levels=30, cmap='viridis', alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8)

            # Допустимая область (зеленая)
            ax.contourf(X1, X2, feas_mask, levels=[0.5, 1.0], colors=['#a8e6cf'], alpha=0.4)

            # Граница неравенства (красная)
            ax.contour(X1, X2, ineq_vals, levels=[0], colors=['crimson'], linewidths=2)

            # Линия равенства x1^2 - x2 = 0 (синяя парабола)
            x1_line = np.linspace(-3.0, 3.0, 100)
            x2_line = x1_line ** 2
            ax.plot(x1_line, x2_line, 'navy', lw=2, label=r'\(x_1^2 - x_2 = 0\)')

            # Сдвиг в область
            if len(shifts) > 0:
                shifts_arr = np.array(shifts)
                ax.plot(shifts_arr[:, 0], shifts_arr[:, 1], 'm:', lw=1.5, marker='o', markersize=3, label='Сдвиг X0')

            # Траектория
            pts = np.array([item[2] for item in hist])
            ax.plot(pts[:, 0], pts[:, 1], color=color, linestyle='-', marker='.', lw=2, markersize=8,
                    label='Траектория')
            ax.scatter(pts[0, 0], pts[0, 1], color=color, marker='o', s=80, edgecolors='black', zorder=5)
            ax.scatter(pts[-1, 0], pts[-1, 1], color='lime', marker='*', s=180, edgecolors='black', zorder=6,
                       label='Финиш')

            # Безусловный минимум
            ax.scatter(a_param, b_param, color='red', marker='X', s=130, edgecolors='black', zorder=7,
                       label='Безусл. min')

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(r'\(x_1\)')
            ax.set_ylabel(r'\(x_2\)')
            ax.set_xlim([-4.0, 4.0])
            ax.set_ylim([-4.0, 4.0])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def print_tables(self):
        def print_hist(name, hist):
            print(f"\n--- {name} ---")
            print(
                f"{'k':<3} | {'K':<8} | {'X*':<20} | {'f0(X*)':<10} | {'f1(X*)':<10} | {'f2(X*)':<10} | {'||X*k - X*(k-1)||':<15}")
            print("-" * 90)
            for step, K, X, fval, f1v, f2v, dist in hist:
                print(
                    f"{step:<3} | {K:<8.1e} | {str(np.round(X, 4)):<20} | {fval:<10.4f} | {f1v:<10.4f} | {f2v:<10.4e} | {dist:<15.5e}")

        print_hist("Эксперимент 1 (Внутри)", self.hist_in)
        print_hist("Эксперимент 2 (Вне)", self.hist_out)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()