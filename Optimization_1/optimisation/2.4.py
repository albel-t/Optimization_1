import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import warnings

warnings.filterwarnings('ignore')

# ========================================================
# 1. ЦЕЛЕВАЯ ФУНКЦИЯ И ОГРАНИЧЕНИЯ (ВАРИАНТ 2)
# ========================================================
# Таблица 2.3: a=4, b=4, c=2, d=3, alpha=55
a_param, b_param, c_param, d_param, alpha_deg = 4.0, 4.0, 2.0, 3.0, 55.0
alpha = np.radians(alpha_deg)


def f0(X):
    x1, x2 = X[0], X[1]
    t1 = ((x1 - a_param) * np.cos(alpha) + (x2 - b_param) * np.sin(alpha)) / c_param
    t2 = ((x2 - b_param) * np.cos(alpha) - (x1 - a_param) * np.sin(alpha)) / d_param
    return t1 ** 2 + t2 ** 2


# Ограничения (fi(X) >= 0)
# 1) x1 - 4/x2 <= 0  ===>  f1(X) = 4/x2 - x1 >= 0
def f1(X):
    x2 = X[1]
    if x2 <= 1e-6: return -1e10  # Защита
    return 4.0 / x2 - X[0]


def grad_f1(X):
    x2 = X[1]
    if x2 <= 1e-6: return np.array([0.0, 0.0])
    return np.array([-1.0, -4.0 / (x2 ** 2)])


# 2) x1 >= 0  ===>  f2(X) = x1 >= 0
def f2(X):
    return X[0]


def grad_f2(X):
    return np.array([1.0, 0.0])


CONSTRAINTS = [f1, f2]
GRAD_CONSTRAINTS = [grad_f1, grad_f2]


def is_feasible(X, tol=1e-6):
    return all(fi(X) >= -tol for fi in CONSTRAINTS)


def is_strictly_feasible(X, tol=1e-6):
    return all(fi(X) > tol for fi in CONSTRAINTS)


# ========================================================
# 2. ПЕРЕМЕЩЕНИЕ В ДОПУСТИМУЮ ОБЛАСТЬ
# ========================================================
def move_to_feasible_region(X0, h=0.05, max_iter=1000):
    X = np.array(X0, dtype=float)
    trajectory = [X.copy()]
    if is_strictly_feasible(X, tol=1e-4):
        return X, trajectory

    for _ in range(max_iter):
        grad_phi = np.zeros_like(X)
        for fi, grad_fi in zip(CONSTRAINTS, GRAD_CONSTRAINTS):
            if fi(X) < 0:
                grad_phi += grad_fi(X)

        norm_g = np.linalg.norm(grad_phi)
        if norm_g < 1e-12:
            break

        X = X + h * (grad_phi / norm_g)
        trajectory.append(X.copy())

        if is_strictly_feasible(X, tol=1e-3):
            return X, trajectory
    return X, trajectory


# ========================================================
# 3. ФУНКЦИИ ШТРАФА
# ========================================================
def internal_barrier_R(X, K):
    for fi in CONSTRAINTS:
        if fi(X) <= 1e-12:
            return 1e20
    s2 = sum(1.0 / fi(X) for fi in CONSTRAINTS)
    return f0(X) + (1.0 / K) * s2


def external_penalty_R(X, K):
    s3 = sum((min(0.0, fi(X))) ** 2 for fi in CONSTRAINTS)
    return f0(X) + K * s3


# ========================================================
# 4. МЕТОД НЕЛДЕРА-МИДА
# ========================================================
def nelder_mead_solver(f, x_init, k_step=0.2, eps=1e-5, max_iter=2000):
    x_init = np.array(x_init, dtype=float)
    n = len(x_init)
    step = k_step
    simplex = [x_init.copy()]
    for j in range(n):
        pt = x_init.copy()
        pt[j] += step
        simplex.append(pt)

    while any(f(pt) >= 1e19 for pt in simplex[1:]) and step > 1e-6:
        step *= 0.5
        simplex = [x_init.copy()]
        for j in range(n):
            pt = x_init.copy()
            pt[j] += step
            simplex.append(pt)
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
# 5. АЛГОРИТМЫ ШТРАФНЫХ ФУНКЦИЙ
# ========================================================
def solve_internal_barrier(x0, K1=0.1, C=5.0, eps=1e-4, max_outer=25):
    history_shifts = []
    if not is_strictly_feasible(x0, tol=1e-4):
        x_curr, history_shifts = move_to_feasible_region(x0)
    else:
        x_curr = np.array(x0, dtype=float)

    # Начинаем с маленького K, чтобы барьер не был слишком жестким
    K = K1
    history = [(0, K, x_curr.copy(), f0(x_curr), 0.0)]
    X_star_prev = x_curr.copy()

    for i in range(1, max_outer + 1):
        # Делаем шаг симплекса очень маленьким и зависимым от K
        current_k_step = 0.05 / K

        # Запускаем Нелдера-Мида
        X_star_i = nelder_mead_solver(lambda X: internal_barrier_R(X, K), x_curr, k_step=current_k_step)

        dist = np.linalg.norm(X_star_i - X_star_prev)
        history.append((i, K, X_star_i.copy(), f0(X_star_i), dist))

        # Если сошлось, выходим
        if dist <= eps:
            break

        K *= C
        x_curr = X_star_i.copy()
        X_star_prev = X_star_i.copy()

    return X_star_i, history, history_shifts

def solve_external_penalty(x0, K1=1.0, C=10.0, eps=1e-3, max_outer=10):
    x_curr = np.array(x0, dtype=float)
    K = K1
    history = [(0, K, x_curr.copy(), f0(x_curr), 0.0)]
    X_star_prev = x_curr.copy()

    for i in range(1, max_outer + 1):
        X_star_i = nelder_mead_solver(lambda X: external_penalty_R(X, K), x_curr, k_step=0.3)
        dist = np.linalg.norm(X_star_i - X_star_prev)
        history.append((i, K, X_star_i.copy(), f0(X_star_i), dist))
        if dist <= eps:
            break
        K *= C
        x_curr = X_star_i.copy()
        X_star_prev = X_star_i.copy()

    return X_star_i, history, []


# ========================================================
# 6. ИНТЕРФЕЙС (2 ОКНА КАК У ОДНОГРУППНИКА)
# ========================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Лабораторная работа 2.4: Вариант 2")
        root.geometry("1600x900")

        # Делаем 1 строка, 2 колонки (как у друга на верхнем скрине)
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
        # Начальные точки для варианта 2 (x1 >= 0 и x2 > 0)
        # Внутри: x1=1, x2=3 (1 <= 4/3)
        X0_inside = np.array([1.0, 3.0])
        # Снаружи: x1=3, x2=2 (3 > 4/2=2)
        X0_outside = np.array([3.0, 2.0])

        # Решаем
        self.res_int_in, self.hist_int_in, _ = solve_internal_barrier(X0_inside, eps=eps)
        self.res_int_out, self.hist_int_out, self.shifts_int = solve_internal_barrier(X0_outside, eps=eps)
        self.res_ext_in, self.hist_ext_in, _ = solve_external_penalty(X0_inside, eps=eps)
        self.res_ext_out, self.hist_ext_out, _ = solve_external_penalty(X0_outside, eps=eps)

        # Сетка для графиков (x2 > 0, так как 4/x2)
        x1_vals = np.linspace(-1.0, 8.0, 300)
        x2_vals = np.linspace(0.2, 8.0, 300)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        Z = np.zeros_like(X1)
        for r in range(X1.shape[0]):
            for c in range(X1.shape[1]):
                Z[r, c] = f0([X1[r, c], X2[r, c]])

        # Допустимая область: x1 <= 4/x2 и x1 >= 0
        feas_mask = (X1 <= 4.0 / X2) & (X1 >= 0)

        titles = [
            "Метод внутренней барьерной функции (2.10)",
            "Метод внешней штрафной функции (2.11)"
        ]

        plot_data = [
            {
                "runs": [
                    {"name": "X0 внутри", "history": self.hist_int_in, "shifts": [], "color": "blue"},
                    {"name": "X0 вне", "history": self.hist_int_out, "shifts": self.shifts_int, "color": "orange"}
                ]
            },
            {
                "runs": [
                    {"name": "X0 внутри", "history": self.hist_ext_in, "shifts": [], "color": "blue"},
                    {"name": "X0 вне", "history": self.hist_ext_out, "shifts": [], "color": "orange"}
                ]
            }
        ]

        for ax, title, data in zip(self.axes, titles, plot_data):
            ax.clear()

            # Линии уровня целевой функции
            cs = ax.contour(X1, X2, Z, levels=30, cmap='viridis', alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8)

            # Закраска допустимой области
            ax.contourf(X1, X2, feas_mask, levels=[0.5, 1.0], colors=['#a8e6cf'], alpha=0.4)

            # Границы (гипербола 4/x2 и ось x1=0)
            x2_line = np.linspace(0.2, 8.0, 100)
            ax.plot(4.0 / x2_line, x2_line, 'crimson', lw=2, label=r'\(4/x_2 - x_1 = 0\)')
            ax.axvline(0, color='navy', lw=2, label=r'\(x_1 = 0\)')

            for run in data["runs"]:
                pts = np.array([item[2] for item in run["history"]])
                if len(run["shifts"]) > 0:
                    shifts_arr = np.array(run["shifts"])
                    ax.plot(shifts_arr[:, 0], shifts_arr[:, 1], 'm:', lw=1.5, marker='o', markersize=3,
                            label=f'Сдвиг X0 ({run["name"]})')

                ax.plot(pts[:, 0], pts[:, 1], color=run["color"], linestyle='-', marker='.', lw=2, markersize=8,
                        label=f'Траектория ({run["name"]})')
                ax.scatter(pts[0, 0], pts[0, 1], color=run["color"], marker='o', s=80, edgecolors='black', zorder=5)
                ax.scatter(pts[-1, 0], pts[-1, 1], color='lime', marker='*', s=180, edgecolors='black', zorder=6)

            # Безусловный минимум
            ax.scatter(a_param, b_param, color='red', marker='X', s=130, edgecolors='black', zorder=7,
                       label='Безусл. min')

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(r'\(x_1\)')
            ax.set_ylabel(r'\(x_2\)')
            ax.set_xlim([-1.0, 8.0])
            ax.set_ylim([0.2, 8.0])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def print_tables(self):
        def print_hist(name, hist):
            print(f"\n--- {name} ---")
            print(f"{'i':<3} | {'K':<8} | {'X*':<20} | {'f0(X*)':<10} | {'||X*i - X*prev||':<15}")
            print("-" * 65)
            for step, K, X, fval, dist in hist:
                print(f"{step:<3} | {K:<8.1e} | {str(np.round(X, 4)):<20} | {fval:<10.4f} | {dist:<15.5e}")

        print_hist("Внутренняя барьерная (Точка 1)", self.hist_int_in)
        print_hist("Внутренняя барьерная (Точка 2)", self.hist_int_out)
        print_hist("Внешняя квадратичная (Точка 1)", self.hist_ext_in)
        print_hist("Внешняя квадратичная (Точка 2)", self.hist_ext_out)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()