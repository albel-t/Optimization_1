import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ========================================================
# 1. ЦЕЛЕВЫЕ ФУНКЦИИ И ГРАДИЕНТЫ
# ========================================================
a_param = -4.0
b_param = 5.0
c_param = 3.0
d_param = 5.0
alpha_deg = 115.0
alpha = np.deg2rad(alpha_deg)


def f1(x):
    x1, x2 = x[0], x[1]
    dx1 = x1 - a_param
    dx2 = x2 - b_param
    u = dx1 * np.cos(alpha) - dx2 * np.sin(alpha)
    v = dx1 * np.sin(alpha) + dx2 * np.cos(alpha)
    return (u / c_param) ** 2 + (v / d_param) ** 2


def grad_f1(x):
    x1, x2 = x[0], x[1]
    dx1 = x1 - a_param
    dx2 = x2 - b_param
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    u = dx1 * cos_a - dx2 * sin_a
    v = dx1 * sin_a + dx2 * cos_a
    du_dx1, du_dx2 = cos_a, -sin_a
    dv_dx1, dv_dx2 = sin_a, cos_a
    df_du = 2 * u / (c_param ** 2)
    df_dv = 2 * v / (d_param ** 2)
    return np.array([df_du * du_dx1 + df_dv * dv_dx1, df_du * du_dx2 + df_dv * dv_dx2])


def f2(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_f2(x):
    return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


# ========================================================
# 2. ПОИСК ШАГА (СВЕНН + ЗОЛОТОЕ СЕЧЕНИЕ)
# ========================================================
def line_search_optimal(f, x0, direction, tol=1e-5):
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.zeros_like(direction)
    d = direction / norm
    h_step = 0.05
    f0 = f(x0)
    while f(x0 + h_step * d) > f0 and h_step > 1e-7:
        h_step *= 0.5
    a, b = 0.0, h_step
    f_prev = f(x0 + b * d)
    step = h_step
    for _ in range(40):
        step *= 2.0
        c = b + step
        f_curr = f(x0 + c * d)
        if f_curr >= f_prev:
            a = b - step / 2.0
            b = c
            break
        f_prev = f_curr
        b = c
    a = max(0.0, a)
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1_val = f(x0 + x1 * d)
    f2_val = f(x0 + x2 * d)
    for _ in range(60):
        if abs(b - a) < tol:
            break
        if f1_val < f2_val:
            b = x2
            x2 = x1
            f2_val = f1_val
            x1 = a + resphi * (b - a)
            f1_val = f(x0 + x1 * d)
        else:
            a = x1
            x1 = x2
            f1_val = f2_val
            x2 = b - resphi * (b - a)
            f2_val = f(x0 + x2 * d)
    return ((a + b) / 2.0) * d


# ========================================================
# 3. МЕТОДЫ ОПТИМИЗАЦИИ
# ========================================================
def steepest_descent(f, grad_f, x_init, eps=1e-4, max_iter=1000):
    x = np.array(x_init, dtype=float)
    trajectory = [x.copy()]
    for _ in range(max_iter):
        grad = grad_f(x)
        if np.sum(grad ** 2) <= eps:
            break
        step = line_search_optimal(f, x, -grad)
        x = x + step
        trajectory.append(x.copy())
    return x, trajectory


def conjugate_gradients(f, grad_f, x_init, eps=1e-4, max_iter=1000):
    x = np.array(x_init, dtype=float)
    n = len(x)
    trajectory = [x.copy()]
    grad = grad_f(x)
    s = -grad.copy()
    for k in range(max_iter):
        if np.sum(grad ** 2) <= eps:
            break
        step = line_search_optimal(f, x, s)
        x_new = x + step
        trajectory.append(x_new.copy())
        grad_new = grad_f(x_new)
        if k % n == 0 or np.dot(s, grad_new) > 0:
            beta = 0.0
            s_next = -grad_new
        else:
            denom = np.sum(grad ** 2)
            beta = np.sum(grad_new ** 2) / denom if denom > 1e-12 else 0.0
            s_next = -grad_new + beta * s
        x = x_new
        grad = grad_new
        s = s_next
    return x, trajectory


# ========================================================
# 4. ИНТЕРФЕЙС (Tkinter)
# ========================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Лабораторная работа 2.3")
        root.geometry("1800x950")

        # Сетка 2x4
        self.fig = plt.Figure(figsize=(18, 9.5))
        self.axes = []
        for i in range(4):
            ax_3d = self.fig.add_subplot(2, 4, i + 1, projection='3d')
            ax_2d = self.fig.add_subplot(2, 4, i + 5)
            self.axes.append((ax_3d, ax_2d))

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Встроенная панель навигации (там есть зум)
        toolbar = NavigationToolbar2Tk(self.canvas, root)
        toolbar.update()
        self.canvas.toolbar = toolbar  # <-- ВАЖНО: привязываем тулбар к канвасу для работы зума

        self.plot()  # Сразу строим

    def plot(self):
        eps = 1e-4
        starts = [
            ("Параболоид — Точка 1", f1, grad_f1, [-1.0, 2.0], (-10, 8), (-2, 12)),
            ("Параболоид — Точка 2", f1, grad_f1, [2.0, -3.0], (-10, 8), (-2, 12)),
            ("Розенброк — Точка 1", f2, grad_f2, [-1.0, 1.0], (-2, 2), (-1, 3.5)),
            ("Розенброк — Точка 2", f2, grad_f2, [2.0, 2.0], (-2, 2), (-1, 3.5)),
        ]

        for i, (title, func, grad_func, x0, lim_x1, lim_x2) in enumerate(starts):
            ax_3d, ax_2d = self.axes[i]
            ax_3d.clear()
            ax_2d.clear()

            res_sd, traj_sd = steepest_descent(func, grad_func, x0, eps=eps)
            res_cg, traj_cg = conjugate_gradients(func, grad_func, x0, eps=eps)

            traj_sd = np.array(traj_sd)
            traj_cg = np.array(traj_cg)

            x1_range = np.linspace(lim_x1[0], lim_x1[1], 80)
            x2_range = np.linspace(lim_x2[0], lim_x2[1], 80)
            X1, X2 = np.meshgrid(x1_range, x2_range)

            Z = np.zeros_like(X1)
            for r in range(X1.shape[0]):
                for c in range(X1.shape[1]):
                    Z[r, c] = func([X1[r, c], X2[r, c]])

            max_z = np.percentile(Z, 75)
            Z_clip = np.clip(Z, 0, max_z)

            # --- 3D ---
            ax_3d.plot_surface(X1, X2, Z_clip, cmap='viridis', alpha=0.6, edgecolor='none')
            z_sd = np.clip([func(p) for p in traj_sd], 0, max_z)
            z_cg = np.clip([func(p) for p in traj_cg], 0, max_z)
            step_sd = max(1, len(traj_sd) // 50)
            step_cg = max(1, len(traj_cg) // 50)
            ax_3d.plot(traj_sd[::step_sd, 0], traj_sd[::step_sd, 1], z_sd[::step_sd], 'r.-', lw=1.5, markersize=4,
                       label='Спуск')
            ax_3d.plot(traj_cg[::step_cg, 0], traj_cg[::step_cg, 1], z_cg[::step_cg], 'b+-', lw=1.5, markersize=4,
                       label='Сопр. град.')
            ax_3d.set_title(f"3D: {title}", fontsize=10, fontweight='bold')
            ax_3d.set_xlabel('x1', fontsize=8)
            ax_3d.set_ylabel('x2', fontsize=8)
            ax_3d.tick_params(labelsize=7)
            ax_3d.legend(fontsize=7)
            ax_3d.view_init(elev=40, azim=-120)

            # --- 2D ---
            levels = np.linspace(0, max_z, 20)
            ax_2d.contour(X1, X2, Z, levels=levels, cmap='viridis')
            ax_2d.plot(traj_sd[::step_sd, 0], traj_sd[::step_sd, 1], 'r.-', lw=1.2, markersize=4, label='Спуск')
            ax_2d.plot(traj_cg[::step_cg, 0], traj_cg[::step_cg, 1], 'b+-', lw=1.2, markersize=4, label='Сопр. град.')

            if i < 2:
                ax_2d.scatter(a_param, b_param, color='gold', marker='*', s=100, label='Аналит. мин')
            else:
                ax_2d.scatter(1.0, 1.0, color='gold', marker='*', s=100, label='Аналит. мин')

            ax_2d.set_title(f"2D: {title}", fontsize=10, fontweight='bold')
            ax_2d.set_xlabel('x1', fontsize=8)
            ax_2d.set_ylabel('x2', fontsize=8)
            ax_2d.tick_params(labelsize=7)
            ax_2d.legend(fontsize=7)

            # Чтобы при зуме круги не превращались в овалы
            ax_2d.set_aspect('equal', adjustable='datalim')

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()