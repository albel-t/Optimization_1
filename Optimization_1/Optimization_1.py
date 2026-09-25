
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def f(x):
    return x**3 + 1.5*x**2 - 6*x - 12

def bisection(a, b, eps=0.01):
    delta = eps / 2
    history = []
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        x1, x2 = c - delta, c + delta
        f1, f2 = f(x1), f(x2)
        history.append((a, b, c, f(c)))
        if f1 < f2:
            b = c
        else:
            a = c
    root = (a + b) / 2
    history.append((a, b, root, f(root)))
    return root, history
def fibonacci_method(a, b, eps=0.01):
    fib = [1, 1]
    while (b - a) / fib[-1] > eps:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1
    history = []
    x1 = a + fib[n-2] / fib[n] * (b - a)
    x2 = a + fib[n-1] / fib[n] * (b - a)
    f1, f2 = f(x1), f(x2)
    history.append((a, b, x1, x2, f1, f2))
    for k in range(1, n - 1):
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            x1 = a + fib[n-k-2] / fib[n-k] * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + fib[n-k-1] / fib[n-k] * (b - a)
            f2 = f(x2)
        history.append((a, b, x1, x2, f1, f2))
    root = (a + b) / 2
    return root, history

def golden_section(a, b, eps=0.01):
    phi = (1 + 5**0.5) / 2
    history = []
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    f1, f2 = f(x1), f(x2)
    history.append((a, b, x1, x2, f1, f2))
    while (b - a) > eps:
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            x1 = b - (b - a) / phi
            f1 = f(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + (b - a) / phi
            f2 = f(x2)
        history.append((a, b, x1, x2, f1, f2))
    root = (a + b) / 2
    return root, history

class App:
    def __init__(self, root):
        self.root = root
        root.title("Методы одномерной оптимизации")
        root.geometry("1300x850")

        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Label(top, text="a =").pack(side=tk.LEFT)
        self.entry_a = tk.Entry(top, width=8)
        self.entry_a.insert(0, "-1.5")
        self.entry_a.pack(side=tk.LEFT, padx=(2, 12))

        tk.Label(top, text="b =").pack(side=tk.LEFT)
        self.entry_b = tk.Entry(top, width=8)
        self.entry_b.insert(0, "7")
        self.entry_b.pack(side=tk.LEFT, padx=(2, 12))

        tk.Label(top, text="eps =").pack(side=tk.LEFT)
        self.entry_eps = tk.Entry(top, width=8)
        self.entry_eps.insert(0, "0.01")
        self.entry_eps.pack(side=tk.LEFT, padx=(2, 12))


        self.info = tk.Label(top, text="", font=("Consolas", 10), justify=tk.LEFT)
        self.info.pack(side=tk.LEFT, padx=20)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(11, 8.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, root)
        toolbar.update()

    def plot(self):
        try:
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            eps = float(self.entry_eps.get())
        except ValueError:
            self.info.config(text="Ошибка ввода!")
            return

        root_b, hist_b = bisection(a, b, eps)
        root_f, hist_f = fibonacci_method(a, b, eps)
        root_g, hist_g = golden_section(a, b, eps)

        self.info.config(
            text=(f"Половинное деление: x* = {root_b:.6f}, итераций: {len(hist_b)}\n"
                  f"Фибоначчи:           x* = {root_f:.6f}, итераций: {len(hist_f)}\n"
                  f"Золотое сечение:     x* = {root_g:.6f}, итераций: {len(hist_g)}")
        )

        for ax in self.axes:
            ax.clear()

        xs = np.linspace(a - 1, b + 1, 800)
        ys = f(xs)

        ax = self.axes[0]
        ax.plot(xs, ys, 'k-', lw=1.5, label='f(x)')
        ax.axhline(0, color='gray', lw=0.8)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(hist_b)))
        for i, (ai, bi, ci, _) in enumerate(hist_b):
            ax.plot([ci], [f(ci)], 'o', color=colors[i], ms=6)
            ax.plot([ai, bi], [f(ai), f(bi)], '-', color=colors[i], lw=0.8, alpha=0.6)
        ax.axvline(root_b, color='red', ls='--', lw=1, label=f'x* = {root_b:.4f}')
        ax.set_title(f'Метод половинного деления (итераций: {len(hist_b)}, x* = {root_b:.6f})')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        ax = self.axes[1]
        ax.plot(xs, ys, 'k-', lw=1.5, label='f(x)')
        ax.axhline(0, color='gray', lw=0.8)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(hist_f)))
        for i, (ai, bi, x1, x2, f1, f2) in enumerate(hist_f):
            ax.plot([x1, x2], [f1, f2], 'o', color=colors[i], ms=6)
            ax.plot([ai, bi], [f(ai), f(bi)], '-', color=colors[i], lw=0.8, alpha=0.5)
        ax.axvline(root_f, color='red', ls='--', lw=1, label=f'x* = {root_f:.4f}')
        ax.set_title(f'Метод Фибоначчи (итераций: {len(hist_f)}, x* = {root_f:.6f})')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        ax = self.axes[2]
        ax.plot(xs, ys, 'k-', lw=1.5, label='f(x)')
        ax.axhline(0, color='gray', lw=0.8)
        colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(hist_g)))
        for i, (ai, bi, x1, x2, f1, f2) in enumerate(hist_g):
            ax.plot([x1, x2], [f1, f2], 'o', color=colors[i], ms=6)
            ax.plot([ai, bi], [f(ai), f(bi)], '-', color=colors[i], lw=0.8, alpha=0.5)
        ax.axvline(root_g, color='red', ls='--', lw=1, label=f'x* = {root_g:.4f}')
        ax.set_title(f'Метод золотого сечения (итераций: {len(hist_g)}, x* = {root_g:.6f})')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    app.plot()
    root.mainloop()