import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from hessian_approx import finite_diffs
from Rosenbrock_problem import plot_rosenbrock_path


class CountedProblem:
    def __init__(self, problem, cache_size=2000):
        self.raw_objective = problem.objective
        self.raw_gradient = problem.gradient
        self._cache_size = cache_size
        self.reset_counts()

    #for no leakage at the start of every run
    def reset_counts(self):
        self.n_f = 0
        self.n_g = 0
        self._f_cache = {}
        self._g_cache = {}

    def objective(self, x):
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self._f_cache:
            if len(self._f_cache) > self._cache_size:
                self._f_cache.clear()
            self.n_f += 1
            self._f_cache[key] = self.raw_objective(x)
        return self._f_cache[key]

    def gradient(self, x):
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self._g_cache:
            if len(self._g_cache) > self._cache_size:
                self._g_cache.clear()
            self.n_g += 1
            self._g_cache[key] = np.asarray(self.raw_gradient(x), dtype=float)
        return self._g_cache[key].copy()

    gradient_at = gradient

class Tracked:
    track_hessian_error = False #set to True on an instance if you want error_history

    def _log(self, x, g):
        P = self.problem
        self.step_history.append(x)
        self.f_history.append(P.raw_objective(x))# uncounted
        self.grad_norm_history.append(np.linalg.norm(g))
        self.f_evals_history.append(P.n_f)
        self.g_evals_history.append(P.n_g)

    def solve(self, x0):
        self.reset()
        self.f_history, self.grad_norm_history = [], []
        self.f_evals_history, self.g_evals_history = [], []
        P = self.problem
        P.reset_counts()

        x = np.array(x0, dtype=float)
        for _ in range(self.max_steps):
            g = P.gradient_at(x)
            self._log(x, g)          # final point is logged too

            if np.linalg.norm(g) < self.grad_tol:
                break
            if len(self.step_history) > 1 and \
               np.linalg.norm(x - self.step_history[-2]) < self.step_tol:
                break

            G, is_inv = self.approx_hessian(x)

            if self.track_hessian_error:
                G_true = finite_diffs(P.raw_gradient, x, dx=1e-8)   # uncounted
                if is_inv:
                    try:
                        G_true = np.linalg.inv(G_true)
                    except np.linalg.LinAlgError:
                        G_true = np.full_like(G_true, np.nan)
                self.error_history.append(np.linalg.norm(G - G_true, "fro"))

            s = -G @ g if is_inv else linalg.solve(G, -g)
            alpha = self.line_search(x, s)
            if alpha < 0:
                self.ascent_dir_count += 1
            x = x + alpha * s
        return x

def track(method_cls, problem, **kwargs):
    tracked_cls = type("Tracked" + method_cls.__name__, (Tracked, method_cls), {})
    tracked_cls.__name__ = method_cls.__name__ # keeps plot titles tidy
    return tracked_cls(problem=CountedProblem(problem), **kwargs)

def plot_convergence_panel(methods, x0):
    """Left: ||g|| vs iterations. Right: ||g|| vs gradient evaluations."""
    fig, (ax_it, ax_ev) = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in methods.items():
        m.solve(x0)
        its = range(len(m.grad_norm_history))
        ax_it.semilogy(its, m.grad_norm_history, "o-", ms=3, label=name)
        ax_ev.semilogy(m.g_evals_history, m.grad_norm_history, "o-", ms=3, label=name)
    ax_it.set_xlabel("iteration k")
    ax_ev.set_xlabel("gradient evaluations")
    for ax in (ax_it, ax_ev):
        ax.set_ylabel(r"$\|\nabla f(x_k)\|$")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.show()

