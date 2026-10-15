import numpy as np
from collections import deque
from optimization import Optimization, QNewton
from line_search import arm_wolf_search

'''Source: https://medium.com/gaussian-machine/implement-l-bfgs-optimization-from-scratch-3e5c909f9079'''

def two_loop(q, S, Y):
    """Return H_k@q from the stored (s, y) pairs (oldest first)"""
    q = np.array(q, dtype=float)
    stack = [] # values from loop 1 to save for loop 2
    # loop 1
    for s, y in zip(reversed(S), reversed(Y)):
        rho = 1.0 / np.dot(y, s)
        a = rho * np.dot(s, q)
        q -= a * y
        stack.append((a, rho, s, y))

    # middle step
    gamma = np.dot(S[-1], Y[-1]) / np.dot(Y[-1], Y[-1]) if S else 1.0
    r = gamma * q

    # loop 2
    for a, rho, s, y in reversed(stack):
        b = rho * np.dot(y, r)
        r += s * (a - b)
    return r # r is H_k@q

class LBFGS(QNewton):
    """Limited-memory BFGS: keeps the last m (s, y) pairs."""
    def __init__(self, *args, m=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m
        self.S = deque(maxlen=m)
        self.Y = deque(maxlen=m)

    def reset(self):
        super().reset()
        self.S = deque(maxlen=self.m)
        self.Y = deque(maxlen=self.m)

    def approx_hessian(self, x):
        g = self.problem.gradient(x)

        # only after the first values are obtained
        if self.xp is not None and self.gp is not None:
            s = x-self.xp
            y = g-self.gp
            #only keep pairs satisfying the curvature condition, otherwise use the old memory
            if np.dot(s, y)>1e-10*np.linalg.norm(s)*np.linalg.norm(y):
                self.S.append(s)
                self.Y.append(y)

        self.xp, self.gp = x.copy(), g.copy()

        #dense H_k, built by applying the two-loop recursion to the identity columns
        #fine for small n, the rest of the code expects a matrix
        S, Y = list(self.S), list(self.Y)
        n = x.shape[0]
        H = np.column_stack([two_loop(e, S, Y) for e in np.eye(n)])
        #not really LBFGS, but solve needs a matrix 
        return (H + H.T) / 2, True


class NonlinearCG(Optimization):
    def __init__(self, *args, variant="PR+", restart_every=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.restart_every = restart_every
        self._reset_state()

    def reset(self):
        super().reset()
        self._reset_state()

    def _reset_state(self):
        self.gp = None #previous gradient
        self.sp = None #previous search direction
        self.k = 0 #iteration counter, used for restarts
        self.alpha_prev = None
        self.gs_prev = None

    def _beta(self, g, gp, sp):
        y = g - gp
        if self.variant == "FR": #Fletcher-Reaves
            return np.dot(g, g) / np.dot(gp, gp)
        if self.variant == "PR+": #Polak-Ribiere
            return max(0.0, np.dot(g, y) / np.dot(gp, gp))
        denom = np.dot(sp, y)
        if abs(denom) < 1e-12:
            return 0.0
        if self.variant == "HS": #Hestenes-Stiefel
            return np.dot(g, y) / denom
        if self.variant == "DY": #Dai-Yuan
            return np.dot(g, g) / denom
        raise ValueError(f"Unknown variant {self.variant}")

    #not a hessian, but that is the function solve expects, this computes s the direction
    def approx_hessian(self, x):
        g = self.problem.gradient(x)
        n = x.shape[0]
        restart = self.restart_every or n

        if self.gp is None or self.k % restart == 0:
            s = -g #steepest descent start
        else:
            s = -g + self._beta(g, self.gp, self.sp) * self.sp
            if np.dot(g, s) >= 0: #not a descent direction
                s = -g

        self.gp, self.sp = g.copy(), s.copy()
        self.k += 1

        #solve() computes the step as s=-G@g, so build a G that gives exactly our s.
        # G = I - outer(s + g, g) / (g.g)  =>  -G @ g = s
        G = np.eye(n) - np.outer(s + g, g) / np.dot(g, g)
        return G, True

    def line_search(self, x, s):
        g = self.problem.gradient(x)
        gs = np.dot(g, s)

        # CG steps are not scaled by a Hessian approximation, so guess the initial step size
        if self.alpha_prev is None:
            alpha0 = min(1.0, 1.0 / np.linalg.norm(s))
        else:
            alpha0 = np.clip(self.alpha_prev * self.gs_prev / gs, 1e-10, 1e2)

        alpha = arm_wolf_search(self.problem, x, s, alpha0=alpha0)
        self.alpha_prev, self.gs_prev = abs(alpha), gs
        return alpha