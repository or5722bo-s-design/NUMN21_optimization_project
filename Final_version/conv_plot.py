from optimization import *
from extra_methods import LBFGS, NonlinearCG
from history import track, plot_convergence_panel
from chebyquad_problem import chebyquad, gradchebyquad

problem = Problem(chebyquad, gradchebyquad)

methods = {
    "BFGS":   track(BFGS, problem),
    "DFP":    track(DFP, problem),
    "L-BFGS": track(LBFGS, problem, m=5),
    "CG (PR+)": track(NonlinearCG, problem, variant="PR+"),
    "CG (FR)":  track(NonlinearCG, problem, variant="FR"),
    "Newton": track(ClassicalNewton, problem, use_exact_ls=False),
}

x0 = np.linspace(0, 1, 8)
plot_convergence_panel(methods, x0)

