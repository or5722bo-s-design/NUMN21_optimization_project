import numpy as np

from optimization import *
from Rosenbrock_problem import rosenbrock, rosenbrock_gradient

### Run Optimizer on Rosenbrock ###

rose_problem = Problem(rosenbrock, rosenbrock_gradient)

# method = GoodBroyden(problem=rose_pr)
# method = SymmetricBroyden(problem=rose_pr)
# method = BadBroyden(problem=rose_pr) # Not working!
# method = DFP(problem=rose_pr)
# method = BFGS(problem=rose_pr)

opts = [
    {
        "name": "Classical Newton (exact ls)",
        "method": ClassicalNewton(problem=rose_problem, use_exact_ls=True),
    },
    {
        "name": "Classical Newton (armijo-wolfe)",
        "method": ClassicalNewton(problem=rose_problem, use_exact_ls=False),
    },
    {
        "name": "Inexact Line Search (armijo-wolfe)",
        "method": InexactLineSearch(problem=rose_problem),
    },
    {
        "name": "Good Broyden",
        "method": GoodBroyden(problem=rose_problem),
    },
    {
        "name": "Bad Broyden (fails)",
        "method": BadBroyden(problem=rose_problem),
    },
    {
        "name": "Symmetric Broyden",
        "method": SymmetricBroyden(problem=rose_problem),
    },
    {
        "name": "DFP",
        "method": DFP(problem=rose_problem),
    },
    {
        "name": "BFGS",
        "method": BFGS(problem=rose_problem),
    },
]

def analyze_solver(name, method, x_points, root):
    """
    Runs the solver on all given starting points, collects and prints information
    about the run.
    
    :param name: Name of the method to print in the analysis.
    :param method: Method which implements the Optimization class.
    :param x_points: Sample starting points, (m x 2) matrix for Rosenbrock.
    :param root: Expected minimum to end at.
    """
    
    x_points = np.array(x_points)
    n = x_points.shape[0]
    
    # Accumulated totals
    step_count = 0
    ascent_dir_count = 0
    n2_dist = 0

    for x in x_points:
        sol = method.solve(x)
        step_count += (len(method.step_history) - 1)
        ascent_dir_count += method.ascent_dir_count
        n2_dist += np.linalg.norm(sol - root)
    
    # Printed analysis
    print(f"--- {name} ---")
    print(f"\tAVG Root Error: {round(n2_dist / n, 5)}")
    print(f"\tAVG Steps Taken: {step_count / n}")
    print(f"\tRatio of ascent steps: {round((ascent_dir_count / step_count) * 100, 3)} %")

# Generate "n" random points, each method will be compared on these same starting values.
n = 40
x_points = np.random.random((n, 2)) * 5 - 3
root = [1., 1.]
# Loop through all methods
for opt in opts:
    name = opt["name"]
    method = opt["method"]
    analyze_solver(name, method, x_points, root)
    

### Run BFGS on Chebyquad ###


