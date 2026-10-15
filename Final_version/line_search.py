import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

### Graph Functions ###

def plot_phi(phi, phi_prime, a, b, exact_sol, inexact_sol):
    """
    Plot the changing function in the step direction on interval [a, b].
    """

    x = np.linspace(a, b, 200)
    y = [phi(x_i) for x_i in x]

    # Plot phi(α)
    plt.plot(x, y, label="phi(α)", color="blue", linewidth=2)

    # Cubic interpolation formula
    f0 = phi(a)
    f1 = phi(b)
    fp0 = (b-a)*phi_prime(a)
    fp1 = (b-a)*phi_prime(b)

    eta = 3*(f1-f0)-2*fp0-fp1
    xi = fp0+fp1-2*(f1-f0)

    # Plot cubic interpolation
    z = (x - a) / (b - a)
    y = [f0 + fp0*z_i + eta*(z_i**2) + xi*(z_i**3) for z_i in z]
    plt.plot(x, y, label="cubic", color="red", linewidth=2, linestyle="--", zorder=10)

    # Plot exact line search solution and our found solution
    plt.plot(
        exact_sol, phi(exact_sol), label="Exact Search Sol", 
        marker="*", markersize=12, zorder=20,
        color="gold", markeredgecolor="black", markeredgewidth=1.5,
    )
    plt.plot(
        inexact_sol, phi(inexact_sol), label="Inexact Search Sol",
        marker="D", markersize=10, zorder=19,
        color="green", markeredgecolor="black", markeredgewidth=1.5,
    )

    plt.legend()
    plt.show()

# TODO One for graphing phi(α), should have components:
#  - phi(α)
#  - Armijo condition
#  - Wolfe condition?
#  - Cubic interpolation over interval [0, mu]

# TODO One for graphing the cubic interpolation

### HELPER FUNCTIONS ###

def _line_search_fns(problem, x, s):
    """
    Get the line search functions phi and phi_prime where:
    | phi(a) = x + as | with "x" and "s" given.

    Returns phi, phi_prime
    """
    def phi(a):
        return problem.objective(x + s * a)
    def phi_prime(a):
        return np.dot(problem.gradient(x + s * a), s)

    return phi, phi_prime

### LINE SEARCH FUNCTIONS ###

def exact_line_search(problem, x, s):
    """
    Exact line search using scipy's minimize scalar function on the function
    | phi(a) = x + as | with "x" and "s" given.
    """

    phi, phi_prime = _line_search_fns(problem, x, s)

    rho = 0.01
    mu = -phi(0) / (rho * phi_prime(0))
    res = so.minimize_scalar(phi, bracket=[0, mu])
    return res.x

def arm_wolf_search(problem, x, s, alpha0 = 1.0):
    """
    Line search based on Armijo-Wolfe conditions.
    
    :param obj_func: Objective function
    :param obj_grad: Objective function gradient
    :param x: Current location
    :param s: Current step direction

    :returns float: The step size determined by the Armijo-Wolfe conditions.
    """
    flipped_s = False
    phi, phi_prime = _line_search_fns(problem, x, s)

    if phi_prime(0) > 0:
        # "s" is not a descent direction, so we will search in the opposite direction.
        flipped_s = True
        phi, phi_prime = _line_search_fns(problem, x, -s)

    # Constants
    max_steps = 100
    lower_bound = 0
    sigma, rho = 0.1, 0.01
    tau1, tau2, tau3 = 9, 0.1, 0.5

    # Pick new value in interval by minimizing a cubic polynomial
    def pick_value_cubic(a, b):
        f0 = phi(a)
        f1 = phi(b)
        fp0 = (b-a)*phi_prime(a)
        fp1 = (b-a)*phi_prime(b)

        eta = 3*(f1-f0)-2*fp0-fp1
        xi = fp0+fp1-2*(f1-f0)

        # Solve for minima of the cubic

        # Boundary points
        cub0 = f0
        cub1 = f0 + fp0 + eta + xi
        
        # Stationary points (quadratic roots)
        st_pts = []
        if abs(xi) > 1e-5:
            root_sq = 4*(eta**2) - 12*xi*fp0
            if root_sq >= 0:
                lroot = (-2*eta - np.sqrt(root_sq)) / (6*xi)
                rroot = (-2*eta + np.sqrt(root_sq)) / (6*xi)

                st_pts.append(lroot)
                st_pts.append(rroot)
        elif abs(eta) > 1e-5:
            root = -fp0 / (2*eta)
            st_pts.append(root)

        min_z = 0 if cub0 < cub1 else 1
        for stpt in st_pts:
            if stpt < 0 or stpt > 1:
                continue
            
            if phi(a + stpt*(b - a)) < phi(a + min_z*(b - a)):
                min_z = stpt

        # Convert back to alpha in range a, b
        new_alpha = a + min_z*(b - a)
        if (new_alpha > a*1.02 and new_alpha > b*1.02) or (new_alpha < a*0.98 and new_alpha < b*0.98):
            raise ValueError("Cubic interval yielded invalid value outside of range.")
        return new_alpha

    # Solve bracket
    def solve_bracket(a, b):
        for _ in range(1, max_steps):
            alpha = pick_value_cubic(a + tau2*(b-a), b - tau3*(b-a))
            if phi(alpha) > phi(0) + rho * alpha * phi_prime(0) or phi(alpha) >= phi(a):
                b = alpha
            else:
                if abs(phi_prime(alpha)) <= -sigma * phi_prime(0):
                    return alpha
                a_prev = a
                a = alpha
                if (b - a_prev) * phi_prime(alpha) >= 0:
                    b = a_prev
        return alpha

    # Bracketing interval
    mu = (lower_bound - phi(0)) / (rho * phi_prime(0))
    a, b = 0, mu

    # Pick an initial guess
    alpha_prev = 0
    alpha = alpha0
    solution = alpha0
    for _ in range(1, max_steps):
        if phi(alpha) < lower_bound:
            solution = alpha
            break
        if phi(alpha) > phi(0) + alpha * rho * phi_prime(0) or phi(alpha) >= phi(alpha_prev):
            solution = solve_bracket(alpha_prev, alpha)
            break
        if abs(phi_prime(alpha)) <= -sigma * phi_prime(0):
            solution = alpha
            break
        if phi_prime(alpha) >= 0:
            solution = solve_bracket(alpha, alpha_prev)
            break
        if mu <= 2 * alpha - alpha_prev:
            tmp = alpha
            alpha = mu
            alpha_prev = tmp
        else:
            tmp = alpha
            alpha = pick_value_cubic(2*alpha - alpha_prev, min(mu, alpha + tau1*(alpha-alpha_prev)))
            alpha_prev = tmp

    # Uncomment for plotting of found solution and cubic interpolation
    # if not flipped_s:
    #     exact_sol = exact_line_search(problem, x, s)
    #     inexact_sol = solution
    #     end = max(exact_sol, inexact_sol) * 2
    #     plot_phi(phi, phi_prime, 0, 1.25 * end, exact_sol, inexact_sol)
    
    return solution if not flipped_s else -solution
