import numpy as np

def finite_diffs(g, x, dx=1e-4):
    """
    Approximate the hessian at a given point by measuring the change of derivatives
    with central finite differences.

    :param g: Gradient function
    :param x: Point of interest
    :returns ndarray: Approximation of Hessian at point x of g.
    """

    g_x = g(x)
    n = g_x.shape[0]

    G = np.zeros((n, n))
    for i in range(0, n):
        e_i = np.zeros(n)
        e_i[i] = 1.0

        # Central difference formula
        g_forw = g(x + dx * e_i).flatten()
        g_back = g(x - dx * e_i).flatten()
        
        G[:,i] = (g_forw - g_back) / (2 * dx)

    # Symmetrize
    return (G + G.transpose()) / 2

def good_broyden(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    Hy = H@y
    denom = np.dot(s, Hy)
    if abs(denom) > 1e-12:
        # Rank 1 update
        H = H + np.outer(s-Hy,s@H)/denom
    return H

def bad_broyden(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    denom = np.dot(y, y)
    if abs(denom) > 1e-12:
        # Rank 1 update
        H = H + np.outer(s-H@y,y) / denom
    return H

def sym_broyden(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    Hy = H@y
    denom = np.dot(s - Hy, y)

    if abs(denom) > 1e-12:
        a = 1 / denom
        u = s - Hy
        
        H = H + a * np.outer(u, u)
    return H

def DFP(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    denom1 = np.dot(y,H@y)
    denom2 = np.dot(y,s)

    if abs(denom1) > 1e-12:
        H = H - np.outer(H@y,y@H)/denom1
    if abs(denom2) > 1e-12:
        H = H + np.outer(s,s)/denom2
    return H

def BFGS(H, s, y):
    """
    :param H: Previous Hess inv approx
    :param s: Delta between previous location and current location.
    :param y: Delta between the current grad and the previous grad.
    """
    sty = np.dot(s,y)

    if abs(sty) > 1e-12:
        Hy = H@y
        yH = y@H
        
        up1 = (1 + np.dot(y,Hy)/sty)*(np.outer(s,s)/sty)
        up2 = (np.outer(s,yH) + np.outer(Hy,s)) / sty

        H = H + up1 - up2
    return H
