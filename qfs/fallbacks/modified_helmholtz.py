import numpy as np
from scipy.special import k0, k1

# off from real GF by factor of 1/(2*np.pi), doesn't matter
def Modified_Helmholtz_Greens_Function(r, k):
    return k0(k*r)
def Modified_Helmholtz_Greens_Function_Derivative(r, k):
    return k*k1(k*r)/r

def Modified_Helmholtz_Kernel_Form(source, target, k=1.0, ifcharge=False,
                                    ifdipole=False, dipvec=None, weights=None):
    """
    Modified Helmholtz Kernel Formation
        for the problem (Delta - k^2)u = 0
    Computes the matrix:
        [ chweight*G_ij + dpweight*(n_j dot grad G_ij) ] weights_j
        where G is the Modified Helmholtz Greens function
            (G(z) = k^2*k0(k*z)/(2*pi))
        and other parameters described below
    Also returns the matrices for the x and y derivatives, if requested

    Parameters:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        k,        required, float,         modified helmholtz parameter
        ifcharge, optional, bool,          include charge contribution
        chweight, optional, float,         scalar weight to apply to charges
        ifdipole, optional, bool,          include dipole contribution
        dpweight, optional, float,         scalar weight to apply to dipoles
        dipvec,   optional, float(2, ns),  dipole orientations
        weights,  optional, float(ns),     quadrature weights

    This function assumes that source and target have no coincident points
    """
    ns = source.shape[1]
    nt = target.shape[1]
    SX = source[0]
    SY = source[1]
    TX = target[0][:,None]
    TY = target[1][:,None]
    if dipvec is not None:
        nx = dipvec[0]
        ny = dipvec[1]
    scale = 1.0/(2*np.pi)
    scale = scale*np.ones(ns) if weights is None else scale*weights
    G = np.zeros([nt, ns], dtype=float)
    if not (ifcharge or ifdipole):
        # no charges, no dipoles, just return appropriate zero matrix
        return G
    else:
        dx = TX - SX
        dy = TY - SY
        r = np.hypot(dx, dy)
        if ifcharge:
            G += Modified_Helmholtz_Greens_Function(r, k)
        if ifdipole:
            G += (nx*dx + ny*dy)*Modified_Helmholtz_Greens_Function_Derivative(r, k)
        if source is target:
            np.fill_diagonal(G, 0.0)
        return G*scale

def Modified_Helmholtz_Layer_Form(source, target=None, k=1.0, ifcharge=False,
                                                                ifdipole=False):
    """
    Laplace Layer Evaluation (potential and gradient in 2D)
    Assumes that source is not target (see function Laplace_Layer_Self_Form)

    Parameters:
        source,   required, Boundary, source
        target,   optional, Boundary, target
        ifcharge, optional, bool,  include effect of charge (SLP)
        chweight, optional, float, scalar weight for the SLP portion
        ifdipole, optional, bool,  include effect of dipole (DLP)
        dpweight, optional, float, scalar weight for the DLP portion

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    dipvec = None if ifdipole is None else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    return Modified_Helmholtz_Kernel_Form(
            source   = source.get_stacked_boundary(T=True),
            target   = target.get_stacked_boundary(T=True),
            k        = k,
            ifcharge = ifcharge,
            ifdipole = ifdipole,
            dipvec   = dipvec,
            weights  = source.weights,
        )
