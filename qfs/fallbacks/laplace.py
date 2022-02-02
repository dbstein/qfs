import numpy as np

def Laplace_Kernel_Form(source, target, ifcharge=False, chweight=None,
                            ifdipole=False, dpweight=None, dipvec=None,
                            weights=None):
    """
    Laplace Kernel Formation
    Computes the matrix:
        [ chweight*G_ij + dpweight*(n_j dot grad G_ij) ] weights_j
        where G is the Laplace Greens function and other parameters described
        below
    Also returns the matrices for the x and y derivatives, if requested

    Parameters:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
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
    scale = -1.0/(2*np.pi)
    scale = scale*np.ones(ns) if weights is None else scale*weights
    chscale = scale if chweight is None else scale*chweight
    dpscale = scale if dpweight is None else scale*dpweight
    G = np.zeros([nt, ns], dtype=float)
    if not (ifcharge or ifdipole):
        # no charges, no dipoles
        # just return appropriate zero matrices
        return G
    else:
        dx = TX - SX
        dy = TY - SY
        id2 = 1.0/(dx*dx + dy*dy)
        if ifcharge:
            # charges effect on potential
            G -= 0.5*np.log(id2)*chscale
        if ifdipole:
            # dipoles effect on potential
            G -= (nx*dx + ny*dy)*id2*dpscale
        if source is target:
            np.fill_diagonal(G, 0.0)
        return G

def Laplace_Layer_Form(source, target=None, ifcharge=False, chweight=None,
                                ifdipole=False, dpweight=None):
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
    return Laplace_Kernel_Form(
            source   = source.get_stacked_boundary(T=True),
            target   = target.get_stacked_boundary(T=True),
            ifcharge = ifcharge,
            chweight = chweight,
            ifdipole = ifdipole,
            dpweight = dpweight,
            dipvec   = dipvec,
            weights  = source.weights
        )
