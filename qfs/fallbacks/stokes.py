import numpy as np

################################################################################
# General Purpose Low Level Source --> Target Kernel Formation

def Stokes_Kernel_Form(source, target, ifforce=False, fweight=None,
                    ifdipole=False, dpweight=None, dipvec=None, weights=None):
    """
    Stokes Kernel Formation

    Parameters:
        source,   required, float(2, ns),  source coordinates
        target,   required, float(2, nt),  target coordinates
        ifforce,  optional, bool,          include force contribution
        fweight,  optional, float,         scalar weight to apply to forces
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
    fscale = 0.25/np.pi
    dscale = 1.0/np.pi
    if weights is not None:
        fscale *= weights
        dscale *= weights
    if fweight is not None:
        fscale *= fweight
    if dpweight is not None:
        dscale *= dpweight
    G = np.zeros([2*nt, 2*ns], dtype=float)
    if not (ifforce or ifdipole):
        # no forces, no dipoles
        # just return appropriate zero matrix
        return G
    else:
        dx = TX - SX
        dy = TY - SY
        id2 = 1.0/(dx*dx + dy*dy)
        if ifforce:
            # forces effect on velocity
            logid = 0.5*np.log(id2)
            G[:nt, :ns] += fscale*(logid + dx*dx*id2)
            GH = fscale*dx*dy*id2
            G[nt:, :ns] += GH
            G[:nt, ns:] += GH            
            G[nt:, ns:] += fscale*(logid + dy*dy*id2)
        if ifdipole:
            # dipoles effect on velocity
            d_dot_n_ir4 = (dx*nx + dy*ny)*id2*id2
            G[:nt, :ns] += dscale*d_dot_n_ir4*dx*dx
            GH = dscale*d_dot_n_ir4*dx*dy
            G[nt:, :ns] += GH
            G[:nt, ns:] += GH
            G[nt:, ns:] += dscale*d_dot_n_ir4*dy*dy
        if source is target:
            np.fill_diagonal(G, 0.0)
            np.fill_diagonal(G[:ns, ns:], 0.0)
            np.fill_diagonal(G[ns:, :ns], 0.0)
        return G

def Stokes_Layer_Form(source, target=None, ifforce=False, fweight=None,
                                                ifdipole=False, dpweight=None):
    """
    Stokes Layer Evaluation (potential and gradient in 2D)

    Parameters:
        source,   required, Boundary, source
        target,   optional, Boundary, target
        ifforce,  optional, bool,  include effect of force (SLP)
        fweight,  optional, float, scalar weight for the SLP portion
        ifdipole, optional, bool,  include effect of dipole (DLP)
        dpweight, optional, float, scalar weight for the DLP portion

    If source is not target, then this function assumes that source and
        target have no coincident points
    If source is target, this function computes a naive quadrature,
        ignoring the i=j term in the sum
    """
    dipvec = None if not ifdipole else source.get_stacked_normal(T=True)
    if target is None:
        target = source
    return Stokes_Kernel_Form(
            source   = source.get_stacked_boundary(T=True),
            target   = target.get_stacked_boundary(T=True),
            ifforce  = ifforce,
            fweight  = fweight,
            ifdipole = ifdipole,
            dpweight = dpweight,
            dipvec   = dipvec,
            weights  = source.weights,
        )

