import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import pybie2d
import scipy as sp
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet

from qfs._two_d_qfs import QFS

eps = 1e-14                # estimated tolerance for integrals
n = 3000                    # number of points in boundary discretization
shell_distance = 0.1       # shell distance to test close eval at
shift_type = 1             # see documentation for QFS
singular = True            # use singular eval or not
inverse_type = 'Circulant' # LU, SVD, Circulant, or FullCirculant

################################################################################
# Setup Test

Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)
Naive_DLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifdipole=True)
Singular_SLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifcharge=True)
Singular_DLP = lambda src, _: Laplace_Layer_Singular_Form(src, ifdipole=True) - 0.5*sign*np.eye(src.N)

def interior_solution_function(x, y):
    xc = 1.1
    yc = 1.1
    return x + 2*y + np.log((x-xc)**2 + (y-yc)**2)
def interior_solution_function_dn(x, y, nx, ny):
    xc = 1.1
    yc = 1.1
    return nx + 2*ny + 1/((x-xc)**2 + (y-yc)**2)*(2*(x-xc)*nx + 2*(y-yc)*ny)
def exterior_solution_function(x, y):
    xc = 0.5
    yc = 0.0
    return np.log((x-xc)**2 + (y-yc)**2)
def exterior_solution_function_dn(x, y, nx, ny):
    xc = 0.5
    yc = 0.0
    return 1/((x-xc)**2 + (y-yc)**2)*(2*(x-xc)*nx + 2*(y-yc)*ny)

# End Setup Test
################################################################################

def get_Kress_V(source):
    N = source.N
    dt = source.dt
    v1 = 4.0*np.sin(np.pi*np.arange(N)/N)**2
    v1[0] = 1.0
    V1 = 0.5*np.log(v1)/dt
    v2 = np.abs(np.fft.fftfreq(N, 1.0/N))
    v2[0] = np.Inf
    v2[int(N/2)] = np.Inf # experimental!
    V2 = 0.5*np.fft.ifft(1.0 / v2).real/dt
    return V1/N + V2
def Laplace_SLP_Circulant_Form(source):
    V = get_Kress_V(source)
    ssource = source.Generate_1pt_Circulant_Boundary()
    A = Laplace_Layer_Form(ssource, target=source, ifcharge=True)
    A[0] = -np.log(source.speed[0])/(2*np.pi)*source.weights[0]
    return A[:,0] + V*source.weights
def Laplace_DLP_Circulant_Form(source, interior):
    sign = 1 if interior else -1
    ssource = source.Generate_1pt_Circulant_Boundary()
    w = Naive_DLP(ssource, source)
    w[0] = -0.25*source.curvature[0]*source.weights[0]/np.pi - 0.5*sign
    return w

############################################################################
# PREPARE QFS BOUNDARIES

# bdy = GSB(c=star(n, f=5, a=0.0 if inverse_type in ['Circulant', 'FullCirculant'] else 0.2))
bdy = GSB(c=star(n, f=5, a=0.0))

print('')
for interior in [True, False]:

    # test SLP
    L1 = Singular_SLP(bdy, bdy)
    l2 = Laplace_SLP_Circulant_Form(bdy)
    L2 = sp.linalg.circulant(l2)
    print('Circulant SLP good? {:0.2e}'.format(np.abs(L1-L2).max()))

    # test LP
    L1 = Singular_DLP(bdy, bdy)
    l2 = Laplace_DLP_Circulant_Form(bdy, interior)
    L2 = sp.linalg.circulant(l2)
    print('Circulant DLP good? {:0.2e}'.format(np.abs(L1-L2).max()))

    print('\nTesting QFS for Laplace', 'interior' if interior else 'exterior', 'evaluation.')

    sign = 1 if interior else -1
    if interior:
        solution_function    = interior_solution_function
        solution_function_dn = interior_solution_function_dn
        far_targ             = PointSet(c=np.array(0.1+0.1j))
    else:
        solution_function    = exterior_solution_function
        solution_function_dn = exterior_solution_function_dn
        far_targ             = PointSet(c=np.array(1.5+1.5j))

    ############################################################################
    # GET DENSITIES

    bc = solution_function(bdy.x, bdy.y)
    bcn = solution_function_dn(bdy.x, bdy.y, bdy.normal_x, bdy.normal_y)
    slp = bcn * sign
    dlp = -bc * sign

    if singular:
        b2c_funcs = [Singular_SLP, Singular_DLP]
    else:
        b2c_funcs = [Naive_SLP, Naive_DLP]

    Evaluator = QFS(bdy, interior, b2c_funcs, singular, Naive_SLP, tol=eps, shift_type=shift_type, inverse_type=inverse_type)

    spot = Evaluator([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    print('  Errors for on-surface eval')

    SLP = Naive_SLP(Evaluator.source, bdy)
    ua = solution_function(bdy.x, bdy.y)
    err = np.abs(SLP.dot(spot) - ua).max()/scale
    print('    Error: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    print('  Errors for off-surface eval')

    shell = GSB(c=Evaluator.get_normal_shift(shell_distance))
    SLP = Naive_SLP(Evaluator.source, shell)
    ua = solution_function(shell.x, shell.y)
    err = np.abs(SLP.dot(spot) - ua).max()/scale
    print('    Error: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    print('  Errors for far eval')

    SLP = Naive_SLP(Evaluator.source, far_targ)
    ua = solution_function(far_targ.x, far_targ.y)
    err = np.abs(SLP.dot(spot) - ua).max()/scale
    print('    Error: {:0.2e}'.format(err))

