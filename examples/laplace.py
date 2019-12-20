import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import pybie2d
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet

from qfs.two_d_qfs import QFS_Boundary, QFS_Evaluator

eps = 1e-12           # estimated tolerance for integrals
FF = 0.0              # fudge factor to increase accuracy by (0.36 --> 1 digit), to make up for addition of errors
rough_function = True # use a rough function or not
n = 400               # number of points in boundary discretization
shell_distance = 1e-5 # shell distance to test close eval at

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
    xc = 1.0
    yc = 1.0
    return x + 2*y + np.log((x-xc)**2 + (y-yc)**2)
def interior_solution_function_dn(x, y, nx, ny):
    xc = 1.0
    yc = 1.0
    return nx + 2*ny + 1/((x-xc)**2 + (y-yc)**2)*(2*(x-xc)*nx + 2*(y-yc)*ny)
def exterior_solution_function(x, y):
    xc = 1.0
    yc = 0.0
    return np.log((x-xc)**2 + (y-yc)**2)
def exterior_solution_function_dn(x, y, nx, ny):
    xc = 1.0
    yc = 0.0
    return 1/((x-xc)**2 + (y-yc)**2)*(2*(x-xc)*nx + 2*(y-yc)*ny)

# End Setup Test
################################################################################

############################################################################
# PREPARE QFS BOUNDARIES

bdy = GSB(c=star(n, f=5, a=0.3))
qfs_bdy = QFS_Boundary(bdy, eps=eps, FF=FF)

print('')
for interior in [True, False]:
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

    _, fbdy, ebdy, cbdy = qfs_bdy.get_bdys(interior)

    ############################################################################
    # GET DENSITIES

    bc = solution_function(bdy.x, bdy.y)
    bcn = solution_function_dn(bdy.x, bdy.y, bdy.normal_x, bdy.normal_y)
    slp = bcn * sign
    dlp = -bc * sign

    Evaluator = QFS_Evaluator(qfs_bdy, interior, b2c_funcs=[Singular_SLP, Singular_DLP], s2c_func=Naive_SLP, on_surface=True)
    spot = Evaluator([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    print('  Errors for on-surface eval')

    SLP = Naive_SLP(ebdy, bdy)
    ua = solution_function(bdy.x, bdy.y)
    err = np.abs(SLP.dot(spot) - ua).max()/scale
    print('    Error, Singular QFS: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    print('  Errors for off-surface eval')

    shell = qfs_bdy.get_shell(shell_distance, interior, 10*bdy.N)
    SLP = Naive_SLP(ebdy, shell)
    ua = solution_function(shell.x, shell.y)
    err = np.abs(SLP.dot(spot) - ua).max()/scale
    print('    Error, Singular QFS: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    print('  Errors for far eval')

    SLP = Naive_SLP(ebdy, far_targ)
    ua = solution_function(far_targ.x, far_targ.y)
    err = np.abs(SLP.dot(spot) - ua).max()/scale
    print('    Error, Singular QFS: {:0.2e}'.format(err))

