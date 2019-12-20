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

Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
Singular_SLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifforce=True)
Singular_DLP = lambda src, _: Stokes_Layer_Singular_Form(src, ifdipole=True) - 0.5*sign*np.eye(2*src.N)

# SLP Function with fixed pressure nullspace
def Fixed_SLP(src, trg):
    Nxx = trg.normal_x[:,None]*src.normal_x
    Nxy = trg.normal_x[:,None]*src.normal_y
    Nyx = trg.normal_y[:,None]*src.normal_x
    Nyy = trg.normal_y[:,None]*src.normal_y
    NN = np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
    MAT = Naive_SLP(src, trg) + NN
    return MAT

def solution_function(x, y, xc, yc, fx, fy):
    dx = x - xc
    dy = y - yc
    r = np.hypot(dx, dy)
    r2 = r**2
    fd = fx*dx + fy*dy
    u = -fx*np.log(r) + fd*dx/r2
    v = -fy*np.log(r) + fd*dy/r2
    p = 2*fd / r2
    return u, v, p
def solution_function_stress_jump(x, y, xc, yc, fx, fy, nx, ny):
    dx = x - xc
    dy = y - yc
    r = np.hypot(dx, dy)
    r2 = r*r
    r4 = r2*r2
    fd = fx*dx + fy*dy
    ux = fd/r2 - 2*fd*dx*dx/r4
    uy = (fy*dx - fx*dy)/r2 - 2*fd*dx*dy/r4
    vx = (fx*dy - fy*dx)/r2 - 2*fd*dx*dy/r4
    vy = fd/r2 - 2*fd*dy*dy/r4
    p = 2 * fd / r2
    Txx = 2*ux - p
    Txy = uy + vx
    Tyy = 2*vy - p
    Tx = Txx*nx + Txy*ny
    Ty = Txy*nx + Tyy*ny
    return Tx, Ty

# End Setup Test
################################################################################

############################################################################
# PREPARE QFS BOUNDARIES

bdy = GSB(c=star(n, f=5, a=0.3))
qfs_bdy = QFS_Boundary(bdy, eps=eps, FF=FF)

print('')
for interior in [True, False]:

    print('\nTesting QFS for Stokes', 'interior' if interior else 'exterior', 'evaluation.')

    sign = 1 if interior else -1
    fx = 1.0
    fy = 0.5
    if interior:
        far_targ             = PointSet(c=np.array(0.1+0.1j))
        xc, yc               = 1.0, 1.0
    else:
        far_targ             = PointSet(c=np.array(1.5+1.5j))
        xc, yc               = 1.0, 0.0

    _, fbdy, ebdy, cbdy = qfs_bdy.get_bdys(interior)

    ############################################################################
    # GET DENSITIES

    bcu, bcv, _ = solution_function(bdy.x, bdy.y, xc, yc, fx, fy)
    bctx, bcty = solution_function_stress_jump(bdy.x, bdy.y, xc, yc, fx, fy, bdy.normal_x, bdy.normal_y)
    slpx = bctx * sign
    slpy = bcty * sign
    dlpx = -bcu * sign
    dlpy = -bcv * sign
    slp = np.concatenate([slpx, slpy])
    dlp = np.concatenate([dlpx, dlpy])

    Evaluator = QFS_Evaluator(qfs_bdy, interior, b2c_funcs=[Singular_SLP, Singular_DLP], s2c_func=Fixed_SLP, on_surface=True, vector=True)

    spot = Evaluator([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    print('  Errors for on-surface eval')

    SLP = Naive_SLP(ebdy, bdy)
    ua, va, pa = solution_function(bdy.x, bdy.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    err = np.abs(SLP.dot(spot) - UA).max()/scale
    print('    Error, Singular QFS: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    print('  Errors for off-surface eval')

    shell = qfs_bdy.get_shell(shell_distance, interior, 10*bdy.N)
    SLP = Naive_SLP(ebdy, shell)
    ua, va, pa = solution_function(shell.x, shell.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    err = np.abs(SLP.dot(spot) - UA).max()/scale
    print('    Error, Singular QFS: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    print('  Errors for far eval')

    SLP = Naive_SLP(ebdy, far_targ)
    ua, va, pa = solution_function(far_targ.x, far_targ.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    err = np.abs(SLP.dot(spot) - UA).max()/scale
    print('    Error, Singular QFS: {:0.2e}'.format(err))

