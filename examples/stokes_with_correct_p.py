import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import pybie2d
import numexpr as ne
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet

from qfs.two_d_qfs import QFS_Boundary, QFS_Evaluator, QFS_Evaluator_Pressure

eps = 1e-14           # estimated tolerance for integrals
FF = 0.0              # fudge factor to increase accuracy by (0.36 --> 1 digit), to make up for addition of errors
rough_function = True # use a rough function or not
n = 300               # number of points in boundary discretization
shell_distance = 1e-5 # shell distance to test close eval at

################################################################################
# Setup Test

Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
Naive_DLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifdipole=True)

from pyfmmlib2d import SFMM
def Stokes_Layer_Apply(src, trg, f):
    s = src.get_stacked_boundary()
    t = trg.get_stacked_boundary()
    out = SFMM(source=s, target=t, forces=f*src.weights, compute_target_velocity=True, compute_target_stress=True)
    u = out['target']['u']
    v = out['target']['v']
    p = out['target']['p']
    return u, v, p

def PSLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N])
    out[:-1,:] = Naive_SLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    sir2 = 0.5/r2/np.pi
    out[-1, 0*src.N:1*src.N] = dx*sir2*src.weights
    out[-1, 1*src.N:2*src.N] = dy*sir2*src.weights
    return out

def PDLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N])
    out[:-1,:] = Naive_DLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    rdotn = dx*src.normal_x + dy*src.normal_y
    ir2 = 1.0/r2
    rdotnir4 = rdotn*ir2*ir2
    out[-1, 0*src.N:1*src.N] = (-src.normal_x*ir2 + 2*rdotnir4*dx)*src.weights
    out[-1, 1*src.N:2*src.N] = (-src.normal_y*ir2 + 2*rdotnir4*dy)*src.weights
    out[-1] /= np.pi
    return out

def resample(f, n):
    import warnings
    import scipy as sp
    import scipy.signal
    if n == len(f):
        out = f
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = sp.signal.resample(f, n)
    return out

def Pressure_SLP(src, trg):
    out = np.zeros([2*trg.N+1, 2*src.N+1])
    out[:-1,:-1] = Naive_SLP(src, trg)
    dx = trg.x[0] - src.x
    dy = trg.y[0] - src.y
    r2 = dx*dx + dy*dy
    sir2 = 0.5/r2/np.pi
    out[-1, 0*src.N:1*src.N] = dx*sir2*src.weights
    out[-1, 1*src.N:2*src.N] = dy*sir2*src.weights
    out[0*trg.N:1*trg.N, -1] = resample(src.normal_x*src.weights, trg.N)
    out[1*trg.N:2*trg.N, -1] = resample(trg.normal_y*trg.weights, trg.N)
    print(resample(src.normal_x*src.weights, trg.N)*src.N/trg.N - trg.normal_x*trg.weights)
    return out

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

    Evaluator = QFS_Evaluator_Pressure(qfs_bdy, interior, b2c_funcs=[PSLP, PDLP], s2c_func=Pressure_SLP, form_b2c=True)

    spot = Evaluator([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    print('  Errors for on-surface eval')

    ua, va, pa = solution_function(bdy.x, bdy.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    ue = Stokes_Layer_Apply(ebdy, bdy, spot.reshape(2, ebdy.N))
    UE = np.concatenate([ue[0], ue[1]])
    err = np.abs(UE - UA).max()/scale
    perr = np.abs(ue[2] - pa).max()/scale
    print('    Error, Singular QFS, velocity: {:0.2e}'.format(err))
    print('    Error, Singular QFS, pressure: {:0.2e}'.format(perr))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    print('  Errors for off-surface eval')

    shell = qfs_bdy.get_shell(shell_distance, interior, 10*bdy.N)
    ua, va, pa = solution_function(shell.x, shell.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    ue = Stokes_Layer_Apply(ebdy, shell, spot.reshape(2, ebdy.N))
    UE = np.concatenate([ue[0], ue[1]])
    err = np.abs(UE - UA).max()/scale
    perr = np.abs(ue[2] - pa).max()/scale
    print('    Error, Singular QFS, velocity: {:0.2e}'.format(err))
    print('    Error, Singular QFS, pressure: {:0.2e}'.format(perr))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    print('  Errors for far eval')

    ua, va, pa = solution_function(far_targ.x, far_targ.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    ue = Stokes_Layer_Apply(ebdy, far_targ, spot.reshape(2, ebdy.N))
    UE = np.concatenate([ue[0], ue[1]])
    err = np.abs(UE - UA).max()/scale
    perr = np.abs(ue[2] - pa).max()/scale
    print('    Error, Singular QFS, velocity: {:0.2e}'.format(err))
    print('    Error, Singular QFS, pressure: {:0.2e}'.format(perr))
