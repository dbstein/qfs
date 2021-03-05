import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import pybie2d
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet

from qfs.stokes_qfs import Stokes_QFS

# estimated tolerance for integrals
tol = 1e-14
# number of points in boundary discretization
n = 300
# shell distance to test close eval at
shell_distance = 0.1
# see documentation for QFS
shift_type = 1
# use singular eval or not
singular = True
# type of inverse to use ('LU', 'SVD', 'Circulant')
s2c_type = 'Circulant'
# Form, Apply, or Circulant
b2c_type = 'Apply'

################################################################################
# Setup Test

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

############################################################################
# Generate options dictionary

options = {
    'tol'        : tol,
    'shift_type' : shift_type,
    'b2c_type'   : b2c_type,
    's2c_type'   : s2c_type,
    'singular'   : singular,
}

############################################################################
# Full Stokes Apply Function

from pyfmmlib2d import SFMM
def Full_Stokes_Layer_Apply(src, trg, f):
    s = src.get_stacked_boundary()
    t = trg.get_stacked_boundary()
    out = SFMM(source=s, target=t, forces=f*src.weights, compute_target_velocity=True, compute_target_stress=True)
    u = out['target']['u']
    v = out['target']['v']
    p = out['target']['p']
    return u, v, p

################################################################################
# Setup boundary

use_circle = s2c_type == 'Circulant' or b2c_type == 'Circulant'
bdy = GSB(c=star(n, f=5, a=0.0 if use_circle else 0.2))

################################################################################
# Run test

print('')
print('Running QFS Tests for Stokes Equation')
print('    options are:')
for key, value in options.items():
    print('        ', (key+':').ljust(15), value)

for interior in [True, False]:

    print('\n    Testing', 'interior' if interior else 'exterior', 'evaluation.')

    ############################################################################
    # get densities

    sign = 1 if interior else -1
    fx = 1.0
    fy = 0.5
    if interior:
        far_targ             = PointSet(c=np.array(0.1+0.1j))
        xc, yc               = 0.9, 0.9
    else:
        far_targ             = PointSet(c=np.array(1.5+1.5j))
        xc, yc               = 0.2, 0.2

    bcu, bcv, _ = solution_function(bdy.x, bdy.y, xc, yc, fx, fy)
    bctx, bcty = solution_function_stress_jump(bdy.x, bdy.y, xc, yc, fx, fy, bdy.normal_x, bdy.normal_y)
    slpx = bctx * sign
    slpy = bcty * sign
    dlpx = -bcu * sign
    dlpy = -bcv * sign
    slp = np.concatenate([slpx, slpy])
    dlp = np.concatenate([dlpx, dlpy])

    ############################################################################
    # Setup qfs operators

    qfs = Stokes_QFS(bdy, interior, True, True, options)

    ############################################################################
    # Solve for effective density and extract effective sources
    
    spot = qfs([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    source = qfs.source

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    ua, va, pa = solution_function(bdy.x, bdy.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    out = Full_Stokes_Layer_Apply(source, bdy, spot.reshape(2, source.N))
    UE = np.concatenate([out[0], out[1]])
    err = np.abs(UE - UA).max()/scale
    perr = np.abs(out[2] - pa).max()/scale
    print('        On surface')
    print('            velocity: {:0.2e}'.format(err))
    print('            pressure: {:0.2e}'.format(perr))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    shell = GSB(c=qfs.get_normal_shift(shell_distance))
    ua, va, pa = solution_function(shell.x, shell.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    out = Full_Stokes_Layer_Apply(source, shell, spot.reshape(2, source.N))
    UE = np.concatenate([out[0], out[1]])
    err = np.abs(UE - UA).max()/scale
    perr = np.abs(out[2] - pa).max()/scale
    print('        Off surface')
    print('            Error, velocity: {:0.2e}'.format(err))
    print('            Error, pressure: {:0.2e}'.format(perr))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    ua, va, pa = solution_function(far_targ.x, far_targ.y, xc, yc, fx, fy)
    UA = np.concatenate([ua, va])
    out = Full_Stokes_Layer_Apply(source, far_targ, spot.reshape(2, source.N))
    UE = np.concatenate([out[0], out[1]])
    err = np.abs(UE - UA).max()/scale
    perr = np.abs(out[2] - pa).max()/scale
    print('        In the far field')
    print('            Error, velocity: {:0.2e}'.format(err))
    print('            Error, pressure: {:0.2e}'.format(perr))
