import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import pybie2d
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet

from qfs.helmholtz_qfs import Helmholtz_QFS
from scipy.special import hankel1
from pyfmmlib2d import HFMM

# estimated tolerance for integrals
tol = 1e-14
# number of points in boundary discretization
n = 300
# shell distance to test close eval at
shell_distance = 0.1
# see documentation for QFS
shift_type = 1
# use singular eval or not
singular = False
# type of inverse to use ('LU', 'SVD', 'Circulant')
s2c_type = 'Circulant'
# Form, Apply, or Circulant
b2c_type = 'Circulant'
# Helmholtz parameter
k = 10.0

################################################################################
# Test problems

def HGF(r, k):
    return 0.25j*hankel1(0, k*r)
def HGFP(r, k):
    return k*0.25j*hankel1(1, k*r)/r
def Apply_CF(src, trg, tau, k=1.0):
    source = src.get_stacked_boundary()
    target = trg.get_stacked_boundary()
    normals = src.get_stacked_normal()
    wtau = tau*src.weights
    out = HFMM(source, target, charge=-1j*k*wtau, dipstr=wtau, dipvec=normals, helmholtz_parameter=k, compute_target_potential=True)
    return out['target']['u']

def interior_solution_function(x, y):
    xc = 1.1
    yc = 1.1
    return HGF(np.hypot(x-xc, y-yc), k)
def interior_solution_function_dn(x, y, nx, ny):
    xc = 1.1
    yc = 1.1
    d = HGFP(np.hypot(x-xc, y-yc), k)
    return -((x-xc)*nx*d + (y-yc)*ny*d)
def exterior_solution_function(x, y):
    xc = 0.5
    yc = 0.0
    return HGF(np.hypot(x-xc, y-yc), k)
def exterior_solution_function_dn(x, y, nx, ny):
    xc = 0.5
    yc = 0.0
    d = HGFP(np.hypot(x-xc, y-yc), k)
    return -((x-xc)*nx*d + (y-yc)*ny*d)

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
# Setup boundary

use_circle = s2c_type == 'Circulant' or b2c_type == 'Circulant'
bdy = GSB(c=star(n, f=5, a=0.0 if use_circle else 0.2))

############################################################################
# Run test

print('')
print('Running QFS Tests for Helmholtz Equation')
print('    options are:')
for key, value in options.items():
    print('        ', (key+':').ljust(15), value)

for interior in [True, False]:

    print('\n    Testing', 'interior' if interior else 'exterior', 'evaluation.')

    if interior:
        solution_function    = interior_solution_function
        solution_function_dn = interior_solution_function_dn
        far_targ             = PointSet(c=np.array(0.1+0.1j))
    else:
        solution_function    = exterior_solution_function
        solution_function_dn = exterior_solution_function_dn
        far_targ             = PointSet(c=np.array(1.5+1.5j))

    ############################################################################
    # get densities

    sign = 1 if interior else -1
    bc = solution_function(bdy.x, bdy.y)
    bcn = solution_function_dn(bdy.x, bdy.y, bdy.normal_x, bdy.normal_y)
    slp = bcn * sign
    dlp = -bc * sign

    ############################################################################
    # Setup qfs operators

    qfs = Helmholtz_QFS(bdy, interior, True, True, k, options)

    ############################################################################
    # Solve for effective density and extract effective sources

    spot = qfs([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    source = qfs.source

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    ua = solution_function(bdy.x, bdy.y)
    ue = Apply_CF(source, bdy, spot, k)
    err = np.abs(ua - ue).max()/scale
    print('        Error on surface:  {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    shell = PointSet(c=qfs.get_normal_shift(shell_distance))
    ua = solution_function(shell.x, shell.y)
    ue = Apply_CF(source, shell, spot, k)
    err = np.abs(ua - ue).max()/scale
    print('        Error off-surface: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    ua = solution_function(far_targ.x, far_targ.y)
    ue = Apply_CF(source, far_targ, spot, k)
    err = np.abs(ua - ue).max()/scale
    print('        Error, far-field:  {:0.2e}'.format(err))
