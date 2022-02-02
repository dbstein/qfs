import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

import pybie2d
star = pybie2d.misc.curve_descriptions.star
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
PointSet = pybie2d.point_set.PointSet
Modified_Helmholtz_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply
MHGF = pybie2d.kernels.low_level.modified_helmholtz.Modified_Helmholtz_Greens_Function
MHGFP = pybie2d.kernels.low_level.modified_helmholtz.Modified_Helmholtz_Greens_Function_Derivative

from qfs.modified_helmholtz_qfs import Modified_Helmholtz_QFS

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
s2c_type = 'LU'
# Form, Apply, or Circulant
b2c_type = 'Apply'
# Helmholtz parameter
k = 1.0

################################################################################
# Test problems

def interior_solution_function(x, y):
    xc = 1.1
    yc = 1.1
    return MHGF(np.hypot(x-xc, y-yc), k)
def interior_solution_function_dn(x, y, nx, ny):
    xc = 1.1
    yc = 1.1
    d = MHGFP(np.hypot(x-xc, y-yc), k)
    return -((x-xc)*nx*d + (y-yc)*ny*d)
def exterior_solution_function(x, y):
    xc = 0.5
    yc = 0.0
    return MHGF(np.hypot(x-xc, y-yc), k)
def exterior_solution_function_dn(x, y, nx, ny):
    xc = 0.5
    yc = 0.0
    d = MHGFP(np.hypot(x-xc, y-yc), k)
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
print('Running QFS Tests for Laplace Equation')
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

    qfs = Modified_Helmholtz_QFS(bdy, interior, True, True, k, options)

    ############################################################################
    # Solve for effective density and extract effective sources

    spot = qfs([slp, dlp])
    scale = max(np.abs(slp).max(), np.abs(dlp).max())

    source = qfs.source

    ############################################################################
    # TEST ERROR IN U ON SURFACE

    ua = solution_function(bdy.x, bdy.y)
    ue = Modified_Helmholtz_Layer_Apply(source, bdy, k=k, charge=spot)
    err = np.abs(ua - ue).max()/scale
    print('        Error on surface:  {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U ON SHELLS

    shell = PointSet(c=qfs.get_normal_shift(shell_distance))
    ua = solution_function(shell.x, shell.y)
    ue = Modified_Helmholtz_Layer_Apply(source, shell, k=k, charge=spot)
    err = np.abs(ua - ue).max()/scale
    print('        Error off-surface: {:0.2e}'.format(err))

    ############################################################################
    # TEST ERROR IN U AT FAR POINT

    ua = solution_function(far_targ.x, far_targ.y)
    ue = Modified_Helmholtz_Layer_Apply(source, far_targ, k=k, charge=spot)
    err = np.abs(ua - ue).max()/scale
    print('        Error, far-field:  {:0.2e}'.format(err))