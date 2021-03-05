import numpy as np
import pybie2d

from qfs._two_d_qfs import QFS, QFS_B2C
from qfs._two_d_qfs import QFS_LU_Inverter, QFS_Square_SVD_Inverter, QFS_Circulant_Inverter
from qfs._two_d_qfs import B2C_Easy_Apply, B2C_Easy_Circulant

from pyfmmlib2d import HFMM
from scipy.special import hankel1

################################################################################
# Apply Functions

def Apply_SLP(src, trg, tau, k=1.0):
    source = src.get_stacked_boundary()
    target = trg.get_stacked_boundary()
    wtau = tau*src.weights
    out = HFMM(source, target, charge=wtau, helmholtz_parameter=k, compute_target_potential=True)
    return out['target']['u']
def Apply_DLP(src, trg, tau, k=1.0):
    source = src.get_stacked_boundary()
    target = trg.get_stacked_boundary()
    normals = src.get_stacked_normal()
    wtau = tau*src.weights
    out = HFMM(source, target, dipstr=wtau, dipvec=normals, helmholtz_parameter=k, compute_target_potential=True)
    return out['target']['u']
def SLP(src, trg, k=1.0):
    source = src.get_stacked_boundary()
    target = trg.get_stacked_boundary()
    weights = src.weights
    SX = source[0]
    SY = source[1]
    TX = target[0][:,None]
    TY = target[1][:,None]
    scale = 0.25j*weights
    dx = TX - SX
    dy = TY - SY
    r = np.hypot(dx, dy)
    return hankel1(0, k*r)*scale
def DLP(src, trg, k=1.0):
    source = src.get_stacked_boundary()
    nx = src.normal_x
    ny = src.normal_y
    target = trg.get_stacked_boundary()
    weights = src.weights
    SX = source[0]
    SY = source[1]
    TX = target[0][:,None]
    TY = target[1][:,None]
    scale = 0.25j*weights
    dx = TX - SX
    dy = TY - SY
    r = np.hypot(dx, dy)
    GD = hankel1(1, k*r)*k/r
    G = (nx*dx + ny*dy) * GD * scale
    return G

def Naive_SLP(k):
    def func(src, trg):
        return SLP(src, trg, k)
    return func
def Naive_DLP(k):
    def func(src, trg):
        return DLP(src, trg, k)
    return func
def Naive_CF(k):
    def func(src, trg):
        return -1j*k*SLP(src, trg, k) + DLP(src, trg, k)
    return func
def Naive_SLP_Apply(k):
    def func(src, trg, x):
        return Apply_SLP(src, trg, x, k)
    return func
def Naive_DLP_Apply(k):
    def func(src, trg, x):
        return Apply_DLP(src, trg, x, k)
    return func

############################################################################
# Configuration for Modified Helmholtz

def set_default_options(options):
    if options is None:
        options = {}
    if 'tol' not in options:
        options['tol'] = 1e-14
    if 'shift_type' not in options:
        options['shift_type'] = 2
    if 'b2c_type' not in options:
        options['b2c_type'] = 'Apply'
    if 's2c_type' not in options:
        options['s2c_type'] = 'LU'
    if 'singular' not in options:
        options['singular'] = False
    return options

def build_s2c(k, options):
    s2c_type = options['s2c_type']
    
    ncf = Naive_CF(k)

    if s2c_type == 'LU':
        s2c = QFS_LU_Inverter(ncf)
    elif s2c_type == 'SVD':
        s2c = QFS_Square_SVD_Inverter(ncf)
    elif s2c_type == 'Circulant':
        s2c = QFS_Circulant_Inverter(ncf)

    return s2c

def build_b2c(slp, dlp, interior, k, options):
    b2c_type = options['b2c_type']
    singular = options['singular']
    funcs = []

    if singular:
        raise NotImplementedError('Singular kernels not implemented yet for Helmholtz')

    if not singular and b2c_type == 'Form':
        if slp:
            funcs.append( Naive_SLP(k) )
        if dlp:
            funcs.append( Naive_DLP(k) )

    if not singular and b2c_type == 'Apply':
        if slp:
            funcs.append( B2C_Easy_Apply( Naive_SLP_Apply(k) ) )
        if dlp:
            funcs.append( B2C_Easy_Apply( Naive_DLP_Apply(k) ) )

    if not singular and b2c_type == 'Circulant':
        if slp:
            funcs.append( B2C_Easy_Circulant(Naive_SLP(k), one_column=False) )
        if dlp:
            funcs.append( B2C_Easy_Circulant(Naive_DLP(k), one_column=False) )

    return QFS_B2C(funcs, singular, form_b2c_mats=not singular and b2c_type == 'Form')

class Helmholtz_QFS(QFS):
    def __init__(self, bdy, interior, slp, dlp, k, options=None):
        """
        Helmholtz QFS
        bdy (GlobalSmoothBoundary)
            boundary to do eval for
        interior (bool)
            which side of boundary to generate evalution operators for
        slp (bool): include SLP in boundary --> check functions?
        dlp (bool): include DLP in boundary --> check functions?
        k (float): Helmholtz parameter
        options (None, or dict):
            acceptable options:
            tol (float): estimated error tolerance
                default: 1e-14
            shift_type:  how to estimated shifted boundaries, see qfs documentation
                default: 2
            b2c_type (string): 'Form', 'Apply', or 'Cicrulant'
                default: 'Apply'
            s2c_type (string): 'LU', 'SVD', or 'Circulant'
                default: 'LU'
            singular (bool):   use on or off surface eval
                default: True
        """
        options = set_default_options(options)
        b2c = build_b2c(slp, dlp, interior, k, options)
        s2c = build_s2c(k, options)
        tol = options['tol']
        st = options['shift_type']
        super().__init__(bdy, interior, b2c, s2c, tol=tol, shift_type=st)
