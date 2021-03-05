import numpy as np
import pybie2d

from qfs._two_d_qfs import QFS, QFS_B2C
from qfs._two_d_qfs import QFS_LU_Inverter, QFS_Square_SVD_Inverter, QFS_Circulant_Inverter
from qfs._two_d_qfs import B2C_Easy_Apply, B2C_Easy_Circulant

################################################################################
# Setup Test

Modified_Helmholtz_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply
Modified_Helmholtz_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
Naive_SLP = lambda src, trg: Modified_Helmholtz_Layer_Form(src, trg, ifcharge=True)
Naive_DLP = lambda src, trg: Modified_Helmholtz_Layer_Form(src, trg, ifdipole=True)

def Naive_SLP(k):
    def func(src, trg):
        return Modified_Helmholtz_Layer_Form(src, trg, k=k, ifcharge=True)
    return func
def Naive_DLP(k):
    def func(src, trg):
        return Modified_Helmholtz_Layer_Form(src, trg, k=k, ifdipole=True)
    return func
def Naive_SLP_Apply(k):
    def func(src, trg, x):
        return Modified_Helmholtz_Layer_Apply(src, trg, k=k, charge=x)
    return func
def Naive_DLP_Apply(k):
    def func(src, trg, x):
        return Modified_Helmholtz_Layer_Apply(src, trg, k=k, dipstr=x)
    return func

# End Setup Test
################################################################################

############################################################################
# Apply and Circulant Functions

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
    
    nslp = Naive_SLP(k)

    if s2c_type == 'LU':
        s2c = QFS_LU_Inverter(nslp)
    elif s2c_type == 'SVD':
        s2c = QFS_Square_SVD_Inverter(nslp)
    elif s2c_type == 'Circulant':
        s2c = QFS_Circulant_Inverter(nslp)

    return s2c

def build_b2c(slp, dlp, interior, k, options):
    b2c_type = options['b2c_type']
    singular = options['singular']
    funcs = []

    if singular:
        raise NotImplementedError('Singular kernels not implemented yet for Modified Helmholtz')

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

    return QFS_B2C(funcs, singular)

class Modified_Helmholtz_QFS(QFS):
    def __init__(self, bdy, interior, slp, dlp, k, options=None):
        """
        Modified Helmholtz QFS
        bdy (GlobalSmoothBoundary)
            boundary to do eval for
        interior (bool)
            which side of boundary to generate evalution operators for
        slp (bool): include SLP in boundary --> check functions?
        dlp (bool): include DLP in boundary --> check functions?
        k (float): Helmholtz parameter (real only!)
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
