import numpy as np

from qfs._two_d_qfs import QFS, QFS_B2C
from qfs._two_d_qfs import QFS_LU_Inverter, QFS_Circulant_Inverter
from qfs._two_d_qfs import QFS_Square_QR_Inverter, QFS_Rectangular_QR_Inverter
from qfs._two_d_qfs import QFS_Square_SVD_Inverter, QFS_Rectangular_SVD_Inverter
from qfs._two_d_qfs import B2C_Easy_Apply, B2C_Easy_Circulant
from qfs._two_d_qfs import QFS_Scalar_Check_Upsampler
from qfs._two_d_qfs import QFS_s2c_factory

################################################################################
# Setup Test

try:
    import pybie2d
    Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
    Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
    Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
    Singular_SLP = lambda src: Laplace_Layer_Singular_Form(src, ifcharge=True)
    Singular_DLP = lambda src: Laplace_Layer_Singular_Form(src, ifdipole=True)
    IDLP = lambda src: Singular_DLP(src) - 0.5*np.eye(src.N)
    EDLP = lambda src: Singular_DLP(src) + 0.5*np.eye(src.N)
except:
    import warnings
    warnings.warn("Operating in fallback mode, only QFS-D with Form backend will work.")
    from qfs.fallbacks.laplace import Laplace_Layer_Form
Naive_SLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifcharge=True)
Naive_DLP = lambda src, trg: Laplace_Layer_Form(src, trg, ifdipole=True)

# End Setup Test
################################################################################

############################################################################
# Apply and Circulant Functions

def get_Kress_V(source):
    N = source.N
    dt = source.dt
    v1 = 4.0*np.sin(np.pi*np.arange(N)/N)**2
    v1[0] = 1.0
    V1 = 0.5*np.log(v1)/dt
    v2 = np.abs(np.fft.fftfreq(N, 1.0/N))
    v2[0] = np.Inf
    v2[int(N/2)] = np.Inf
    V2 = 0.5*np.fft.ifft(1.0 / v2).real/dt
    return V1/N + V2
def Kress_SLP_Apply(source, x):
    Vh = np.fft.fft(get_Kress_V(source))
    wx = x*source.weights
    u1 = Laplace_Layer_Apply(source, charge=x)
    u1 -= np.log(source.speed)/(2*np.pi)*wx
    u2 = np.fft.ifft(np.fft.fft(wx)*Vh).real
    return u1 + u2
def Singular_DLP_Apply(source, x):
    u = Laplace_Layer_Apply(source, dipstr=x)
    u -= 0.25*source.curvature*source.weights/np.pi*x
    return u
def IDLP_Apply(source, x):
    return Singular_DLP_Apply(source, x) - 0.5*x
def EDLP_Apply(source, x):
    return Singular_DLP_Apply(source, x) + 0.5*x
def Laplace_SLP_Circulant_Form(source):
    V = get_Kress_V(source)
    ssource = source.Generate_1pt_Circulant_Boundary()
    A = Laplace_Layer_Form(ssource, target=source, ifcharge=True)
    A[0] = -np.log(source.speed[0])/(2*np.pi)*source.weights[0]
    return A[:,0] + V*source.weights
def Laplace_DLP_Circulant_Form(source):
    ssource = source.Generate_1pt_Circulant_Boundary()
    w = Naive_DLP(ssource, source)
    w[0] = -0.25*source.curvature[0]*source.weights[0]/np.pi
    return w[:,0]
def ICDLP(source):
    w = Laplace_DLP_Circulant_Form(source)
    w[0] -= 0.5
    return w
def ECDLP(source):
    w = Laplace_DLP_Circulant_Form(source)
    w[0] += 0.5
    return w

############################################################################
# Configuration for Laplace

def set_default_options(options):
    if options is None:
        options = {}
    if 'b2c_type' not in options:
        options['b2c_type'] = 'Apply'
    if 's2c_type' not in options:
        options['s2c_type'] = 'QR'
    if 'singular' not in options:
        options['singular'] = True
    return options

def set_default_kwargs(kwargs):
    if 'tol' not in kwargs:
        kwargs['tol'] = 1e-14
    if 'shift_type' not in kwargs:
        kwargs['shift_type'] = 5
    if 'source_upsample_factor' not in kwargs:
        kwargs['source_upsample_factor'] = 1.0
    if 'check_upsample_factor' not in kwargs:
        kwargs['check_upsample_factor'] = 1.0
    return kwargs

def build_s2c(options):
    return QFS_s2c_factory(Naive_SLP)(options)

def build_b2c(slp, dlp, interior, options):
    b2c_type = options['b2c_type']
    singular = options['singular']
    funcs = []

    if singular and b2c_type == 'Form':
        if slp:
            funcs.append( Singular_SLP )
        if dlp:
            funcs.append( IDLP if interior else EDLP )

    if singular and b2c_type == 'Apply':
        if slp:
            funcs.append( B2C_Easy_Apply(Kress_SLP_Apply) )
        if dlp:
            funcs.append( B2C_Easy_Apply(IDLP_Apply if interior else EDLP_Apply) )

    if singular and b2c_type == 'Circulant':
        if slp:
            funcs.append( B2C_Easy_Circulant(Laplace_SLP_Circulant_Form, one_column=True) )
        if dlp:
            funcs.append( B2C_Easy_Circulant(ICDLP if interior else ECDLP, one_column=True) )

    if not singular and b2c_type == 'Form':
        if slp:
            funcs.append( Naive_SLP )
        if dlp:
            funcs.append( Naive_DLP )

    if not singular and b2c_type == 'Apply':
        if slp:
            funcs.append( B2C_Easy_Apply(lambda src, trg, x: Laplace_Layer_Apply(src, trg, charge=x)) )
        if dlp:
            funcs.append( B2C_Easy_Apply(lambda src, trg, x: Laplace_Layer_Apply(src, trg, dipstr=x)) )

    if not singular and b2c_type == 'Circulant':
        if slp:
            funcs.append( B2C_Easy_Circulant(Naive_SLP, one_column=False) )
        if dlp:
            funcs.append( B2C_Easy_Circulant(Naive_DLP, one_column=False) )

    return QFS_B2C(funcs, singular)

class Laplace_QFS(QFS):
    def __init__(self, bdy, interior, slp, dlp, options=None, **kwargs):
        """
        Laplace QFS
            Effective source representation is the Single Layer, i.e.
                SLP(src, trg)

        bdy (GlobalSmoothBoundary)
            boundary to do eval for
        interior (bool)
            which side of boundary to generate evalution operators for
        slp (bool): include SLP in boundary --> check functions?
        dlp (bool): include DLP in boundary --> check functions?
        options (None, or dict):
            acceptable options:
            b2c_type (string): 'Form', 'Apply', or 'Cicrulant'
                default: 'Apply'
            s2c_type (string): 'LU', 'SVD', or 'Circulant'
                default: 'LU'
            singular (bool):   use on or off surface eval
                default: True
        kwargs:
            see documentation for main QFS class
        """
        options = set_default_options(options)
        kwargs = set_default_kwargs(kwargs)
        b2c = build_b2c(slp, dlp, interior, options)
        s2c = build_s2c(options)
        cu = QFS_Scalar_Check_Upsampler()
        super().__init__(bdy, interior, b2c, s2c, cu, **kwargs)
