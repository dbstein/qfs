import numpy as np

from qfs._two_d_qfs import QFS_Pressure, QFS_B2C
from qfs._two_d_qfs import QFS_LU_Inverter, QFS_Circulant_Inverter
from qfs._two_d_qfs import QFS_Square_QR_Inverter, QFS_Rectangular_QR_Inverter
from qfs._two_d_qfs import QFS_Square_SVD_Inverter, QFS_Rectangular_SVD_Inverter
from qfs._two_d_qfs import B2C_Easy_Apply, B2C_Easy_Circulant
from qfs._two_d_qfs import QFS_Vector_Check_Upsampler
from qfs._two_d_qfs import QFS_s2c_factory

################################################################################
# Setup Test

try:
    import pybie2d
    Laplace_Layer_Apply = pybie2d.kernels.high_level.laplace.Laplace_Layer_Apply
    Laplace_Layer_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Form
    Stokes_Layer_Apply = pybie2d.kernels.high_level.stokes.Stokes_Layer_Apply
    Stokes_Layer_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Form
    Stokes_Layer_Singular_Form = pybie2d.kernels.high_level.stokes.Stokes_Layer_Singular_Form
    Singular_SLP = lambda src: Stokes_Layer_Singular_Form(src, ifforce=True)
    Singular_DLP = lambda src: Stokes_Layer_Singular_Form(src, ifdipole=True)
except:
    from qfs.fallbacks.laplace import Laplace_Layer_Form
    from qfs.fallbacks.stokes import Stokes_Layer_Form
Naive_SLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifforce=True)
Naive_DLP = lambda src, trg: Stokes_Layer_Form(src, trg, ifdipole=True)

# End Setup Test
################################################################################

############################################################################
# Apply and Circulant Functions

# Layer Form Functions for Stokes
def Fixer(src, trg, SLP):
    Nxx = trg.normal_x[:,None]*src.normal_x*src.weights
    Nxy = trg.normal_x[:,None]*src.normal_y*src.weights
    Nyx = trg.normal_y[:,None]*src.normal_x*src.weights
    Nyy = trg.normal_y[:,None]*src.normal_y*src.weights
    return SLP(src, trg) + np.array(np.bmat([[Nxx, Nxy], [Nyx, Nyy]]))
Fixed_SLP =  lambda src, trg: Fixer(src, trg, Naive_SLP)
IDLP = lambda src: Singular_DLP(src) - 0.5*np.eye(2*src.N)
EDLP = lambda src: Singular_DLP(src) + 0.5*np.eye(2*src.N)

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
def Kress_SLP_Apply(source, x):
    Vh = np.fft.fft(get_Kress_V(source))
    wx = x*source.weights
    u1 = Laplace_Layer_Apply(source, charge=x)
    u1 -= np.log(source.speed)/(2*np.pi)*wx
    u2 = np.fft.ifft(np.fft.fft(wx)*Vh).real
    return u1 + u2
def Kress_Stokes_SLP_Apply(source, x):
    N = source.N
    mu = 1.0
    U = Stokes_Layer_Apply(source, forces=x)
    x1 = x[:N]
    x2 = x[N:]
    Lu = Kress_SLP_Apply(source, x1) - Laplace_Layer_Apply(source, charge=x1)
    Lv = Kress_SLP_Apply(source, x2) - Laplace_Layer_Apply(source, charge=x2)
    LU = np.concatenate([Lu, Lv])
    tx = source.tangent_x
    ty = source.tangent_y
    wtx = source.weights*(tx*x1 + ty*x2) / (4*np.pi*mu)
    DU = np.concatenate([tx*wtx, ty*wtx])
    return U + 0.5*mu*LU + DU
def Singular_Stokes_DLP_Apply(source, x):
    N = source.N
    x1 = x[:N]
    x2 = x[N:]
    U = Stokes_Layer_Apply(source, dipstr=x)
    u = U[:N]
    v = U[N:]
    sc = -0.5*source.curvature*source.weights/np.pi
    tx = source.tangent_x
    ty = source.tangent_y
    sct = sc*(tx*x1 + ty*x2)
    u += tx*sct
    v += ty*sct
    return np.concatenate([u, v])
def IDLP_Apply(source, x):
    return Singular_Stokes_DLP_Apply(source, x) - 0.5*x
def EDLP_Apply(source, x):
    return Singular_Stokes_DLP_Apply(source, x) + 0.5*x
def Stokes_SLP_Circulant_Form(source):
    N = source.N
    weights = source.weights[0]
    mu = 1.0
    sx = source.x[0]
    sxT = source.x
    sy = source.y[0]
    syT = source.y
    tx = source.tangent_x[0]
    ty = source.tangent_y[0]
    dx = sxT - sx
    dy = syT - sy
    rr = dx*dx + dy*dy
    rr[0] = np.Inf
    irr = 1.0/rr
    weights = weights/(4.0*np.pi*mu)
    A00 = weights*dx*dx*irr
    A01 = weights*dx*dy*irr
    A11 = weights*dy*dy*irr
    A00[0] = weights*tx*tx
    A01[0] = weights*tx*ty
    A11[0] = weights*ty*ty
    S = Laplace_SLP_Circulant_Form(source)
    muS = 0.5*mu*S
    A = np.empty((2*N,2), dtype=float)
    A[:N, 0] = A00 + muS
    A[:N, 1] = A01
    A[N:, 0] = A01
    A[N:, 1] = A11 + muS
    return A
def Stokes_DLP_Circulant_Form(source):
    ssource = source.Generate_1pt_Circulant_Boundary()
    w = Naive_DLP(ssource, source)
    sn = source.N
    scale = -0.5*source.curvature[0]*source.weights[0]/np.pi
    tx = source.tangent_x[0]
    ty = source.tangent_y[0]
    w[0, 0] = scale*tx*tx
    w[sn,0] = scale*tx*ty
    w[0, 1] = scale*tx*ty
    w[sn,1] = scale*ty*ty
    return w
def Laplace_SLP_Circulant_Form(source):
    V = get_Kress_V(source)
    ssource = source.Generate_1pt_Circulant_Boundary()
    A = Laplace_Layer_Form(ssource, target=source, ifcharge=True)
    A[0] = -np.log(source.speed[0])/(2*np.pi)*source.weights[0]
    return A[:,0] + V*source.weights
def ICDLP(source):
    w = Stokes_DLP_Circulant_Form(source)
    sn = source.N
    w[0, 0] -= 0.5
    w[sn,1] -= 0.5
    return w
def ECDLP(source):
    w = Stokes_DLP_Circulant_Form(source)
    sn = source.N
    w[0, 0] += 0.5
    w[sn,1] += 0.5
    return w
# Pressure Form Functions for Stokes
def Pressure_SLP(src, trg):
    # extract from src/trg
    sx = src.x
    sy = src.y
    w = src.weights
    tx = trg.x[:,None]
    ty = trg.y[:,None]
    # computation
    dx = tx - sx
    dy = ty - sy
    r2 = dx*dx + dy*dy
    ir2 = 1.0/r2
    sir2 = w*ir2*0.5/np.pi
    out = np.empty([trg.N, 2*src.N])
    out[:, :src.N] = dx*sir2
    out[:, src.N:] = dy*sir2
    return out
def Pressure_DLP(src, trg):
    # extract from src/trg
    sx = src.x
    sy = src.y
    w = src.weights
    nx = src.normal_x
    ny = src.normal_y
    tx = trg.x[:,None]
    ty = trg.y[:,None]
    # computation
    dx = tx - sx
    dy = ty - sy
    r2 = dx*dx + dy*dy
    ir2 = 1.0/r2
    rdotnir4 = (dx*nx + dy*ny)*ir2*ir2
    ww = w/np.pi
    out = np.empty([trg.N, 2*src.N])
    out[:, :src.N] = ww*(-nx*ir2 + 2*rdotnir4*dx)
    out[:, src.N:] = ww*(-ny*ir2 + 2*rdotnir4*dy)
    return out

############################################################################
# Configuration for Stokes

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
        kwargs['source_upsample_factor'] = 1.3
    if 'check_upsample_factor' not in kwargs:
        kwargs['check_upsample_factor'] = 1.6
    return kwargs

def build_s2c(options, interior):
    NSLP = Fixed_SLP if True else Naive_SLP
    return QFS_s2c_factory(NSLP, vector=True)(options)

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
            funcs.append( B2C_Easy_Apply(Kress_Stokes_SLP_Apply) )
        if dlp:
            funcs.append( B2C_Easy_Apply(IDLP_Apply if interior else EDLP_Apply) )

    if singular and b2c_type == 'Circulant':
        if slp:
            funcs.append( B2C_Easy_Circulant(Stokes_SLP_Circulant_Form, one_column=True, vector=True) )
        if dlp:
            funcs.append( B2C_Easy_Circulant(ICDLP if interior else ECDLP, one_column=True, vector=True) )

    if not singular and b2c_type == 'Form':
        if slp:
            funcs.append( Naive_SLP )
        if dlp:
            funcs.append( Naive_DLP )

    if not singular and b2c_type == 'Apply':
        if slp:
            funcs.append( B2C_Easy_Apply(lambda src, trg, x: Stokes_Layer_Apply(src, trg, forces=x)) )
        if dlp:
            funcs.append( B2C_Easy_Apply(lambda src, trg, x: Stokes_Layer_Apply(src, trg, dipstr=x)) )

    if not singular and b2c_type == 'Circulant':
        if slp:
            funcs.append( B2C_Easy_Circulant(Naive_SLP, one_column=False, vector=True) )
        if dlp:
            funcs.append( B2C_Easy_Circulant(Naive_DLP, one_column=False, vector=True) )

    return QFS_B2C(funcs, singular, vector=True)

def build_ps2c(options):
    return Pressure_SLP

def build_pb2c(slp, dlp):
    funcs = []
    if slp:
        funcs.append( Pressure_SLP )
    if dlp:
        funcs.append( Pressure_DLP )
    return funcs

class Stokes_QFS(QFS_Pressure):
    def __init__(self, bdy, interior, slp, dlp, options=None, **kwargs):
        """
        Stokes QFS
            Effective source representation is SLP, i.e.
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
        """
        options = set_default_options(options)
        kwargs = set_default_kwargs(kwargs)
        b2c = build_b2c(slp, dlp, interior, options)
        s2c = build_s2c(options, interior)
        pb2c = build_pb2c(slp, dlp)
        ps2c = build_ps2c(options)
        cu = QFS_Vector_Check_Upsampler()
        super().__init__(bdy, interior, b2c, s2c, cu, pb2c, ps2c, **kwargs)
