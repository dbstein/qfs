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
    Modified_Helmholtz_Layer_Apply = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Apply
    Modified_Helmholtz_Layer_Form = pybie2d.kernels.high_level.modified_helmholtz.Modified_Helmholtz_Layer_Form
    Laplace_Layer_Singular_Form = pybie2d.kernels.high_level.laplace.Laplace_Layer_Singular_Form
    numba_k0 = pybie2d.misc.numba_special_functions.numba_k0
    numba_k1 = pybie2d.misc.numba_special_functions.numba_k1
    interpolate_to_p = pybie2d.misc.basic_functions.interpolate_to_p
    Naive_SLP = lambda src, trg: Modified_Helmholtz_Layer_Form(src, trg, ifcharge=True)
    Naive_DLP = lambda src, trg: Modified_Helmholtz_Layer_Form(src, trg, ifdipole=True)
except:
    import warnings
    warnings.warn("Operating in fallback mode, only QFS-D with Form backend will work.")
    from qfs.fallbacks.modified_helmholtz import Modified_Helmholtz_Layer_Form

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

alpert_x = np.array([ 8.371529832014113E-04, 1.239382725542637E-02, 6.009290785739468E-02, 1.805991249601928E-01, 4.142832599028031E-01, 7.964747731112430E-01, 1.348993882467059E+00, 2.073471660264395E+00, 2.947904939031494E+00, 3.928129252248612E+00, 4.957203086563112E+00, 5.986360113977494E+00, 6.997957704791519E+00, 7.999888757524622E+00, 8.999998754306120E+00 ])
alpert_w = np.array([ 3.190919086626234E-03, 2.423621380426338E-02, 7.740135521653088E-02, 1.704889420286369E-01, 3.029123478511309E-01, 4.652220834914617E-01, 6.401489637096768E-01, 8.051212946181061E-01, 9.362411945698647E-01, 1.014359775369075E+00, 1.035167721053657E+00, 1.020308624984610E+00, 1.004798397441514E+00, 1.000395017352309E+00, 1.000007149422537E+00 ])
alpert_a = 10
alpert_g = 2*alpert_x.shape[0]
alpert_W =  np.concatenate([ alpert_w[::-1], alpert_w ])
def get_alpert_I(source):
    alpert_X = np.concatenate([ 2.0*np.pi - alpert_x[::-1]*source.dt, alpert_x*source.dt ])
    return interpolate_to_p(np.eye(source.N), alpert_X)

def Singular_SLP(bdy, k):
    aI = get_alpert_I(bdy)
    x = bdy.x
    y = bdy.y
    nx = bdy.normal_x
    ny = bdy.normal_y
    w = bdy.weights

    N = x.shape[0]
    sel1 = np.arange(N)
    sel2 = sel1[:,None]
    sel = np.mod(sel1 + sel2, N)
    Yx = x[sel]
    Yy = y[sel]
    IYx = aI.dot(Yx)
    IYy = aI.dot(Yy)
    FYx = np.row_stack(( IYx, Yx[alpert_a:(-alpert_a+1)] ))
    FYy = np.row_stack(( IYy, Yy[alpert_a:(-alpert_a+1)] ))
    W = w[sel]
    IW = aI.dot(W)*alpert_W[:,None]
    FW = np.row_stack(( IW, W[alpert_a:(-alpert_a+1)] ))
    # evaluate greens function
    dx = x - FYx
    dy = y - FYy
    r = np.sqrt(dx**2 + dy**2)
    kr = k*r
    inv_twopi = 0.5/np.pi
    GF = inv_twopi*numba_k0(kr)
    A = (GF*FW).T
    # reconstruct
    Ag = A[:,:alpert_g]
    IAg = Ag.dot(aI)
    IAg[:,alpert_a:(-alpert_a+1)] += A[:,alpert_g:]
    # reorganize
    inv_sel = np.mod((sel1 + sel2[::-1] + 1), N)
    MAT = IAg[sel1, inv_sel.T].T
    MAT /= w
    return MAT.T*w
def Singular_DLP(bdy, k):
    DL = Laplace_Layer_Singular_Form(bdy, ifdipole=True)
    C = Alpert_DLP_Correction(bdy, k)
    C /= bdy.weights
    C = C.T*bdy.weights
    return DL-C
def Alpert_DLP_Correction(bdy, k):
    aI = get_alpert_I(bdy)
    x = bdy.x
    y = bdy.y
    nx = bdy.normal_x
    ny = bdy.normal_y
    w = bdy.weights

    N = x.shape[0]
    sel1 = np.arange(N)
    sel2 = sel1[:,None]
    sel = np.mod(sel1 + sel2, N)
    Yx = x[sel]
    Yy = y[sel]
    IYx = aI.dot(Yx)
    IYy = aI.dot(Yy)
    FYx = np.row_stack(( IYx, Yx[alpert_a:(-alpert_a+1)] ))
    FYy = np.row_stack(( IYy, Yy[alpert_a:(-alpert_a+1)] ))
    W = w[sel]
    IW = aI.dot(W)*alpert_W[:,None]
    FW = np.row_stack(( IW, W[alpert_a:(-alpert_a+1)] ))
    # evaluate greens function
    dx = x - FYx
    dy = y - FYy
    r = np.sqrt(dx**2 + dy**2)
    kr = k*r
    inv_twopi = 0.5/np.pi
    k1kr_m_kir2 = inv_twopi*(k*numba_k1(kr)/r - (1.0/r)**2)
    GFx = dx*k1kr_m_kir2
    GFy = dy*k1kr_m_kir2
    A = ((nx*GFx+ny*GFy)*FW).T
    # reconstruct
    Ag = A[:,:alpert_g]
    IAg = Ag.dot(aI)
    IAg[:,alpert_a:(-alpert_a+1)] += A[:,alpert_g:]
    # reorganize
    inv_sel = np.mod((sel1 + sel2[::-1] + 1), N)
    return IAg[sel1, inv_sel.T].T
def IDLP(bdy, k):
    return Singular_DLP(bdy, k) - 0.5*np.eye(bdy.N)
def EDLP(bdy, k):
    return Singular_DLP(bdy, k) + 0.5*np.eye(bdy.N)
def generate_self_kfunc(k, _func):
    def func(bdy):
        return _func(bdy, k)
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
    if 'b2c_type' not in options:
        options['b2c_type'] = 'Form'
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

def build_s2c(k, options):
    return QFS_s2c_factory(Naive_SLP(k))(options)

def build_b2c(slp, dlp, interior, k, options):
    b2c_type = options['b2c_type']
    singular = options['singular']
    funcs = []

    if singular and b2c_type == 'Form':
        if slp:
            funcs.append( generate_self_kfunc(k, Singular_SLP) )
        if dlp:
            funcs.append( generate_self_kfunc(k, IDLP if interior else EDLP) )

    if singular and b2c_type == 'Apply':
        raise NotImplementedError('Singular Apply Functions for Modified Helmholtz Kernel Not Implemented.')

    if singular and b2c_type == 'Circulant':
        raise NotImplementedError('Circulant Apply Functions for Singular Modified Helmholtz Kernel Not Implemented.')

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
    def __init__(self, bdy, interior, slp, dlp, k, options=None, **kwargs):
        """
        Modified Helmholtz QFS
            Effective source representation is SLP, i.e.
                SLP(src, trg, k)

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
        kwargs = set_default_kwargs(kwargs)
        b2c = build_b2c(slp, dlp, interior, k, options)
        s2c = build_s2c(k, options)
        cu = QFS_Scalar_Check_Upsampler()
        super().__init__(bdy, interior, b2c, s2c, cu, **kwargs)
