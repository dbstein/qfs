import numpy as np
import scipy as sp
import scipy.signal
import warnings
import shapely.geometry as shg

try:
    import pybie2d
    GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary
    PointSet = pybie2d.point_set.PointSet
except:
    from qfs.fallbacks.boundaries import PointSet
    from qfs.fallbacks.boundaries import Global_Smooth_Boundary as GSB

M_EPS = np.log(np.finfo(float).eps) / (-2*np.pi)

################################################################################
# Boundary Shifting Functions

def shift_bdy_complex(bdy, M):
    """
    Shift bdy parametrization M*h units in imaginary direction
    if M > 0, shifts into interior
    if M < 0, shifts into exterior
    """
    return np.fft.ifft(np.fft.fft(bdy.c)*np.exp(-M*bdy.dt*bdy.k))
def shift_bdy_normal(bdy, M):
    """
    Shift bdy M*h units along normal
    if M > 0, shifts into interior
    if M < 0, shifts into exterior
    """
    return bdy.c - M*bdy.dt*bdy.normal_c
def shift_bdy_weighted_normal(bdy, M):
    """
    Shift bdy M*h*speed units along normal
    if M > 0, shifts into interior
    if M < 0, shifts into exterior
    """
    return bdy.c - M*bdy.dt*bdy.speed*bdy.normal_c
def shift_bdy_int(bdy, M, modes):
    """
    Shift bdy 'M*h units' according to formula in paper
    if modes=1, this is speed-weigthed normal translation
    if M > 0, shifts into interior
    if M < 0, shifts into exterior
    """
    kk = -M*bdy.k*bdy.dt
    DXX = np.ones(bdy.N, dtype=float)
    DXY = kk.copy()
    if True: # faster version of the below
        kkj = kk.copy()
        fac = 1
        for j in range(1, modes+1):
            kkj *= kk
            fac *= (2*j)
            DXX += kkj/fac
            kkj *= kk
            fac *= (2*j+1)
            DXY += kkj/fac
    else:
        for j in range(1, modes+1):
            DXX += kk**(2*j)/sp.special.factorial(2*j)
            DXY += kk**(2*j+1)/sp.special.factorial(2*j+1)
    DXY = -1j*DXY
    psi = DXY / DXX
    xh = np.fft.fft(bdy.x)
    yh = np.fft.fft(bdy.y)
    out_xh = (xh - psi*yh) / DXX / (1 + psi**2)
    out_yh = (yh + psi*xh) / DXX / (1 + psi**2)
    out_x = np.fft.ifft(out_xh).real
    out_y = np.fft.ifft(out_yh).real
    return out_x + 1j*out_y
def _shift_bdy_int(bdy, M, modes):
    """
    Shift bdy 'M*h units' according to formula in paper
    if modes=1, this is speed-weigthed normal translation
    if M > 0, shifts into interior
    if M < 0, shifts into exterior
    """
    kk = -M*bdy.k*bdy.dt
    DXX = np.ones(bdy.N, dtype=float)
    DXY = kk.copy()
    for j in range(1, modes+1):
        DXX += kk**(2*j)/sp.special.factorial(2*j)
        DXY += kk**(2*j+1)/sp.special.factorial(2*j+1)
    DXY = -1j*DXY
    psi = DXY / DXX
    xh = np.fft.fft(bdy.x)
    yh = np.fft.fft(bdy.y)
    out_xh = (xh - psi*yh) / DXX / (1 + psi**2)
    out_yh = (yh + psi*xh) / DXX / (1 + psi**2)
    out_x = np.fft.ifft(out_xh).real
    out_y = np.fft.ifft(out_yh).real
    return out_x + 1j*out_y
def polygon_from_bdyc(bdyc):
    """
    Given complex vector of boundary coordinates,
    Return a Shapely Polygon
    """
    return shg.Polygon(zip(bdyc.real, bdyc.imag))
def speed_from_bdyc(bdyc):
    """
    Given a complex vector of boundary coordinates,
    Return the speed of the parametrization
    """
    N = bdyc.size
    assert N % 2 == 0, 'Boundary must have even size'
    k = np.fft.fftfreq(N, 1.0/N)
    k[N // 2] = 0.0
    return np.abs(np.fft.ifft(np.fft.fft(bdyc)*1j*k))
def get_upsampled_bdyc(bdyc, OS):
    """
    Given a complex vector of boundary coordinates, return a resampled
    boundary (as complex vector of coordinates), resampled by factor OS
    The length of the discretization is guaranteed to be even.
    """
    ON = 2 * (int(bdyc.size * OS) // 2)
    return resample(bdyc, ON)

################################################################################
# Resampling Functions

def resample(f, n):
    return sp.signal.resample(f, n)
def vector_resample(f, n):
    """
    Resample a linear vector f where the first element is contained in
    f[:n] and the second in f[n:]
    """
    n1 = f.size // 2
    r1 = resample(f[:n1], n)
    r2 = resample(f[n1:], n)
    return np.concatenate([r1, r2])
def resampling_matrix(n1, n2):
    """
    Generate a resampling matrix from length n1 to length n2
    """
    return resample(np.eye(n1), n2)
def vector_resampling_matrix(n1, n2):
    """
    Block diagonal resampling matrix for vector of length 2*n1 to length 2*n2
    applies reampling matrix independently to the first n1 and second n1
    """
    rs = resampling_matrix(n1, n2)
    mat = np.zeros([2*n2, 2*n1], dtype=float)
    mat[0*n2:1*n2, 0*n1:1*n1] = rs
    mat[1*n2:2*n2, 1*n1:2*n1] = rs
    return mat

################################################################################
# Matrix solvers

class SVD_Solver(object):
    def __init__(self, A, tol=1e-15):
        self.A = A
        self.U, S, self.VH = np.linalg.svd(self.A, full_matrices=False)
        S[S < tol] = np.Inf
        self.SI = 1.0/S
        self.VV = self.VH.conj().T
        self.UU = self.U.conj().T
    def __call__(self, b):
        mult = self.SI[:,None] if len(b.shape) > 1 else self.SI
        return self.VV.dot(mult*self.UU.dot(b))

class QR_Solver(object):
    def __init__(self, A):
        self.A = A
        self.sh = A.shape
        self.overdetermined = self.sh[0] < self.sh[1]
        self.build()
    def build(self):
        if self.overdetermined:
            self.build_overdetermined()
        else:
            self.build_underdetermined()
    def build_overdetermined(self):
        self.Q, self.R = sp.linalg.qr(self.A.T, mode='economic', check_finite=False)
    def build_underdetermined(self):
        self.Q, self.R = sp.linalg.qr(self.A, mode='economic', check_finite=False)
    def __call__(self, b):
        if self.overdetermined:
            return self.solve_overdetermined(b)
        else:
            return self.solve_underdetermined(b)
    def solve_overdetermined(self, b):
        return self.Q.conj().dot(sp.linalg.solve_triangular(self.R, b, trans=1))
    def solve_underdetermined(self, b):
        return sp.linalg.solve_triangular(self.R, self.Q.conj().T.dot(b))

class LU_Solver(object):
    def __init__(self, A):
        self.A = A
        self.A_LU = sp.linalg.lu_factor(self.A)
    def __call__(self, b):
        return sp.linalg.lu_solve(self.A_LU, b)

class Circulant_Solver(object):
    def __init__(self, a, tol):
        self.a = a
        self.tol = tol
        ah = np.fft.fft(a)
        sel = np.abs(ah) < self.tol
        ah = np.fft.fft(a)
        ah[sel] = np.Inf
        self.iah = 1.0 / ah
    def __call__(self, b):
        realit = self.a.dtype == float and b.dtype == float
        w = np.fft.ifft(np.fft.fft(b) * self.iah)
        if realit: w = w.real
        return w

def circulant_multiply(a, b):
    if a.dtype == float and b.dtype == float:
        return np.fft.irfft(np.fft.rfft(a)*np.fft.rfft(b))
    else:
        return np.fft.ifft(np.fft.fft(a)*np.fft.fft(b))

def block_circulant_multiply(a, b, c, d, r1, r2):
    x1 = circulant_multiply(a, r1) + circulant_multiply(b, r2)
    x2 = circulant_multiply(c, r1) + circulant_multiply(d, r2)
    return x1, x2

def CM(A, R):
    return circulant_multiply(A[:,0], R)
def BCM(A, R):
    n = A.shape[0] // 2
    return np.concatenate( block_circulant_multiply(A[:n, 0], A[:n, 1], A[n:, 0], A[n:, 1], R[:n], R[n:]) )
def PBCM(A, R, theta):
    n = A.shape[0] // 2
    a = A[:n, 0]
    b = A[:n, 1]
    c = A[n:, 0]
    d = A[n:, 1]
    rx = R[:n]
    ry = R[n:]
    ct, st = np.cos(theta), np.sin(theta)
    rot_a =  ct*a*ct[0] + ct*b*st[0] + st*c*ct[0] + st*d*st[0]
    rot_b = -ct*a*st[0] + ct*b*ct[0] - st*c*st[0] + st*d*ct[0]
    rot_c = -st*a*ct[0] - st*b*st[0] + ct*c*ct[0] + ct*d*st[0]
    rot_d =  st*a*st[0] - st*b*ct[0] - ct*c*st[0] + ct*d*ct[0]
    rr =  ct*rx + st*ry
    rt = -st*rx + ct*ry
    ur, ut = block_circulant_multiply(rot_a, rot_b, rot_c, rot_d, rr, rt)
    ux = ct*ur - st*ut
    uy = st*ur + ct*ut
    return np.concatenate([ux, uy])

def get_ct_st(t, ct, st):
    if ct is None and st is None:
        n = t.size
        ct = np.cos(t)
        st = np.sin(t)
    else:
        n = ct.size
    return ct, st, n
def fx_fy_2_fr_ft(Rxy, t=None, ct=None, st=None):
    ct, st, n = get_ct_st(t, ct, st)
    Rx = Rxy[:n]
    Ry = Rxy[n:]
    Rr =  ct*Rx + st*Ry
    Rt = -st*Rx + ct*Ry
    return np.concatenate([Rr, Rt])
def fr_ft_2_fx_fy(Rrt, t=None, ct=None, st=None):
    ct, st, n = get_ct_st(t, ct, st)
    Rr = Rrt[:n]
    Rt = Rrt[n:]
    Rx = ct*Rr - st*Rt
    Ry = st*Rr + ct*Rt
    return np.concatenate([Rx, Ry])

class PolarBlockCirculant2x2_solver(object):
    """
    Matrices for equations like Stokes will typically be in the xy-->xy rep
    These are not circulant!  This constructs an operator that transforms inputs
    from xy --> rt, inverts in the rt-->rt rep, and then converts back to xy
    """
    def __init__(self, a, b, c, d, theta, tol):
        """
        a, b, c, d: first columns of the sublocks to the matrix
        ( A  B )
        ( C  D )
        Of the s2c operator in xy-->xy coordinates
        theta: the theta values for the points in the circlular boundary
        """
        self.tol = tol
        self.n = a.size
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.theta = theta
        self.cost = np.cos(theta)
        self.sint = np.sin(theta)
        ct, st = self.cost, self.sint
        self.rot_a =  ct*a*ct[0] + ct*b*st[0] + st*c*ct[0] + st*d*st[0]
        self.rot_b = -ct*a*st[0] + ct*b*ct[0] - st*c*st[0] + st*d*ct[0]
        self.rot_c = -st*a*ct[0] - st*b*st[0] + ct*c*ct[0] + ct*d*st[0]
        self.rot_d =  st*a*st[0] - st*b*ct[0] - ct*c*st[0] + ct*d*ct[0]
        self.solver = BlockCirculant2x2_Solver(self.rot_a, self.rot_b, self.rot_c, self.rot_d, self.tol)
    def __call__(self, f):
        """
        f: forces in the x/y coordinate representation
        returns: velocities in the x/y coordinate representation
        """
        ct, st = self.cost, self.sint
        fx = f[:self.n]
        fy = f[self.n:]
        fr =  ct*fx + st*fy
        ft = -st*fx + ct*fy
        ur, ut = self.solver(fr, ft)
        ux = ct*ur - st*ut
        uy = st*ur + ct*ut
        return np.concatenate([ux, uy])

class BlockCirculant2x2_Solver(object):
    def __init__(self, a, b, c, d, tol):
        self.tol = tol
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.a_solver = Circulant_Solver(self.a, self.tol)
        self.s = d - circulant_multiply(c, self.a_solver(b))
        self.s_solver = Circulant_Solver(self.s, self.tol)
    def __call__(self, r1, r2):
        rhs1 = r2 - circulant_multiply(self.c, self.a_solver(r1))
        x2 = self.s_solver(rhs1)
        rhs2 = r1 - circulant_multiply(self.b, x2)
        x1 = self.a_solver(rhs2)
        return x1, x2

################################################################################
# Classes to handle check upsampling

class QFS_Check_Upsampler(object):
    def build(self, n, un):
        self.n = n
        self.un = un
    def __call__(self, f):
        raise NotImplementedError
class QFS_Scalar_Check_Upsampler(QFS_Check_Upsampler):
    def __call__(self, f):
        return resample(f, self.un)
class QFS_Vector_Check_Upsampler(QFS_Check_Upsampler):
    def __call__(self, f):
        return vector_resample(f, self.un)
class QFS_Null_Check_Upsampler(QFS_Check_Upsampler):
    def __call__(self, f):
        return f

################################################################################
# Classes to handle s2c interaction

class QFS_Inverter(object):
    def __init__(self, func, vector=False):
        self.func = func
        self.vector = vector
    def build(self, source, check, tol):
        self.source = source
        self.check = check
        self.tol = tol
        self.sN = self.source.N
        self.cN = self.check.N
    def resampling_matrix(self):
        func = vector_resampling_matrix if self.vector else resampling_matrix
        return func(self.cN, self.sN)
    def resample(self, tau):
        func = vector_resample if self.vector else resample
        return func(tau, self.sN)
    def shallow_copy(self):
        return type(self)(self.func, self.vector)
    def __call__(self, tau):
        raise NotImplementedError

class QFS_Square_Dense_Inverter(QFS_Inverter):
    def build(self, source, check, tol):
        super().build(source, check, tol)
        # source --> check mat
        self.mat = self.func(self.source, self.check)
        if self.sN == self.cN:
            self.square_source_mat = self.mat
            self.resampled = False
        else:
            UM = self.resampling_matrix()
            self.square_source_mat = self.mat.dot(UM)
            self.resampled = True
        self.Inverter = self.get_inverter()
    def get_inverter(self):
        raise NotImplementedError
    def __call__(self, tau):
        w = self.Inverter(tau)
        if self.resampled:
            w = self.resample(w)
        return w

class QFS_LU_Inverter(QFS_Square_Dense_Inverter):
    def get_inverter(self):
        return LU_Solver(self.square_source_mat)

class QFS_Square_SVD_Inverter(QFS_Square_Dense_Inverter):
    def get_inverter(self):
        return SVD_Solver(self.square_source_mat, self.tol)

class QFS_Square_QR_Inverter(QFS_Square_Dense_Inverter):
    def get_inverter(self):
        return QR_Solver(self.square_source_mat)

class QFS_Rectangular_Dense_Inverter(QFS_Inverter):
    def build(self, source, check, tol):
        super().build(source, check, tol)
        # source --> check mat
        self.mat = self.func(self.source, self.check)
        self.Inverter = self.get_inverter()
    def get_inverter(self):
        raise NotImplementedError
    def __call__(self, tau):
        w = self.Inverter(tau)
        return w

class QFS_Rectangular_SVD_Inverter(QFS_Rectangular_Dense_Inverter):
    def get_inverter(self):
        return SVD_Solver(self.mat, self.tol)

class QFS_Rectangular_QR_Inverter(QFS_Rectangular_Dense_Inverter):
    def get_inverter(self):
        return QR_Solver(self.mat)

class QFS_Circulant_Inverter(QFS_Inverter):
    def build(self, source, check, tol):
        super().build(source, check, tol)
        self.source0 = self.source.Generate_1pt_Circulant_Boundary()
        if self.sN != self.cN:
            self.fine_check = GSB(c=resample(check.c, self.sN))
            self.mat = self.func(self.source0, self.fine_check)
            self.resampled = True
        else:
            self.mat = self.func(self.source0, self.check)
            self.resampled = False
        if self.vector:
            a = self.mat[:self.sN,0]
            b = self.mat[:self.sN,1]
            c = self.mat[self.sN:,0]
            d = self.mat[self.sN:,1]
            theta = self.source.t
            self.Inverter = PolarBlockCirculant2x2_solver(a, b, c, d, theta, tol)
        else:
            self.Inverter = Circulant_Solver(self.mat[:,0], self.tol)
    def __call__(self, tau):
        if self.resampled:
            tau = self.resample(tau)
        return self.Inverter(tau)

def QFS_s2c_factory(func, vector=False):
    def build_s2c(options):
        s2c_type = options['s2c_type']

        if s2c_type == 'LU':
            s2c = QFS_LU_Inverter(func, vector=vector)
        elif s2c_type == 'Square_SVD':
            s2c = QFS_Square_SVD_Inverter(func, vector=vector)
        elif s2c_type == 'Square_QR':
            s2c = QFS_Square_QR_Inverter(func, vector=vector)
        elif s2c_type in ['Rectangular_SVD', 'SVD']:
            s2c = QFS_Rectangular_SVD_Inverter(func, vector=vector)
        elif s2c_type in ['Rectangular_QR', 'QR']:
            s2c = QFS_Rectangular_QR_Inverter(func, vector=vector)
        elif s2c_type == 'Circulant':
            s2c = QFS_Circulant_Inverter(func, vector=vector)
        else:
            raise Exception('s2c_type not recognized.')

        return s2c
    return build_s2c

################################################################################
# Classes to handle b2c interaction

class QFS_B2C(object):
    def __init__(self, funcs, singular, vector=False, form_b2c_mats=False):
        """
        Boundary to Check interaction class for QFS

        funcs: list of functions which create an object with a matvec,
            which evaluate a layer potential from boundary --> check surface
            i.e. you must be able to do this with the func:
            M = func(src, trg)  or  M = func(src)
            u = M.dot(tau)
        singular, bool: whether to use singular evaluation or not
            if singular == True, then each func should be a funciton of one
                variable, giving singular boundary --> boundary evaluation
            if singular == False, then each func should be a function of two
                variables, giving naive boundary --> check surface evaluation.
                the boundary will be upsampled
        """
        self.funcs = funcs
        self.singular = singular
        self.vector = vector
        self.form_b2c_mats = form_b2c_mats
    def build(self, bdy, fine_N=None, check=None):
        self.bdy = bdy
        if self.singular:
            self.resampler = lambda f: f
        else:
            if fine_N is None:
                raise Exception('If QFS_B2C is not singular, you must provide a fine_N.')
            if check is None:
                raise Exception('If QFS_B2C is not singular, you must provide a check surface.')
            self.fine_N = fine_N
            func = vector_resample if self.vector else resample
            self.resampler = lambda f: func(f, self.fine_N)
            self.fine = GSB(c=resample(self.bdy.c, self.fine_N))
            self.check = check
        if self.singular:
            self.mats = [func(self.bdy) for func in self.funcs]
        else:
            self.mats = [func(self.fine, self.check) for func in self.funcs]
        if self.form_b2c_mats:
            if self.vector:
                self.resampling_matrix = vector_resampling_matrix(self.bdy.N, self.fine_N)
            else:
                self.resampling_matrix = resampling_matrix(self.bdy.N, self.fine_N)
            self.b2c_mats = [mat.dot(self.resampling_matrix) for mat in self.mats]
    def __call__(self, tau_list):
        if self.form_b2c_mats:
            ucs = [mat.dot(tau) for mat, tau in zip(self.b2c_mats, tau_list)]
        else:
            ftau_list = [self.resampler(tau) for tau in tau_list]
            ucs = [mat.dot(ftau) for mat, ftau in zip(self.mats, ftau_list)]
        return np.sum(ucs, axis=0)
            
################################################################################
# Easy b2c generators for common cases

def B2C_Easy_Apply(func):
    class Easy_Apply(object):
        def __init__(self, *args):
            self.args = args
        def dot(self, x):
            return func(*self.args, x)
    return Easy_Apply

def B2C_Easy_Circulant(func, one_column, vector=False):
    if one_column:
        return B2C_Easy_Circulant_One_Column(func, vector)
    else:
        return B2C_Easy_Circulant_Full_Column(func, vector)

def B2C_Easy_Circulant_One_Column(func, vector=False):
    """
    Takes a function with signature:
    m = func(src);
        m:   for the 0-th column of the matrix
    """
    if vector:
        class Easy_Circulant(object):
            def __init__(self, src):
                self.m = func(src)
                self.N = src.N
                self.theta = src.t
            def dot(self, x):
                w = PBCM(self.m, x, self.theta)
                return vector_resample(w, self.N)
    else:
        class Easy_Circulant(object):
            def __init__(self, src):
                m = func(src)
                if len(m.shape) == 2:
                    m = m[:,0]
                self.mh = np.fft.fft(m)
            def dot(self, x):
                out = np.fft.ifft(self.mh*np.fft.fft(x))
                return out.real if x.dtype == float else out
    return Easy_Circulant

def B2C_Easy_Circulant_Full_Column(func, vector=False):
    """
    Takes a function with signature:
    m = func(src, trg);
        src: global smooth boundary
        m:   the naive evaluation matrix
    This function does not waste time generating the whole matrix
    And only generates the first column (or two columns, if vector=True)
    """
    if vector:
        class Easy_Circulant(object):
            def __init__(self, src, trg):
                self.N = trg.N
                # upsample the target point to the number of source points
                trg = GSB(c=resample(trg.c, src.N))
                ssrc = src.Generate_1pt_Circulant_Boundary()
                self.m = func(ssrc, trg)
                self.theta = src.t
            def dot(self, x):
                w = PBCM(self.m, x, self.theta)
                return vector_resample(w, self.N)
    else:
        class Easy_Circulant(object):
            def __init__(self, src, trg):
                self.N = trg.N
                # upsample the target point to the number of source points
                trg = GSB(c=resample(trg.c, src.N))
                ssrc = src.Generate_1pt_Circulant_Boundary()
                m = func(ssrc, trg)[:,0]
                self.mh = np.fft.fft(m)
            def dot(self, x):
                out = np.fft.ifft(self.mh*np.fft.fft(x))
                w = out.real if x.dtype == float else out
                # this is too fine; downsample before returning
                return resample(w, self.N)
    return Easy_Circulant

################################################################################
# Input Checking Functions (mostly to be filled in)

def _check_bdy(bdy):
    assert type(bdy) == GSB, 'bdy must be of type GlobalSmoothBoundary'
    return bdy
def _check_interior(interior):
    assert type(interior) == bool, 'interior must be a bool'
    return interior
def _check_b2c(b2c):
    assert isinstance(b2c, QFS_B2C), 'b2c must be of type B2C'
    return b2c
def _check_s2c_inverter(s2c_inverter):
    assert isinstance(s2c_inverter, QFS_Inverter), 's2c_inverter must be of type QFS_Inverter'
    return s2c_inverter
def _check_cu(cu):
    assert isinstance(cu, QFS_Check_Upsampler), 'cu must be of type QFS_Check_Upsampler'
    return cu
def _check_tol(tol):
    assert type(tol) == float, 'tol must be a float'
    assert tol > 0, 'tol must be positive'
    return tol
def _check_maximum_distance(maximum_distance):
    assert maximum_distance == None or type(maximum_distance) == float, \
        'maximum_distance must be None or float'
    if type(maximum_distance) == float:
        assert maximum_distance > 0, 'maximum_distance must be positive'
    return maximum_distance
def _check_shift_type(shift_type):
    assert type(shift_type) == int or shift_type in ['complex', 'normal', 'weighted_normal'], \
        "shift_type must be an int, 'complex', 'normal', or 'weighted_normal'"
    if type(shift_type) == int:
        assert shift_type >= 1, \
            'if shift_type is an int, it must be 1 or larger'
    return shift_type

################################################################################
# The actual QFS Class

class QFS(object):
    def __init__(self, bdy, interior, b2c, s2c_inverter, cu, tol=1e-14, *,
                    maximum_distance=None, shift_type=2,
                    source_upsample_factor=1.0, check_upsample_factor=1.0, closer_source=False):
        """
        Quadrature by Fundamental Solutions Object

        bdy: GlobalSmoothBoundary
            boundary to generate evaluator for
        interior: bool
            is this for interior or exterior evaluation?
        b2c: QFS_B2C
            boundary --> check evaluator class
        s2c_inverter: QFS_Inverter
            source --> check inversion method
        cu: QFS_Check_Upsampler
            check --> upsampled check upsampler
        tol: float
            estimated error in evaluations
            note this is not guaranteed --- it is based on Poisson eq theory,
            and is used in multiple steps, so errors can add
            but it is usually pretty accurate as a guideline
        maximum_distance: float (or None)
            maximum allowable separation between corresponding points on the 
            effective source and check curves. if you have a rapidly decaying
            greens function, it is useful to set this to ensure too much
            information isn't lost
            if this is set to None, no maximum distance is imposed
        shift_type: (int, 'complex', normal')
            How to shift the boundary to get check and effective surfaces.
            If an int, done according to the formula in paper.
                1 gives speed-weighted normal vector
                Higher values are unstable. 2 is a pretty good compromise.
            If 'complex', uses complex translation. This makes error estimates
                (see tol) most accurate, but is *extremely* unstable. Only use
                if you have a *very smooth and well resolved* boundary
            If 'normal', uses constant-distance normal translation. Very
                innacurate and mostly for testing purposes
        source_upsample_factor: how much to upsample source by, AFTER positioning it
            this upsampling is above and beyond what is needed by Laplace asymptotics
        check_upsample_factor: how much to upsample the check curve, AFTER positioning it
            this is only used in the s2c phase --- b2c still evals onto check, this
            gets upsampled, and then fed into an upsampled s2c evaluator
        """
        # check and store inputs
        self.bdy = _check_bdy(bdy)
        self.interior = _check_interior(interior)
        self.b2c = _check_b2c(b2c)
        self.cu = _check_cu(cu)
        self.s2c_inverter = _check_s2c_inverter(s2c_inverter)
        self.tol = _check_tol(tol)
        self.maximum_distance = _check_maximum_distance(maximum_distance)
        self.shift_type = _check_shift_type(shift_type)
        self.source_upsample_factor = source_upsample_factor
        self.check_upsample_factor = check_upsample_factor
        
        if closer_source:
            self.Initial_MS = np.log(self.tol) / (-2*np.pi) / self.source_upsample_factor
            self.Initial_MC = (M_EPS - self.Initial_MS) / self.source_upsample_factor
        else:
            # compute M (used for computing where check/source curves go)
            # these are the initial values
            self.Initial_MS = np.log(self.tol) / (-2*np.pi)        
            # self.Initial_MC = min(M_EPS - self.Initial_MS, self.Initial_MS)
            self.Initial_MC = M_EPS - self.Initial_MS

        # Ready Boundary --> Check Evaluator
        self._ready_b2c()
        # Get Source Curve
        self._get_source()
        # upsample source, if requested
        self._upsample_source()
        # upsample check (and boundary), if requested
        self._upsample_check()
        # Ready Source --> Check Inverter
        self._ready_s2c()

    ############################################################################
    # Public Methods

    def __call__(self, tau_list):
        """
        Get effective density on souce curve given densities on boundary
        """
        # evaluate onto the check surface
        u = self.b2c(tau_list)
        # upsample to upcheck surface
        fu = self._check_upsample(u)
        # solve for density
        return self.s2c_inverter(fu)

    def u2s(self, u):
        """
        get the density on source curve given u on the boundary
        """
        self._ready_s2b()
        # upsample u to upcheck surface
        fu = self._check_upsample(u)
        return self.s2b_inverter(fu)

    def get_normal_shift(self, M):
        return shift_bdy_normal(self.bdy, M)

    def get_b2b_matrix(self):
        """
        Generate singular bdy --> bdy matrix

        If all b2c and s2c support formation, proceeds via dense matrix operations
            (reasonably fast)
        If all b2c and s2c are circulant, generates circulant representation
            (very fast)
        Otherwise, reverts to brute force construction
            (slow)
        """
        raise NotImplementedError

    ############################################################################
    # Private methods

    def _ready_s2c(self):
        self.s2c_inverter.build(self.source, self.upcheck, 1e-15)
    def _ready_b2c(self):
        singular = self.b2c.singular
        if singular:
            self.b2c.build(self.bdy)
            self.check = self.bdy
            self.fine = self.bdy
        else:
            sign = 1 if self.interior else -1
            self.check, self.MC = self._place_shfited_bdy(self.Initial_MC, sign, False)
            OF = self.Initial_MS/np.abs(self.MC) + 1
            FN = 2*int(0.5*OF*self.bdy.N)
            self.b2c.build(self.bdy, FN, self.check)
            self.fine = self.b2c.fine
    def _ready_s2b(self):
        if not hasattr(self, 's2b_inverter'):
            self.s2b_inverter = self.s2c_inverter.shallow_copy()
            self.s2b_inverter.build(self.source, self.upbdy, 1e-15)
    def _get_shifted_bdy(self, M):
        if self.shift_type == 'complex':
            return shift_bdy_complex(self.bdy, M)
        elif self.shift_type == 'normal':
            return shift_bdy_normal(self.bdy, M)
        elif self.shift_type == 'weighted_normal':
            return shift_bdy_weighted_normal(self.bdy, M)
        else:
            return shift_bdy_int(self.bdy, M, self.shift_type)
    def _place_shfited_bdy(self, M, sign, upsample):
        # assess distance between first-try and maximum_distance
        if self.maximum_distance is not None:
            sbdy = self._get_shifted_bdy(MH)
            sep = np.abs(bdy.c-sbdy.c).max()
            if sep > self.maximum_distance:
                OS = sep / self.maximum_distance
                MH = sign*M/OS
            else:
                OS = 1.0
                MH = sign*M
        else:
            OS = 1.0
            MH = sign*M
        # okay, now we have our first guess for M and oversampling required
        # try it out, if it fails, move closer and refine
        valid = False
        while not valid:
            sbdyc = self._get_shifted_bdy(MH)
            if upsample:
                sbdyc = get_upsampled_bdyc(sbdyc, OS)
            sbdyp = polygon_from_bdyc(sbdyc)
            sbdys = speed_from_bdyc(sbdyc)
            valid = sbdyp.is_valid and sbdys.min() > self.bdy.speed.min()*0.5
            if not valid:
                OS *= 1.1
                MH /= 1.1
                if OS > 10:
                    raise Exception('Source oversampling factor > 10; QFS cannot be efficiently generated; refine boundary.')
        return GSB(c=sbdyc), MH
    def _get_source(self):
        sign = -1 if self.interior else 1
        self.source, self.MS = self._place_shfited_bdy(self.Initial_MS, sign, True)
    def _upsample_source(self):
        if self.source_upsample_factor != 1.0:
            un = 2*int(self.source.N*self.source_upsample_factor*0.5)
            self.source = GSB(c=resample(self.source.c, un))
    def _upsample_check(self):
        if self.check_upsample_factor != 1.0:
            un = 2*int(self.check.N*self.check_upsample_factor*0.5)
            self.upcheck = GSB(c=resample(self.check.c, un))
            self.upbdy = GSB(c=resample(self.bdy.c, un))
            self.cu.build(self.check.N, un)
        else:
            self.upcheck = self.check
            self.upbdy = self.bdy
            self.cu = QFS_Null_Check_Upsampler()
    def _check_upsample(self, f):
        return self.cu(f)

def l2v(f):
    err = 'density must be 1-vector with length 2*N, or 2-vector with shape [2,N]'
    sh = f.shape
    if len(sh) == 1:
        assert sh[0] % 2 == 0, err
        return f.reshape(2, sh[0]//2)
    if len(sh) == 2:
        assert sh[0] == 2, err
        return f
    else:
        raise Exception(err)
class QFS_Pressure(QFS):
    def __init__(self, bdy, interior, b2c, s2c_inverter, cu, b2p_funcs, s2p_func, 
                            tol=1e-14, *, maximum_distance=None, shift_type=2,
                            source_upsample_factor=1.0, check_upsample_factor=1.0):
        """
        QFS Evaluator for vector functions with pressure (such as Stokes)
        additional args:
            b2p_funcs: boundary to pressure targ functions
                (only naive eval, for now!)
            s2p_func: effective source to pressure targ function
                (see description for s2c_func)
            As we only need one point, these can be straight up forms!
            No need to get fancy here with Circulant or Apply functions

            b2p_funcs and s2p_func must operate on the same densities that
            b2c / s2c methods operate on
        """
        super().__init__(bdy, interior, b2c, s2c_inverter, cu, tol,
                    maximum_distance=maximum_distance, shift_type=shift_type,
                    source_upsample_factor=source_upsample_factor,
                    check_upsample_factor=check_upsample_factor)
        self.b2p_funcs = b2p_funcs
        self.s2p_func = s2p_func
        self._generate_pressure_target()
        self._generate_pressure_matrices()
    def _check_pressure_pt(self, px, py, dist):
        pt = shg.Point(px, py)
        # is it inside (our outside) of the boundary?
        check1 = self.bdy_path.contains(pt)
        if not self.interior: check1 = not check1
        # is it as far away as we need it to be?
        dx = self.bdy.x - px
        dy = self.bdy.y - py
        d = np.hypot(dx, dy).min()
        check2 = d > 0.9*dist
        return check1 and check2
    def _generate_pressure_target(self):
        sign = 1 if self.interior else -1
        bdy = self.bdy
        h = bdy.speed[0] * bdy.dt

        dist = 6*sign
        pN = bdy.N
        self.bdy_path = shg.Polygon(zip(bdy.x, bdy.y))
        pressure_pt = self._get_shifted_bdy(dist)[0]
        sep = np.abs(pressure_pt - bdy.c[0])
        check = self._check_pressure_pt(pressure_pt.real, pressure_pt.imag, sep)

        while not check:
            dist /= 2
            pN = pN * 2
            pressure_pt = self._get_shifted_bdy(dist)[0]
            sep = np.abs(pressure_pt - bdy.c[0])
            check = self._check_pressure_pt(pressure_pt.real, pressure_pt.imag, sep)
        self.pressure_x = pressure_pt.real
        self.pressure_y = pressure_pt.imag
        self.pressure_fine_bdy = GSB(c=resample(self.bdy.c, pN))
        self.pressure_targ = PointSet(x=np.array(self.pressure_x), y=np.array(self.pressure_y))
    def _generate_pressure_matrices(self):
        # get pressure fine bdy --> pressure targ mats
        self.f2p_mats = [func(self.pressure_fine_bdy, self.pressure_targ) for func in self.b2p_funcs]
        # get source --> pressure targ mats
        self.s2p_mat = self.s2p_func(self.source, self.pressure_targ)
        # get source --> boundary[0] pressure targ mat
        self.pressure_bdy_targ = PointSet(c=self.bdy.c[0])
        self.s2pb0_mat = self.s2p_func(self.source, self.pressure_bdy_targ)
    def _sigma_adjust(self, sigma, pd):
        w = l2v(sigma)
        w[0] += self.source.normal_x*pd
        w[1] += self.source.normal_y*pd
        return w.ravel()
    def __call__(self, tau_list):
        """
        Given list of densities taus (corresponding to b2c_funcs)
        Give back the density on the source curve
        """
        sigma = super().__call__(tau_list)
        if self.interior:
            # pressure from effective charges
            pe = self.s2p_mat.dot(sigma)
            # directly computed pressure
            ftaus = [vector_resample(tau, self.pressure_fine_bdy.N) for tau in tau_list]
            pts = [f2p_mat.dot(tau) for f2p_mat, tau in zip(self.f2p_mats, ftaus)]
            return self._sigma_adjust(sigma, pe-np.sum(pts))
        else:
            return sigma
    def convert_uvp(self, u, v, p):
        pt = p[0] if type(p) == np.ndarray else p
        sigma = self.u2s(np.concatenate([u, v]))
        # pressure from effective charges
        pe = self.s2pb0_mat.dot(sigma)
        return self._sigma_adjust(sigma, pe-pt)

class EmptyQFS(object):
    def __init__(self, bdy, interior, tol=1e-14, shift_type=2):
        """
        Generates just the curves for QFS, but nothing else...

        primarily to computer source and check curves:
            EmptyQFS.source
            EmptyQFS.check
        along with fine_n
            EmptyQFS.fine_n
            (this is the n that the boundary should be upsampled to in order
            to use naive evaluation onto the check surface)

        bdy: GlobalSmoothBoundary
            boundary to generate evaluator for
        interior: bool
            is this for interior or exterior evaluation?
        tol: float
            estimated error in evaluations
            note this is not guaranteed --- it is based on Poisson eq theory,
            and is used in multiple steps, so errors can add
            but it is usually pretty accurate as a guideline
        shift_type: (int, 'complex', normal')
            How to shift the boundary to get check and effective surfaces.
            If an int, done according to the formula in paper.
                1 gives speed-weighted normal vector
                Higher values are unstable. 2 is a pretty good compromise.
            If 'complex', uses complex translation. This makes error estimates
                (see tol) most accurate, but is *extremely* unstable. Only use
                if you have a *very smooth and well resolved* boundary
            If 'normal', uses constant-distance normal translation. Very
                innacurate and mostly for testing purposes
        """
        # check and store inputs
        self.bdy = bdy
        self.interior = interior
        self.tol = tol
        self.shift_type = shift_type
        
        # compute M (used for computing where check/source curves go)
        # these are the initial values
        self.Initial_MS = np.log(self.tol) / (-2*np.pi)
        self.Initial_MC = M_EPS - self.Initial_MS

        # Get Check Curve
        self._get_check()
        # Get Source Curve
        self._get_source()

    ############################################################################
    # Public Methods

    def get_normal_shift(self, M):
        return shift_bdy_normal(self.bdy, M)

    ############################################################################
    # Private methods

    def _get_shifted_bdy(self, M):
        if self.shift_type == 'complex':
            return shift_bdy_complex(self.bdy, M)
        elif self.shift_type == 'normal':
            return shift_bdy_normal(self.bdy, M)
        elif self.shift_type == 'weighted_normal':
            return shift_bdy_weighted_normal(self.bdy, M)
        else:
            return shift_bdy_int(self.bdy, M, self.shift_type)
    def _place_shfited_bdy(self, M, sign, upsample):
        # assess distance between first-try and maximum_distance
        if self.maximum_distance is not None:
            sbdy = self._get_shifted_bdy(MH)
            sep = np.abs(bdy.c-sbdy.c).max()
            if sep > self.maximum_distance:
                OS = sep / self.maximum_distance
                MH = sign*M/OS
            else:
                OS = 1.0
                MH = sign*M
        else:
            OS = 1.0
            MH = sign*M
        # okay, now we have our first guess for M and oversampling required
        # try it out, if it fails, move closer and refine
        valid = False
        while not valid:
            sbdyc = self._get_shifted_bdy(MH)
            if upsample:
                sbdyc = get_upsampled_bdyc(sbdyc, OS)
            sbdyp = polygon_from_bdyc(sbdyc)
            sbdys = speed_from_bdyc(sbdyc)
            valid = sbdyp.is_valid and sbdys.min() > self.bdy.speed.min()*0.5
            if not valid:
                OS *= 1.1
                MH /= 1.1
        return GSB(c=sbdyc), MH
    def _get_source(self):
        sign = -1 if self.interior else 1
        self.source, self.MS = self._place_shfited_bdy(self.Initial_MS, sign, True)
    def _get_check(self):
        sign = 1 if self.interior else -1
        self.check, self.MC = self._place_shfited_bdy(self.Initial_MC, sign, False)
        OF = self.Initial_MS/np.abs(self.MC) + 1
        self.fine_n = 2*int(0.5*OF*self.bdy.N)
