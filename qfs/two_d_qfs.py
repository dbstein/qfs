import numpy as np
import scipy as sp
import scipy.signal
import warnings
import shapely.geometry as shg
import numexpr as ne

import pybie2d
GSB = pybie2d.boundaries.global_smooth_boundary.global_smooth_boundary.Global_Smooth_Boundary

M_EPS = np.log(np.finfo(float).eps) / (-2*np.pi)

def even_it(n):
    return 2*int(np.ceil(0.5*n))
def resample(f, n):
    if n == len(f):
        out = f
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = sp.signal.resample(f, n)
    return out
def vector_resample(f, n):
    n1 = f.size // 2
    r1 = resample(f[:n1], n)
    r2 = resample(f[n1:], n)
    return np.concatenate([r1, r2])
def resample_matrix(n1, n2):
    return resample(np.eye(n1), n2)
def vector_resampling_matrix(n1, n2):
    rs = resample_matrix(n1, n2)
    mat = np.zeros([2*n2, 2*n1], dtype=float)
    mat[0*n2:1*n2, 0*n1:1*n1] = rs
    mat[1*n2:2*n2, 1*n1:2*n1] = rs
    return mat
def pressure_resampling_matrix(n1, n2):
    rs = resample_matrix(n1, n2)
    mat = np.zeros([2*n2+1, 2*n1+1], dtype=float)
    mat[0*n2:1*n2, 0*n1:1*n1] = rs
    mat[1*n2:2*n2, 1*n1:2*n1] = rs
    mat[-1, -1] = 1.0
    return mat
def polygon_from_boundary(bdy):
    return shg.Polygon(zip(bdy.x, bdy.y))
def build_error_estimate(bdy, alpha, scaleit=True):
    if scaleit: alpha = alpha*bdy.dt
    return np.fft.ifft(np.fft.fft(bdy.c)*np.exp(-alpha*bdy.k))
def sign_from_side(interior):
    return 1 if interior else -1

def get_fbdy_oversampling(bdy, M, MC, FF=0.0, eps=1e-10):
    OF = (M+FF)/MC + 1
    return even_it(bdy.N*OF) if OF > 1 else bdy.N
def generate_source_boundary(bdy, interior=True, M=4, fsuf=None):
    sign = -sign_from_side(interior)
    this_M = sign*M
    OS = 1.0
    valid = False
    this_M_orig = this_M
    if fsuf != None:
        this_M /= fsuf
        OS *= fsuf
    while not valid:
        ebdy1 = build_error_estimate(bdy, this_M)
        this_N = even_it(bdy.N*OS)
        ebdy = GSB(c=resample(ebdy1, this_N))
        valid = polygon_from_boundary(ebdy).is_valid and ebdy.speed.min() > bdy.speed.min()*0.5
        if not valid:
            this_M /= 1.1
            OS *= 1.1
    return ebdy, this_N, this_M
def generate_check_boundary(bdy, interior=True, MC=2):
    sign = sign_from_side(interior)
    valid = False
    this_MC = sign*MC
    while not valid:
        cbdy = GSB(c=build_error_estimate(bdy, this_MC))
        valid = polygon_from_boundary(cbdy).is_valid and cbdy.speed.min() > bdy.speed.min()*0.5
        if not valid:
            this_MC /= 1.1
    return cbdy, this_MC

class QFS_Boundary(object):
    """
    Container class for Quadrature by Fundamental Solutions
    """
    def __init__(self, bdy, eps=1e-12, FF=0, forced_source_upsampling_factor=None):
        """
        Initialize QFS object

        eps:       target goal for integrals
        FF: fudge factor- how much to up targets by
        """
        self.bdy = bdy
        self.eps = eps
        self.FF = FF
        self.raw_M = np.log(self.eps) / (-2*np.pi)
        self.MS = self.raw_M + self.FF
        self.MC = min(M_EPS - self.MS, self.MS)
        self.forced_source_upsampling_factor = forced_source_upsampling_factor
        # generate the source boundaries
        sbdy = generate_source_boundary(self.bdy, True, self.MS, forced_source_upsampling_factor)
        self.interior_source_bdy = sbdy[0]
        self.interior_source_N = sbdy[1]
        self.interior_source_M = sbdy[2]
        sbdy = generate_source_boundary(self.bdy, False, self.MS, forced_source_upsampling_factor)
        self.exterior_source_bdy = sbdy[0]
        self.exterior_source_N = sbdy[1]
        self.exterior_source_M = sbdy[2]
        # generate the check boundary
        cbdy = generate_check_boundary(self.bdy, True, self.MC)
        self.interior_check_bdy = cbdy[0]
        self.interior_check_MC = cbdy[1]
        cbdy = generate_check_boundary(self.bdy, False, self.MC)
        self.exterior_check_bdy = cbdy[0]
        self.exterior_check_MC = cbdy[1]
        # generate the fine boundary
        self.interior_fine_N = get_fbdy_oversampling(bdy, self.MS, np.abs(self.interior_check_MC), self.FF, self.eps)
        self.interior_fine_bdy = GSB(c=resample(bdy.c, self.interior_fine_N))
        self.exterior_fine_N = get_fbdy_oversampling(bdy, self.MS, np.abs(self.exterior_check_MC), self.FF, self.eps)
        self.exterior_fine_bdy = GSB(c=resample(bdy.c, self.exterior_fine_N))
    def get_bdys(self, interior):
        if interior:
            return self.bdy, self.interior_fine_bdy, self.interior_source_bdy, self.interior_check_bdy
        else:
            return self.bdy, self.exterior_fine_bdy, self.exterior_source_bdy, self.exterior_check_bdy
    def get_shell(self, alpha, interior, N=None):
        sign = sign_from_side(interior)
        bdy1 = build_error_estimate(self.bdy, alpha*sign, scaleit=False)
        if N is not None:
            bdy1 = resample(bdy1, N)
        return GSB(c=bdy1)

    def plot(self, ax, interior):
        wrap = lambda b: np.pad(b, (0,1), mode='wrap')
        bdy, fbdy, sbdy, cbdy = self.get_bdys(interior)
        ax.plot(wrap(bdy.x), wrap(bdy.y), color='black', linewidth=3)
        ax.plot(wrap(sbdy.x), wrap(sbdy.y), color='red', linewidth=3)
        ax.plot(wrap(cbdy.x), wrap(cbdy.y), color='blue', linewidth=3)
        # get error estimates
        e1 = build_error_estimate(fbdy, self.raw_M*sign_from_side(interior))
        e2 = build_error_estimate(sbdy, self.raw_M*sign_from_side(interior))
        ax.plot(wrap(e1.real), wrap(e1.imag), color='gray', linewidth=1)
        ax.plot(wrap(e2.real), wrap(e2.imag), color='yellow', linewidth=1)

class QFS_Evaluator(object):
    def __init__(self, qfs_bdy, interior, b2c_funcs, s2c_func, on_surface=False, form_b2c=False, vector=False):
        """
        Generate a function that takes on-surface density(s) --> source effective density
        interior: whether to construct evaluators for interior points (or exterior)
        b2c_funcs: list of functions to construct mats used for bdy --> check
            of the form MAT = bdy_eval(src, trg)
        s2c_func: function to construct mat for source --> check
            of the form MAT = source_eval(src, trg)
        on_surface: are b2c_funcs actually b2b operators?
        vector: is this for a vector density, e.g. Stokes?
        """
        self.qfs_bdy = qfs_bdy
        self.interior = interior
        self.b2c_funcs = b2c_funcs
        self.s2c_func = s2c_func
        self.on_surface = on_surface
        self.form_b2c = form_b2c
        self.vector = vector

        self.resample_matrix = vector_resampling_matrix if self.vector else resample_matrix
        self.resample = vector_resample if self.vector else resample

        self.bdy, self.fine_bdy, self.source_bdy, self.check_bdy = self.qfs_bdy.get_bdys(self.interior)

        # set things up for on-surface if that's the case
        if self.on_surface:
            self.fine_bdy = self.bdy
            self.check_bdy = self.bdy

        # get bdy --> check mats
        self.f2c_mats = [func(self.fine_bdy, self.check_bdy) for func in self.b2c_funcs]
        # form b2c
        if self.form_b2c:
            self.b2f_upsampler = self.resample_matrix(self.bdy.N, self.fine_bdy.N)
            self.b2c_mats = [f2c_mat.dot(self.b2f_upsampler) for f2c_mat in self.f2c_mats]

        # get source --> check mats
        self.s2c_mat = self.s2c_func(self.source_bdy, self.check_bdy)

        # get the LU decomposition of the source --> check matrix
        self.b2s_upsampler = self.resample_matrix(self.bdy.N, self.source_bdy.N)
        self.square_source_mat = self.s2c_mat.dot(self.b2s_upsampler)
        self.source_lu = sp.linalg.lu_factor(self.square_source_mat)

    def __call__(self, taus):
        """
        Given list of densities taus (corresponding to b2c_funcs)
        Give back the density on the source curve
        """
        return self.cu2s(self._integrate_to_check(taus))

    def bu2s(self, u):
        """
        get the density on source curve given u on the boundary
        """
        if not hasattr(self, 'bu2s_lu'):
            if self.on_surface:
                self.bu2s_lu = self.source_lu
            else:
                mat = self.s2c_func(self.source_bdy, self.bdy)
                smat = mat.dot(self.b2s_upsampler)
                self.bu2s_lu = sp.linalg.lu_factor(smat)
        w1 = sp.linalg.lu_solve(self.bu2s_lu, u)
        return self.resample(w1, self.source_bdy.N)    

    def cu2s(self, u):
        """
        get the density on source curve given u on the check curve
        """
        w1 = sp.linalg.lu_solve(self.source_lu, u)
        return self.resample(w1, self.source_bdy.N)    

    def u2s(self, u):
        """
        get the density on source curve given u on the check curve
        """
        return self.cu2s(u)

    def _integrate_to_check(self, taus):
        """
        Given list of densities taus (corresponding to b2c_funcs)
        Compute integral on check surface
        This function is for diagonstic purposes
        """
        if self.form_b2c:
            ucs = [mat.dot(tau) for mat, tau in zip(self.b2c_mats, taus)]
        else:
            ftaus = [self.resample(tau, self.fine_bdy.N) for tau in taus]
            ucs = [mat.dot(ftau) for mat, ftau in zip(self.f2c_mats, ftaus)]
        uc = np.sum(ucs, axis=0)
        return uc

    def initialize_preconditioner(self, s2b_func=None):
        """
        Note that this function is only sensible if all b2c funcs get fed the same thing
        """
        self.s2b_func = s2b_func if s2b_func is not None else self.s2c_func
        self.s2b_mat = self.s2b_func(self.source_bdy, self.bdy)
        if self.form_b2c:
            self.b2c_mat = np.sum(self.b2c_mats, axis=0)
            self.partial_mat = self.b2s_upsampler.dot(sp.linalg.lu_solve(self.source_lu, self.b2c_mat))
        else:
            self.f2c_mat = np.sum(self.f2c_mats, axis=0)
            self.b2f_upsampler = self.resample_matrix(self.bdy.N, self.fine_bdy.N)
            self.partial_mat = self.b2s_upsampler.dot(sp.linalg.lu_solve(self.source_lu, self.f2c_mat.dot(self.b2f_upsampler)))
        self.full_mat = self.s2b_mat.dot(self.partial_mat)
        self.full_mat_inv = np.linalg.inv(self.full_mat)
    def precondition(self, f):
        return self.full_mat_inv.dot(f)


class QFS_Evaluator_Pressure(object):
    def __init__(self, qfs_bdy, interior, b2c_funcs, s2c_func, form_b2c=False):
        """
        Generate a function that takes on-surface density(s) --> source effective density
        Handles the pressure null-space correctly

        interior: whether to construct evaluators for interior points (or exterior)
        b2c_funcs: list of functions to construct mats used for bdy --> check
            of the form MAT = bdy_eval(src, trg)
        s2c_func: function to construct mat for source --> check
            of the form MAT = source_eval(src, trg)
        
        THIS IS ASSUMED TO BE FOR A VECTOR PROBLEM
        DOES NOT SUPPORT ON-SURFACE EVALUTION
        """
        self.qfs_bdy = qfs_bdy
        self.interior = interior
        self.b2c_funcs = b2c_funcs
        self.s2c_func = s2c_func
        self.form_b2c = form_b2c

        self.resample_matrix = vector_resampling_matrix
        self.resample = vector_resample

        self.bdy, self.fine_bdy, self.source_bdy, self.check_bdy = self.qfs_bdy.get_bdys(self.interior)

        # get bdy --> check mats
        self.f2c_mats = [func(self.fine_bdy, self.check_bdy) for func in self.b2c_funcs]
        # form b2c
        if self.form_b2c:
            self.b2f_upsampler = self.resample_matrix(self.bdy.N, self.fine_bdy.N)
            self.b2c_mats = [f2c_mat.dot(self.b2f_upsampler) for f2c_mat in self.f2c_mats]

        # get source --> check mats
        self.s2c_mat = self.s2c_func(self.source_bdy, self.check_bdy)

        # get the LU decomposition of the source --> check matrix
        self.b2s_upsampler = pressure_resampling_matrix(self.bdy.N, self.source_bdy.N)
        self.square_source_mat = self.s2c_mat.dot(self.b2s_upsampler)
        self.source_lu = sp.linalg.lu_factor(self.square_source_mat)

    def __call__(self, taus):
        """
        Given list of densities taus (corresponding to b2c_funcs)
        Give back the density on the source curve
        """
        return self.cu2s(self._integrate_to_check(taus))

    def bu2s(self, u):
        """
        get the density on source curve given u on the boundary
        """
        if not hasattr(self, 'bu2s_lu'):
            mat = self.s2c_func(self.source_bdy, self.bdy)
            smat = mat.dot(self.b2s_upsampler)
            self.bu2s_lu = sp.linalg.lu_factor(smat)
        w1 = sp.linalg.lu_solve(self.bu2s_lu, u)
        return self.resample(w1[:-1], self.source_bdy.N)

    def cu2s(self, u):
        """
        get the density on source curve given u/p on the check curve
        """
        w1 = sp.linalg.lu_solve(self.source_lu, u)
        return self.resample(w1[:-1], self.source_bdy.N)

    def u2s(self, u):
        """
        get the density on source curve given u on the check curve
        """
        return self.cu2s(u)

    def _integrate_to_check(self, taus):
        """
        Given list of densities taus (corresponding to b2c_funcs)
        Compute integral on check surface
        This function is for diagonstic purposes
        """
        if self.form_b2c:
            ucs = [mat.dot(tau) for mat, tau in zip(self.b2c_mats, taus)]
        else:
            ftaus = [self.resample(tau, self.fine_bdy.N) for tau in taus]
            ucs = [mat.dot(ftau) for mat, ftau in zip(self.f2c_mats, ftaus)]
        uc = np.sum(ucs, axis=0)
        return uc

    def initialize_preconditioner(self, s2b_func=None):
        """
        Note that this function is only sensible if all b2c funcs get fed the same thing
        """
        self.s2b_func = s2b_func if s2b_func is not None else self.s2c_func
        self.s2b_mat = self.s2b_func(self.source_bdy, self.bdy)
        if self.form_b2c:
            self.b2c_mat = np.sum(self.b2c_mats, axis=0)
            self.partial_mat = self.b2s_upsampler.dot(sp.linalg.lu_solve(self.source_lu, self.b2c_mat))
        else:
            self.f2c_mat = np.sum(self.f2c_mats, axis=0)
            self.b2f_upsampler = self.resample_matrix(self.bdy.N, self.fine_bdy.N)
            self.partial_mat = self.b2s_upsampler.dot(sp.linalg.lu_solve(self.source_lu, self.f2c_mat.dot(self.b2f_upsampler)))
        self.full_mat = self.s2b_mat.dot(self.partial_mat)
        self.full_mat_inv = np.linalg.inv(self.full_mat)
    def precondition(self, f):
        return self.full_mat_inv.dot(f)



