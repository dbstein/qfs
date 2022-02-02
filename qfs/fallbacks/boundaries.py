import numpy as np

def my_resample(f, n):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sp.signal.resample(f, n)
        
class PointSet(object):
    def __init__(self, x=None, y=None, c=None):
        """
        Initialize a Point Set.

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates

        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
        If both are provided the real vectors will be used
        """
        if x is not None and y is not None:
            self.shape = x.shape
            self.x = x.ravel()
            self.y = y.ravel()
            self.c = self.x + 1j*self.y
        elif c is not None:
            self.shape = c.shape
            self.c = c.ravel()
            self.x = self.c.real
            self.y = self.c.imag
        else:
            raise Exception('Not enough parameters provided to define Point Set.')
        self.N = self.x.shape[0]
    # end __init__ function definition

    def get_stacked_boundary(self, T=True):
        self.stack_boundary()
        return self.stacked_boundary_T if T else self.stacked_boundary

    def stack_boundary(self):
        if not hasattr(self, 'boundary_stacked'):
            self.stacked_boundary = np.column_stack([self.x, self.y])
            self.stacked_boundary_T = self.stacked_boundary.T
            self.boundary_stacked = True

class Global_Smooth_Boundary(PointSet):
    """
    This class impelements a "global smooth boundary" for use in
    Boundary Integral methods

    Instantiation: see documentation to self.__init__()
    """
    def __init__(self, x=None, y=None, c=None):
        """
        This function initializes the boundary element.

        x (optional): real vector of x-coordinates
        y (optional): real vector of y-coordinates
        c (optional): complex vector with c.real giving x-coordinates
            and c.imag giving y-coordinates
        The user must provide at least one of the following sets of inputs:
            (1) x and y            
                (x and y positions, as real vectors)
            (2) c
                (x and y positions as a complex vector, x=c.real, y=cimag)
        If inside_point is not provided, it will be computed as the mean
            this may not actually be inside!

        As of now, its not clear to me that everything will
        work if n is odd. For now, I will throw an error if x/y/c have an
        odd number of elements in them
        """
        super(Global_Smooth_Boundary, self).__init__(x, y, c)
        if self.N % 2 != 0:
            raise Exception('The Global_Smooth_Boundary class only accepts \
                                                                    even N.')
        self.t, self.dt = np.linspace(0, 2*np.pi, self.N, \
                                                endpoint=False, retstep=True)
        self.k = np.fft.fftfreq(self.N, self.dt/(2.0*np.pi)) # fourier modes
        self.k[int(self.N/2)] = 0.0 # wipe out nyquist frequency
        self.ik = 1j*self.k
        self.chat = np.fft.fft(self.c)
        self.cp = np.fft.ifft(self.chat*self.ik)
        self.cpp = np.fft.ifft(self.chat*self.ik**2)
        self.speed = np.abs(self.cp)
        self.tangent_c = self.cp/self.speed
        self.tangent_x = self.tangent_c.real
        self.tangent_y = self.tangent_c.imag
        self.normal_c = -1.0j*self.tangent_c
        self.normal_x = self.normal_c.real
        self.normal_y = self.normal_c.imag
        self.curvature = -(np.conj(self.cpp)*self.normal_c).real/self.speed**2
        self.weights = self.dt*self.speed
        self.complex_weights = self.dt*self.cp
        self.scaled_cp = self.cp/self.N
        self.scaled_speed = self.speed/self.N
        self.max_h = np.max(self.weights)
        self.area = self.dt*np.sum(self.x*self.cp.imag)
        self.perimeter = self.dt*np.sum(self.speed)
    def stack_normal(self):
        if not hasattr(self, 'normal_stacked'):
            self.stacked_normal = np.column_stack([self.normal_x, self.normal_y])
            self.stacked_normal_T = self.stacked_normal.T
        self.normal_stacked = True
    def get_stacked_normal(self, T=True):
        self.stack_normal()
        return self.stacked_normal_T if T else self.stacked_normal
    def generate_resampled_boundary(self, new_N):
        sfc = my_resample(self.c, new_N)
        return Global_Smooth_Boundary(c=sfc)
