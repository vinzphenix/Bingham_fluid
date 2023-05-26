import matplotlib.pyplot as plt
import numpy as np

ftSz1, ftSz2, ftSz3 = 15, 13, 11

# Gauss-Legendre quadrature
xG_P1 = np.array([0.])  # integration points over [-1, 1]
wG_P1 = np.array([2.])  # weights over [-1, 1]
xG_P2 = np.array([-1. / np.sqrt(3), 1. / np.sqrt(3)])  # integration points over [-1, 1]
wG_P2 = np.array([1., 1.])  # weights over [-1, 1]

# Simpson'rule
# xG_P2 = np.array([-1., 0., 1.])  # integration points over [-1, 1]
# wG_P2 = np.array([1./3., 4/3., 1./3.])  # weights over [-1, 1]
# xG_P1, wG_P1 = xG_P2, wG_P2


PHI_P1 = [
    lambda xi: (1. - xi) * 0.5,
    lambda xi: (1. + xi) * 0.5,
]
DPHI_P1 = [
    lambda xi: 0. * xi - 0.5,
    lambda xi: 0. * xi + 0.5,
]

PHI_P2 = [
    lambda xi: xi * (xi - 1.) * 0.5,
    lambda xi: 1. - xi * xi,
    lambda xi: xi * (xi + 1.) * 0.5,
]
DPHI_P2 = [
    lambda xi: xi - 0.5,
    lambda xi: -2. * xi,
    lambda xi: xi + 0.5
]

class Simulation_1D:
    def __init__(self, params: dict):
                
        self.H = params['H']  # Half-channel width
        self.K = params['K']  # Viscosity
        self.tau_zero = params['tau_zero']  # yield stress
        self.f = params['f']  # body force (pressure gradient)
        self.save = params['save']  # Boolean
        self.plot_density = params['plot_density']
        self.dimensions = params['dimensions']

        # Reference velocity imposed by (1) pressure gradient, (2) channel width, (3) viscosity
        self.V = self.f * (self.H ** 2) / (8. * self.K)
        self.Bn = self.tau_zero * self.H / (self.K * self.V)
        self.y_zero = self.tau_zero / self.f

        self.iteration = 0
        self.degree = params['degree']
        self.n_elem = params['n_elem']

        self.idx_nodes_elem = np.r_[[
            np.arange(i * self.degree, (i+1) * self.degree + 1) for i in range(self.n_elem)
        ]]

        if self.degree == 1:
            self.n_node = self.n_elem + 1
            self.xG, self.wG = xG_P1, wG_P1
            self.PHI, self.DPHI = PHI_P1, DPHI_P1
        elif self.degree == 2:
            self.n_node = 2 * self.n_elem + 1
            self.xG, self.wG = xG_P2, wG_P2
            self.PHI, self.DPHI = PHI_P2, DPHI_P2
        else:
            raise ValueError("Element order should be 1 or 2")

        self.nG = len(self.xG)
        self.n_var = self.n_node + 2 * self.nG * self.n_elem
        # horizontal velocities --- bounds on viscosity term --- bounds on yield-stress term

        self.generate_mesh1D(params['random_seed'], params['fix_interface'])

    def generate_mesh1D(self, random_seed=-1, fix_interface=False):
        if random_seed == -1:
            y = np.linspace(-self.H / 2., self.H / 2., self.n_elem + 1)
        else:
            rng = np.random.default_rng(random_seed)  # 1 3
            dy = rng.random(self.n_elem)
            dy /= np.sum(dy)
            y = (self.H * np.r_[0., np.cumsum(dy)] - self.H / 2.)

        if fix_interface:
            idx_bot = np.argmin(np.abs(y + self.y_zero))
            idx_top = np.argmin(np.abs(y - self.y_zero))
            y[idx_bot], y[idx_top] = -self.y_zero, self.y_zero

        self.set_y(y)
        return

    def set_y(self, new_y):
        self.y = new_y
        self.dy = np.diff(self.y)
        self.ym = (self.y[:-1] + self.y[1:]) / 2.

    def set_reconstruction(self, dudy_reconstructed):
        self.dudy_reconstructed = dudy_reconstructed
