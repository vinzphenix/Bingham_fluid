import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers

ftSz1, ftSz2, ftSz3 = 15, 13, 11


class Simulation_1D:
    def __init__(self, params):
        
        # H, K, tau_zero=0.2, f=1., deg=2, nElem=5, random_seed=-1,
                        # fix_interface=False, save=False, plot_density=25
        
        self.H = params['H']  # Half-channel width
        self.K = params['K']  # Viscosity
        self.tau_zero = params['tau_zero']  # yield stress
        self.f = params['f']  # body force (pressure gradient)
        self.save = params['save']  # Boolean
        self.plot_density = params['plot_density']

        # Reference velocity imposed by (1) pressure gradient, (2) channel width, (3) viscosity
        self.V = self.f * (self.H ** 2) / (8. * self.K)
        self.y0 = self.tau_zero / self.f
        self.Bn = self.tau_zero * self.H / (self.K * self.V)

        self.n_iterations = 0
        self.degree = params['degree']
        self.nElem = params['nElem']
        self.nPrimNode = self.nElem + 1

        # self.n_local_node = self.degree + 1
        self.idx_nodes_elem = np.r_[[
            np.arange(i * self.degree, (i+1) * self.degree + 1) for i in range(self.nElem)
        ]]

        if self.degree == 1:
            self.nNode = self.nElem + 1
            self.nG, self.xG, self.wG = 1, xG_P1, wG_P1
            self.PHI, self.DPHI = PHI_P1, DPHI_P1
        elif self.degree == 2:
            self.nNode = 2 * self.nElem + 1
            self.nG, self.xG, self.wG = 2, xG_P2, wG_P2
            self.PHI, self.DPHI = PHI_P2, DPHI_P2
        else:
            raise ValueError("Element order should be 1 or 2")

        
        self.nVar = self.nNode + 2 * self.nG * self.nElem
        # horizontal velocities --- bounds on viscosity term --- bounds on yield-stress term

        self.generate_mesh1D(params['random_seed'], params['fix_interface'])

    def generate_mesh1D(self, random_seed=-1, fix_interface=False):
        if random_seed == -1:
            y = np.linspace(-self.H / 2., self.H / 2., self.nElem + 1)
        else:
            rng = np.random.default_rng(random_seed)  # 1 3
            dy = rng.random(self.nElem)
            dy /= np.sum(dy)
            y = (self.H * np.r_[0., np.cumsum(dy)] - self.H / 2.)

        if fix_interface:
            idx_bot, idx_top = np.argmin(np.abs(y + self.y0)), np.argmin(np.abs(y - self.y0))
            y[idx_bot], y[idx_top] = -self.y0, self.y0

        self.set_y(y)
        return

    def set_y(self, new_y):
        self.y = new_y
        self.dy = np.diff(self.y)
        self.ym = (self.y[:-1] + self.y[1:]) / 2.

    def set_reconstruction(self, dudy_reconstructed):
        self.dudy_reconstructed = dudy_reconstructed


xG_P1 = np.array([0.])  # integration points over [-1, 1]
wG_P1 = np.array([2.])  # weights over [-1, 1]

xG_P2 = np.array([-1. / np.sqrt(3), 1. / np.sqrt(3)])  # integration points over [-1, 1]
wG_P2 = np.array([1., 1.])  # weights over [-1, 1]
# yg = ym[i] + xg * dy[i] / 2.

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


def solve_FE(sim: Simulation_1D, atol=1e-8, rtol=1e-6):
    # nNode = 2 * sim.nElem + 1
    # nG = 2  # quad shape fcts -> two gauss point needed
    # nVar = nNode + 2 * nG * sim.nElem  # velocities --- bounds on viscosity term --- bounds on yield-stress term

    c = np.zeros(sim.nVar)
    I1 = sim.nNode
    I2 = I1 + sim.nG * sim.nElem

    # COST
    for i in range(sim.nElem):
        for g, (wg, xg) in enumerate(zip(sim.wG, sim.xG)):

            idx_nodes_elem = [i, i + 1] if sim.degree == 1 else [2 * i, 2 * i + 1, 2 * i + 2]
            for idx, phi in zip(idx_nodes_elem, sim.PHI):
                c[idx] -= wg * sim.f * sim.dy[i] / 2. * phi(xg)

            # c[2*i + 0] -= wg * sim.f * dy[i] * PHI[0](xg)
            # c[2*i + 1] -= wg * sim.f * dy[i] * PHI[1](xg)
            # c[2*i + 2] -= wg * sim.f * dy[i] * PHI[2](xg)

            c[I1 + sim.nG * i + g] += sim.K / 2. * wg * sim.dy[i] / 2.
            c[I2 + sim.nG * i + g] += sim.tau_zero * wg * sim.dy[i] / 2.

    A = np.zeros((2, sim.nVar))
    A[0, 0], A[1, sim.nNode - 1] = 1., 1.
    b = np.zeros(2)
    G = np.zeros((5 * sim.nElem * sim.nG, sim.nVar))
    h = np.zeros(5 * sim.nElem * sim.nG)
    idx_st = 3 * sim.nG * sim.nElem

    # CONSTRAINTS
    for i in range(sim.nElem):
        for g, (wg, xg) in enumerate(zip(sim.wG, sim.xG)):

            # | du |^2 < t
            # ROTATED CONE : (Ub DPhib/dy + Uc DPhic/dy + Ut DPhit/dy)^2 <= 2 * sig * 0.5
            # LORENTZ CONE : hypot[(sig - 0.5)/sqrt(2), (...)] <= (sig + 0.5)/sqrt(2)
            # ((sig + 0.5)/sqrt2, (sig - 0.5)/sqrt2, (...)) in Lorentz cone
            # s1 = (sig + 0.5)/sqrt2 ; s2 = (sig - 0.5)/sqrt2 ; s3 = (...)
            # G x + s = h  with  s >= 0 (conic inequality)

            G[3 * (sim.nG * i + g) + 0, I1 + sim.nG * i + g] = -1. / np.sqrt(2.)
            h[3 * (sim.nG * i + g) + 0] = 0.5 / np.sqrt(2.)

            G[3 * (sim.nG * i + g) + 1, I1 + sim.nG * i + g] = -1. / np.sqrt(2.)
            h[3 * (sim.nG * i + g) + 1] = -0.5 / np.sqrt(2.)

            dxi_dy = 2. / sim.dy[i]
            idx_nodes_elem = [i, i + 1] if sim.degree == 1 else [2 * i, 2 * i + 1, 2 * i + 2]
            for idx, dphi in zip(idx_nodes_elem, sim.DPHI):
                G[3 * (sim.nG * i + g) + 2, idx] = -dphi(xg) * dxi_dy
            # G[3*(sim.nG*i + g) + 2, 2*i + 0] = -DPHI[0](xg) * dxi_dy
            # G[3*(sim.nG*i + g) + 2, 2*i + 1] = -DPHI[1](xg) * dxi_dy
            # G[3*(sim.nG*i + g) + 2, 2*i + 2] = -DPHI[2](xg) * dxi_dy

            # | Ub DPhib/dy + Uc DPhic/dy + Ut DPhit/dy | < t
            # (t, |...|) in Lorentz cone
            # s1 = t; s2 = |...|
            # G x + s = h  with  s >= 0 (conic inequality)

            G[idx_st + 2 * (sim.nG * i + g) + 0, I2 + sim.nG * i + g] = -1.
            for idx, dphi in zip(idx_nodes_elem, sim.DPHI):
                G[idx_st + 2 * (sim.nG * i + g) + 1, idx] = -dphi(xg) * dxi_dy
            # G[idx_st + 2*(sim.nG*i + g) + 1, 2*i + 0] = -DPHI[0](xg) * dxi_dy
            # G[idx_st + 2*(sim.nG*i + g) + 1, 2*i + 1] = -DPHI[1](xg) * dxi_dy
            # G[idx_st + 2*(sim.nG*i + g) + 1, 2*i + 2] = -DPHI[2](xg) * dxi_dy

    c, G, h, A, b = matrix(c), matrix(G), matrix(h), matrix(A), matrix(b)
    dims = {'l': 0, 'q': [3 for i in range(sim.nElem * sim.nG)] +
            [2 for i in range(sim.nElem * sim.nG)], 's': []}

    solvers.options['abstol'] = atol
    solvers.options['reltol'] = rtol
    res = solvers.conelp(c, G, h, dims, A, b)
    u_num = np.array(res['x'])[:I1].reshape(sim.nNode)
    s_num = np.array(res['x'])[I1:I2].reshape((sim.nElem, sim.nG))
    t_num = np.array(res['x'])[I2:].reshape((sim.nElem, sim.nG))
    return u_num, s_num, t_num


def display_warning(text: str):
    print(text)
    print("The mesh may be too coarse, or too anisotropic")
    print("Process terminated")
    return


def solve_interface_tracking(sim: Simulation_1D, atol=1e-8, rtol=1e-6):

    def eval_dudy(xi, u_local, dy_local):
        
        dudy = np.zeros_like(xi)
        for u_j, dphi_j in zip(u_local, sim.DPHI):
            dudy[:] += u_j * dphi_j(xi)
        return dudy * 2. / dy_local

    def compute_strains(u, use_abs=False):  # compute the average value of |D| over each interval
        
        func = (lambda a: np.abs(a)) if use_abs else (lambda a: a)
        strains = np.zeros(sim.nElem)
        for i in range(sim.nElem):
            idx_nodes_elem = [i, i + 1] if sim.degree == 1 else [2 * i, 2 * i + 1, 2 * i + 2]
            du_gauss_pt = eval_dudy(sim.xG, u[idx_nodes_elem], sim.dy[i])
            strains[i] = np.dot(sim.wG / 2., func(du_gauss_pt))
        return strains

    def compute_new_targets(idx_elem_switch):
        
        n_pts = 2 * sim.nG  # ideally, take the information in two strained element
        y_zero_guesses = np.empty(idx_elem_switch.shape[0])
        strains = np.empty(idx_elem_switch.shape[0])

        for loop, (i, direction) in enumerate(idx_elem_switch):  # for each interface

            matrix, vector = np.ones((n_pts, 2)), np.empty(n_pts)
            y_interface = sim.ym[i] - direction * sim.dy[i] / 2.

            # Add relation (position, strain) of the current element in the approximation matrix
            matrix[:sim.nG, 1] = sim.ym[i] + sim.xG * sim.dy[i] / 2.
            vector[:sim.nG] = t_num[i]
                        
            # Add relation (position, strain) of the neighbour element in the approximation matrix
            if 0 <= i + direction < sim.nElem:
                matrix[sim.nG:, 1] = sim.ym[i + direction] + sim.xG * sim.dy[i + direction] / 2.
                vector[sim.nG:] = t_num[i + direction]
            elif sim.degree == 1:
                print(i, direction)
                display_warning("Cannot reconstruct linear approximation with only one value")
                return np.nan * y_zero_guesses
            else:
                # cut the system with its current values as no other available
                matrix = matrix[:sim.nG] 
                vector = vector[:sim.nG]

            coefs = np.linalg.solve(np.dot(matrix.T, matrix), np.dot(matrix.T, vector))
            y_zero_guesses[loop] = -coefs[0] / coefs[1]
            strains[loop] = np.dot(coefs, np.array([1., y_interface]))

        return y_zero_guesses, strains

    # def check_sol_C1(u, idx_elem_switch, tol):  # verify if |D| is continuous at interface (|D| = 0)

    #     # for i in range(1, sim.nPrimNode - 1):
    #     #     u_local_lo = u_nodes[sim.idx_nodes_elem[i - 1]]
    #     #     u_local_up = u_nodes[sim.idx_nodes_elem[i]]
    #     #     strain_lo = eval_dudy(np.array([+1.]), u_local_lo, sim.dy[i - 1])
    #     #     strain_up = eval_dudy(np.array([-1.]), u_local_up, sim.dy[i])
    #     #     print(f"delta strain [{i:d}]= {np.abs(strain_lo - strain_up)[0]:.3f}")
    #     #     if np.abs(strain_lo - strain_up) > tol:
    #     #         return False

    #     return True

    # Set some parameters
    sim.it, max_it = 0, 20
    tol_unyielded = 1.e-3
    update_coef = 1.  # 1. for full update, 0.5 for half-update, 0. for no update

    # Solve first time with initial mesh
    u_nodes, s_num, t_num = solve_FE(sim, atol=atol, rtol=rtol)
    plot_solution_1D(sim, u_nodes)

    while sim.it < max_it:  # and if sol is C1, break

        print("")
        mask_strain = np.mean(t_num, axis=1) > tol_unyielded  # mean over the gauss points
        mask_unstrain_lo = ~np.r_[True, mask_strain[:-1]]
        mask_unstrain_up = ~np.r_[mask_strain[1:], True]

        # Search for strained elements, neighbours of rigid elements
        idxs_switch_up, = np.where(mask_strain * ~np.r_[True, mask_strain[:-1]])
        idxs_switch_dw, = np.where(mask_strain * ~np.r_[mask_strain[+1:], True])
        idxs_switch_dw = np.c_[idxs_switch_dw, -np.ones_like(idxs_switch_dw)]
        idxs_switch_up = np.c_[idxs_switch_up, +np.ones_like(idxs_switch_up)]
        idxs_switch = np.r_[idxs_switch_dw, idxs_switch_up]
        if len(idxs_switch) == 0:
            display_warning("No rigid region found")
            return u_nodes

        y_zero_guesses, strain_interfaces = compute_new_targets(idxs_switch)

        # if check_sol_C1(u_nodes, idxs_switch, tol=1e-6):
        if np.all(np.abs(strain_interfaces) < 1e-6):
            break

        if np.any(np.isnan(y_zero_guesses)):
            return u_nodes
        
        next_y = np.copy(sim.y)
        
        for (i, direction), y_zero_guess in zip(idxs_switch, y_zero_guesses):

            # Update new node position
            y_this_elem = next_y[sim.idx_nodes_elem[i]]
            node_to_move = np.argmin(np.abs(y_this_elem - y_zero_guess)) + sim.idx_nodes_elem[i, 0]
            next_y[node_to_move] += (y_zero_guess - next_y[node_to_move]) * update_coef
            
            print(direction == 1)
            print(f"iteration {sim.it:2d} : {'root of du/dy':>15s} = {y_zero_guess:6.3f}", end="")
            print(f"   y0 update = {next_y[node_to_move]:6.3f}")
            
        plot_solution_1D(sim, u_nodes)
        print("")

        sim.set_y(next_y)
        u_nodes, s_num, t_num = solve_FE(sim, atol=1e-12, rtol=1e-10)
        sim.it += 1

    sim.n_iterations = sim.it
    return u_nodes


def plot_solution_1D(sim: Simulation_1D, u_nodes):
    H, K, tau_zero, f, u_inf = sim.H, sim.K, sim.tau_zero, sim.f, sim.V
    y0, nElem, y = sim.y0, sim.nElem, sim.y
    hh = H / 2.
    pts_per_elem = sim.plot_density + 1

    u_inf, hh = 1., 1.

    def get_analytical_sol(y_eval):
        e0, eta, u_ana = 2. * y0 / H, 2. * y_eval / H, np.zeros(len(y_eval))
        m_bot = (-1. <= eta) & (eta <= -e0)
        m_mid = (-e0 - 1e-6 <= eta) & (eta <= e0)
        m_top = (e0 - 1e-6 <= eta) & (eta <= 1.)
        u_ana[m_top] = -sim.Bn / 2. * (1. - eta[m_top]) + (1. - np.square(eta[m_top]))
        u_ana[m_bot] = -sim.Bn / 2. * (1. + eta[m_bot]) + (1. - np.square(eta[m_bot]))
        u_ana[m_mid] = (1. - sim.Bn / 4.) ** 2
        return sim.V * u_ana

    # Set coordinate arrays
    u_ext_nodes = u_nodes if sim.degree == 1 else u_nodes[::2]
    u_mid_nodes = np.array([]) if sim.degree == 1 else u_nodes[1::2]

    u_vertex = np.dstack((u_ext_nodes[:-1], u_ext_nodes[1:])).flatten()
    y_vertex = np.dstack((y[:-1], y[1:])).flatten()
    y_middle = (y[:-1] + y[1:]) / 2.
    y_dense = np.empty(pts_per_elem * nElem)
    # y_nodes = np.empty(2 * nElem + 1)
    # y_nodes[::2], y_nodes[1::2] = y, (y[:-1] + y[1:]) / 2.

    xi_vertex = np.array([-1., 1.])
    this_xi = np.linspace(-1., 1., pts_per_elem, endpoint=True)

    # Compute numerical solutions
    u_num, du_num = np.zeros_like(y_dense), np.zeros_like(y_dense)
    du_vertex, du_middle = np.zeros(2 * sim.nElem), np.zeros(sim.nElem)
    dy = np.diff(y)
    for i in range(nElem):
        indices_this_elem = range(i * pts_per_elem, (i + 1) * pts_per_elem)
        y_dense[indices_this_elem] = np.linspace(y[i], y[i + 1], pts_per_elem, endpoint=True)
        idx_nodes_elem = [i, i + 1] if sim.degree == 1 else [2 * i, 2 * i + 1, 2 * i + 2]
        for idx, phi, dphi in zip(idx_nodes_elem, sim.PHI, sim.DPHI):
            u_num[indices_this_elem] += u_nodes[idx] * phi(this_xi)
            du_num[indices_this_elem] += 2. / dy[i] * u_nodes[idx] * dphi(this_xi)
            du_vertex[[2 * i, 2 * i + 1]] += 2. / dy[i] * u_nodes[idx] * dphi(xi_vertex)
            du_middle[i] += 2. / dy[i] * u_nodes[idx] * dphi(0.)
    du_num[-1] = du_vertex[-1]

    if tau_zero > 0.:
        tau_xy_num = 1 + K * np.abs(du_num) / tau_zero
        tau_xy_num[(-y0 < y_dense) & (y_dense < y0)] = np.nan
        tau_xy_vertex = 1 + K * np.abs(du_vertex) / tau_zero
        tau_xy_vertex[(-y0 < y_vertex) & (y_vertex < y0)] = np.nan
        tau_xy_middle = 1 + K * np.abs(du_middle) / tau_zero
        tau_xy_middle[(-y0 < y_middle) & (y_middle < y0)] = np.nan
        label_tau_ref = r"\tau_0"
    else:
        tau_xy_num = K * np.abs(du_num) / (H * f)
        tau_xy_vertex = K * np.abs(du_vertex) / (H * f)
        tau_xy_middle = K * np.abs(du_middle) / (H * f)
        label_tau_ref = r"h \partial_x p"

    # Compute analytic solutions
    u_ana, u_ana_vertex = get_analytical_sol(y_dense), get_analytical_sol(y_vertex)
    du_ana = (tau_zero - y_dense * f) / K
    du_ana[y_dense < y0] = 0.
    du_ana[y_dense < -y0] = (-tau_zero - y_dense[y_dense < -y0] * f) / K
    du_ana_middle = (tau_zero - y_middle * f) / K
    du_ana_middle[y_middle < y0] = 0.
    du_ana_middle[y_middle < -y0] = (-tau_zero - y_middle[y_middle < -y0] * f) / K
    tau_ref = tau_zero if sim.tau_zero > 0. else (H * f)
    tau_xy_ana = np.abs(f * y_dense) / tau_ref
    tau_xy_ana_middle = np.abs(f * y_middle) / tau_ref

    # FIGURE
    plt.rcParams["text.usetex"] = sim.save
    lw, alp = 5., 0.5  # setting to display analytical solution
    al = 0.3  # setting for the error plot
    # figsize = (9.5, 5.75)
    figsize = (12., 8.)
    fig, axs = plt.subplots(2, 3, figsize=figsize, sharey="all")

    ax = axs[0, 0]
    ax.set_xlabel(r"$u(y) / U_{\infty}$", fontsize=ftSz2)
    ax.set_title("Velocity profile", fontsize=ftSz1)
    ax.set_ylabel(r"$2y/h$", fontsize=ftSz2)
    ax.plot(u_ana / u_inf, y_dense / hh, ls='-', color='C0', alpha=alp, lw=lw, label="Analytical")
    ax.plot(u_ext_nodes / u_inf, y / hh, marker="o", ls="", color='C1')
    if len(u_mid_nodes) > 0:  # second order segment
        ax.plot(u_mid_nodes / u_inf, y_middle / hh, marker=".", ls="", color='C1')
    ax.plot([], [], color='C1', ls='-', marker='o', label="Numerical")
    ax.plot(u_num / u_inf, y_dense / hh, color='C1')

    ax = axs[0, 1]
    ax.set_xlabel(r"$h / U_{\infty} \partial_y u $", fontsize=ftSz2)
    ax.set_title("Strain rate profile", fontsize=ftSz1)
    ax.plot(du_ana / u_inf, y_dense / hh, label="Analytical", color="C0", alpha=alp, lw=lw)
    ax.plot(du_vertex / u_inf, y_vertex / hh, ls='', marker='o', color='C1')
    ax.plot([], [], color='C1', ls='-', marker='o', label="Numerical")
    pos = pts_per_elem * np.arange(1, sim.nElem)
    du_disc, y_disc = np.insert(du_num, pos, np.nan), np.insert(y_dense, pos, np.nan)
    ax.plot(du_disc / u_inf, y_disc / hh, marker='', ms=2, color='C1')
    # ax.plot(du_num, y_dense / H, marker='', ms=2, color='C1')

    ax = axs[0, 2]
    ax.set_xlabel(r"$|\tau_{{xy}}| \:/\: {:s}$".format(label_tau_ref), fontsize=ftSz2)
    ax.set_title("Shear stress profile", fontsize=ftSz1)
    ax.plot(tau_xy_ana, y_dense / hh, label="Analytical", color="C0", alpha=alp, lw=lw)
    ax.plot(tau_xy_vertex, y_vertex / hh, ls='', marker='o', color='C1')
    ax.plot([], [], color='C1', ls='-', marker='o', label="Numerical")
    # pos = np.where(np.abs(np.diff(du_num)) >= 5 * np.mean(np.abs(np.diff(du_num))))[0] + 1
    # ax.plot(np.insert(tau_xy_num, pos, np.nan), np.insert(y_dense, pos, np.nan) / H, marker='', ms=2, color='C1')
    ax.plot(tau_xy_num, y_dense / hh, marker='', ms=2, color='C1')

    ax = axs[1, 0]
    # ax.set_title(r"Velocity error", fontsize=ftSz1)
    rel_error_nodes = (u_vertex - u_ana_vertex) / np.amax(np.abs(u_ana_vertex))
    rel_error_dense = (u_num - u_ana) / np.amax(np.abs(u_ana))
    ax.set_xlabel(r"$(u^h - u) / u_{max}$", fontsize=ftSz2)
    ax.set_ylabel(r"$2y/h$", fontsize=ftSz2)
    ax.plot(rel_error_nodes, y_vertex / hh, marker="x", ls="", label="Error", color='red')
    error_line, y_line = (rel_error_nodes, y_vertex) if sim.degree == 1 else (
        rel_error_dense, y_dense)
    ax.plot(error_line, y_line / hh, ls="-", color='red', alpha=al)

    ax = axs[1, 1]
    # ax.set_title("Strain rate error", fontsize=ftSz1)
    rel_error_middle = (du_middle - du_ana_middle) / np.amax(np.abs(du_ana_middle))
    rel_error_dense = (du_num - du_ana) / np.amax(np.abs(du_ana))
    ax.set_xlabel(r"$(\partial_y u^h - \partial_y u) / (\partial_y u)_{max}$", fontsize=ftSz2)
    ax.plot(rel_error_middle, y_middle / hh, marker="x", ls="", label="Error", color='red')
    error_line, y_line = (rel_error_middle, y_middle) if sim.degree == 1 else (
        rel_error_dense, y_dense)
    ax.plot(error_line, y_line / hh, ls="-", color='red', alpha=al)

    ax = axs[1, 2]
    # ax.set_title("Shear stress error", fontsize=ftSz1)
    rel_error_middle = (tau_xy_middle - tau_xy_ana_middle) / np.amax(np.abs(tau_xy_ana_middle))
    rel_error_dense = (tau_xy_num - tau_xy_ana) / np.amax(np.abs(tau_xy_ana))
    ax.set_xlabel(r"$(\tau^h - \tau) / \tau_{max}$", fontsize=ftSz2)
    ax.plot(rel_error_middle, y_middle / hh, marker="x", ls="", label="Error", color='red')
    error_line, y_line = (rel_error_middle, y_middle) if sim.degree == 1 else (
        rel_error_dense, y_dense)
    ax.plot(error_line, y_line / hh, ls="-", color='red', alpha=al)

    for ax in axs[1, :]:
        ax.vlines(0., -H / 2. / hh, H / 2. / hh, color='black', alpha=0.5)
    for ax in axs.flatten():
        xmin, xmax = ax.get_xbound()
        ax.fill_between([xmin, xmax], [-y0 / hh, -y0 / hh],
                        [y0 / hh, y0 / hh], color='grey', alpha=0.25)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=ftSz3)
        ax.grid(ls=':')

    fig.tight_layout()
    if sim.save:
        filename = f"res_P{sim.degree:d}_iteration_{sim.it:02d}"
        fig.savefig(f"./figures/{filename:s}.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":

    params = dict(H=1., K=1., tau_zero=0.2, f=1., degree=1, nElem=6,
                  random_seed=-1, fix_interface=False, save=False, plot_density=25)

    sim = Simulation_1D(params)

    # Solve the problem ITERATE
    u_nodes = solve_interface_tracking(sim, atol=1e-12, rtol=1e-10)

    # Solve problem ONE SHOT
    # u_nodes, s_num, t_num = solve_FE(sim, atol=1e-12, rtol=1e-10)

    plot_solution_1D(sim, u_nodes)
