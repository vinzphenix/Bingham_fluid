from bingham_1D_structure import *
from bingham_1D_post_pro import plot_reconstruction, plot_solution_1D
from cvxopt import matrix, solvers


def solve_FE(sim: Simulation_1D, atol=1e-8, rtol=1e-6):

    c = np.zeros(sim.n_var)
    I1 = sim.n_node
    I2 = I1 + sim.nG * sim.n_elem

    # COST
    for i in range(sim.n_elem):
        for g, (wg, xg) in enumerate(zip(sim.wG, sim.xG)):

            idx_nodes_elem = [i, i + 1] if sim.degree == 1 else [2 * i, 2 * i + 1, 2 * i + 2]
            for idx, phi in zip(idx_nodes_elem, sim.PHI):
                c[idx] -= wg * sim.f * sim.dy[i] / 2. * phi(xg)

            # c[2*i + 0] -= wg * sim.f * dy[i] * PHI[0](xg)
            # c[2*i + 1] -= wg * sim.f * dy[i] * PHI[1](xg)
            # c[2*i + 2] -= wg * sim.f * dy[i] * PHI[2](xg)

            c[I1 + sim.nG * i + g] += sim.K / 2. * wg * sim.dy[i] / 2.
            c[I2 + sim.nG * i + g] += sim.tau_zero * wg * sim.dy[i] / 2.

    A = np.zeros((2, sim.n_var))
    A[0, 0], A[1, sim.n_node - 1] = 1., 1.
    b = np.zeros(2)
    G = np.zeros((5 * sim.n_elem * sim.nG, sim.n_var))
    h = np.zeros(5 * sim.n_elem * sim.nG)
    idx_st = 3 * sim.nG * sim.n_elem

    # CONSTRAINTS
    for i in range(sim.n_elem):
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
    dims = {'l': 0, 'q': [3 for i in range(sim.n_elem * sim.nG)] +
            [2 for i in range(sim.n_elem * sim.nG)], 's': []}

    solvers.options['abstol'] = atol
    solvers.options['reltol'] = rtol
    res = solvers.conelp(c, G, h, dims, A, b)
    u_num = np.array(res['x'])[:I1].reshape(sim.n_node)
    s_num = np.array(res['x'])[I1:I2].reshape((sim.n_elem, sim.nG))
    t_num = np.array(res['x'])[I2:].reshape((sim.n_elem, sim.nG))
    print(f"{res['primal objective'] + 1. / 192.:.3e}")
    return u_num, s_num, t_num


def display_warning(text: str):
    print(text)
    print("The mesh may be too coarse, or too anisotropic")
    print("Process terminated")
    return


def locate_interfaces(t_num, tol):
    # Small cheat as know there are two interfaces

    mask_strain = np.mean(t_num, axis=1) > tol  # mean over the gauss points

    # Search for strained elements, neighbours of rigid elements
    idxs_switch_up, = np.where(mask_strain * ~np.r_[True, mask_strain[:-1]])
    idxs_switch_dw, = np.where(mask_strain * ~np.r_[mask_strain[+1:], True])

    # Add the direction in which elements are liquid
    idxs_switch_dw = np.c_[idxs_switch_dw, -np.ones_like(idxs_switch_dw)]
    idxs_switch_up = np.c_[idxs_switch_up, +np.ones_like(idxs_switch_up)]

    idxs_switch = np.r_[idxs_switch_dw, idxs_switch_up]

    return idxs_switch


def reconstruct(t_num, i, direction):

    n_pts = 2 * sim.nG  # ideally, take the information in two strained element
    coefs = np.empty(2)
    # y_zero_guess = np.empty(idx_elem_switch.shape[0])
    # strains = np.empty(idx_elem_switch.shape[0])
    # coefs = np.empty((idx_elem_switch.shape[0], 2))

    # for loop, (i, direction) in enumerate(idx_elem_switch):  # for each interface

    matrix, vector = np.ones((n_pts, 2)), np.empty(n_pts)
    y_interface = sim.ym[i] - direction * sim.dy[i] / 2.  # current interface node location

    # Add relation (position, strain) of the current element in the approximation matrix
    matrix[:sim.nG, 1] = sim.ym[i] + sim.xG * sim.dy[i] / 2.
    vector[:sim.nG] = t_num[i]

    # Add relation (position, strain) of the neighbour element in the approximation matrix
    if 0 <= i + direction < sim.n_elem:
        matrix[sim.nG:, 1] = sim.ym[i + direction] + sim.xG * sim.dy[i + direction] / 2.
        vector[sim.nG:] = t_num[i + direction]
    elif sim.degree == 1:
        display_warning("Cannot reconstruct linear approximation with only one value")
        return np.nan, np.nan, np.nan
    else:
        # Only possible in the case of P2 elements with 2 gauss points per element
        # Enough to generate a linear approximation (interpolation then)
        matrix = matrix[:sim.nG]
        vector = vector[:sim.nG]

    coefs = np.linalg.solve(np.dot(matrix.T, matrix), np.dot(matrix.T, vector))
    y_zero_guess = -coefs[0] / coefs[1]
    strain = np.dot(coefs, np.array([1., y_interface]))

    return y_zero_guess, strain, coefs


def display_info(sim: Simulation_1D, y_zero_guess, next_y, node_to_move):
    y_zero_guess *= 1. if sim.dimensions else 2. / sim.H
    moved_y = next_y[node_to_move] if sim.dimensions else 2. / sim.H * next_y[node_to_move]
    print(f"iter {sim.iteration:2d} : {'du/dy root':>15s} = {y_zero_guess:6.3f}", end="")
    print(f"   y[{node_to_move:d}] update = {moved_y:6.3f}")
    return


def solve_interface_tracking(sim: Simulation_1D, atol=1e-9, rtol=1e-8):

    # Set some parameters
    optimalMesh = False
    sim.iteration, max_it = 0, 20
    tol_unyielded = 1.e-5  # should be related to the tolerance of the solver
    update_coef = 1.  # 1. for full update, 0.5 for half-update, 0. for no update

    while (sim.iteration < max_it) and (not optimalMesh):  # and if sol is C1, break

        # Solve and plot the velocity / strain fields
        u_nodes, s_num, t_num = solve_FE(sim, atol=atol, rtol=rtol)
        if sim.iteration == 0:
            plot_solution_1D(
                sim, u_nodes, mini_display=False,
                extra_name="first", window="First Iteration"
            )
        else:
            plot_reconstruction(
                sim, y_old, u_old, sim.y, u_nodes,
                idxs_switch, coefs_list,
            )
        y_old, u_old = sim.y.copy(), u_nodes.copy()
        sim.iteration += 1

        # Locate the current interface on the mesh
        idxs_switch = locate_interfaces(t_num, tol_unyielded)
        if len(idxs_switch) != 2:
            display_warning("No rigid region found")
            return u_nodes

        # Reconstruct and move the nodes
        optimalMesh = True  # switched to false if strain at interface not close to 0
        coefs_list = []
        next_y = np.copy(sim.y)

        for (i, direction) in idxs_switch:

            y_zero_guess, strain_interface, coefs = reconstruct(t_num, i, direction)
            coefs_list += [coefs]

            if np.isnan(y_zero_guess):
                return u_nodes

            elif np.abs(strain_interface) > tol_unyielded:
                optimalMesh = False

                # Update new node position
                y_this_elem = next_y[[i, i + 1]]
                node_to_move = np.argmin(np.abs(y_this_elem - y_zero_guess)) + i
                next_y[node_to_move] += (y_zero_guess - next_y[node_to_move]) * update_coef

                display_info(sim, y_zero_guess, next_y, node_to_move)

        sim.set_y(next_y)

    return u_nodes


if __name__ == "__main__":

    params = dict(
        H=1., K=1., tau_zero=0.25, f=1., degree=1, n_elem=5,
        random_seed=8, fix_interface=True,
        save=False, plot_density=25, dimensions=False
    )

    sim = Simulation_1D(params)

    # y_tmp = np.array([-0.5, -0.47, -0.44, -0.25, 0.10, 0.25, 0.35, 0.5])  # used for basic plots
    y_tmp = np.array([-0.5, -0.4, -0.2, -0.02, 0.45, 0.5])  # used for basic plots
    # y_tmp = np.array([-0.5, -0.4, -0.30, 0.30, 0.4, 0.5])
    # z = 0.3335  # 0.25
    # y_tmp = np.array([-0.5, -z, 0., z, 0.5])  # -1/192 +
    sim.set_y(y_tmp)

    # Solve the problem ITERATE
    u_num = solve_interface_tracking(sim, atol=1e-12, rtol=1e-10)

    # Solve problem ONE SHOT
    # u_num, s_num, t_num = solve_FE(sim, atol=1e-12, rtol=1e-10)

    plot_solution_1D(sim, u_num, mini_display=False, extra_name="last", window="Final solution")
