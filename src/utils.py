from bingham_structure import *


def get_analytical_poiseuille(sim: Simulation_2D):

    # dp_dx = sim.f[0]
    dp_dx = 1.
    H = np.amax(sim.coords[:, 1])  # half channel width (centered at y = 0)
    U_inf = dp_dx * (H ** 2) / (2. * sim.K)  # max velocity when tau_zero = 0

    Bn = sim.tau_zero * H / (sim.K * U_inf)

    y_zero = sim.tau_zero / dp_dx
    eta_zero = y_zero / H
    eta = sim.coords[:, 1] / H
    u_analytical = np.zeros_like(eta)

    mask_top = (eta_zero < eta) & (eta <= 1.)
    mask_bot = (-1. <= eta) & (eta < -eta_zero)
    mask_mid = ~(mask_top | mask_bot)

    u_analytical[mask_top] = -Bn * (1. - eta[mask_top]) + (1. - np.square(eta[mask_top]))
    u_analytical[mask_bot] = -Bn * (1. + eta[mask_bot]) + (1. - np.square(eta[mask_bot]))
    u_analytical[mask_mid] = (1. - Bn / 2.) ** 2

    return np.c_[U_inf * u_analytical, np.zeros_like(u_analytical)]


def check_incompressibility(sim: Simulation_2D, k: int, u_field):
    s: float = 0.
    for elem in sim.n2e_map[sim.n2e_st[k]: sim.n2e_st[k+1]]:
        loc_k = np.argwhere(sim.elem_node_tags[elem, :3] == k)[0][0]
        print("k = ", loc_k)
        local_nodes = sim.elem_node_tags[elem]
        sub_s = 0.
        for g, wg in enumerate(sim.weights_q):
            dudx = 0.
            dvdy = 0.
            dphi = np.dot(sim.dv_shape_functions_at_q[g], sim.inverse_jacobians[elem]/sim.determinants[elem])
            for j, node in enumerate(local_nodes):
                dudx += u_field[node, 0] * dphi[j, 0]
                dvdy += u_field[node, 1] * dphi[j, 1]
                # print(f"u[{node:3d}] = {u_field[node, 0]:.3f}")
            tmp = wg * sim.q_shape_functions[g, loc_k] * (dudx + dvdy) * sim.determinants[elem]
            s += tmp
            tmp = (dudx + dvdy)
            sub_s += tmp
            print(tmp)
    print("s = ", s)
    return


def dummy(sim: Simulation_2D):
    
    u_nodes = np.zeros((sim.n_node, 2))
    u_nodes[:, 0] = (1. - sim.coords[:, 1]**2) / 2.
    u_nodes[:, 1] = 1 * sim.coords[:, 0] * (1. + sim.coords[:, 1])
    # x_centered = sim.coords[:, 0] - np.mean(sim.coords[:, 0])
    # y_centered = sim.coords[:, 1] - np.mean(sim.coords[:, 1])
    # u_nodes[:, 0] = +x_centered**2-y_centered**2
    # u_nodes[:, 1] = -2*x_centered*y_centered
    # u_nodes[:, 0] = (0.25 - sim.coords[:, 1] ** 2) * np.sin(np.pi * sim.coords[:, 0]) ** 2
    # u_nodes[:, 1] = (0.25 - sim.coords[:, 1] ** 2) * sim.coords[:, 0] * (3. - sim.coords[:, 0])
    p_field = np.zeros(sim.primary_nodes.size - sim.nodes_singular_p.size)
    t_num = np.ones((sim.n_elem, sim.ng_loc))

    return u_nodes, p_field, t_num
