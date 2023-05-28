from bingham_structure import *
from bingham_fem_solver import solve_FE_sparse
from bingham_fem_mosek import solve_FE_mosek
from bingham_post_pro import plot_1D_slice, plot_solution_2D, plot_solution_2D_matplotlib
from bingham_tracking import solve_interface_tracking


def load_solution(model_name, model_variant):

    res_file_name = f"../res/{model_name:s}_{model_variant:s}"

    with open(res_file_name + "_params.txt", 'r') as file:
        K = float(next(file).strip('\n'))
        tau_zero = float(next(file).strip('\n'))
        f = [float(component) for component in next(file).strip('\n').split(' ')]
        element = next(file).strip('\n')

    u_num = np.loadtxt(res_file_name + "_velocity.txt")
    p_num = np.loadtxt(res_file_name + "_pressure.txt")
    t_num = np.loadtxt(res_file_name + "_strain.txt")
    coords = np.loadtxt(res_file_name + "_coords.txt")

    dic_params = dict(K=K, tau_zero=tau_zero, f=f, element=element, model_name=model_name)

    return dic_params, u_num, p_num, t_num, coords


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


def check_incompressibility(k):
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


def dummy():
    u_nodes = np.zeros((sim.n_node, 2))
    u_nodes[:, 0] = (1. - sim.coords[:, 1]**2) / 2.
    u_nodes[:, 1] = 1 * sim.coords[:, 0] * (1. + sim.coords[:, 1])
    # x_centered = sim.coords[:, 0] - np.mean(sim.coords[:, 0])
    # y_centered = sim.coords[:, 1] - np.mean(sim.coords[:, 1])
    # u_nodes[:, 0] = +x_centered**2-y_centered**2
    # u_nodes[:, 1] = -2*x_centered*y_centered
    p_field = np.zeros(sim.primary_nodes.size - sim.nodes_singular_p.size)
    t_num = np.ones((sim.n_elem, sim.ng_loc))
    return u_nodes, p_field, t_num


if __name__ == "__main__":

    # 1: solve problem oneshot,
    # 2: solve problem iterative,
    # 3: load previous,
    # 4: dummy solver debug

    if len(sys.argv) == 3 and sys.argv[1] == "-mode":
        mode = int(sys.argv[2])
        if not (1 <= mode <= 4):
            print("'mode' should be an integer from 1 to 4")
            exit(1)
    else:
        mode = 2

    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    # parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="th", model_name="test")
    # parameters = dict(K=1., tau_zero=0.25, f=[0., 0.], element="th", model_name="rectangle")
    
    # beta = np.sin(0.)
    # parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="th", model_name="rectanglerot")
    parameters = dict(K=1., tau_zero=0.2, f=[0., 0.], element="th", model_name="pipe_dense")
    # parameters = dict(K=1., tau_zero=0.1, f=[0., 0.], element="th", model_name="pipeneck")

    # parameters = dict(K=1., tau_zero=50., f=[0., 0.], element="th", model_name="cylinder")
    # parameters = dict(K=1., tau_zero=500., f=[0., 0.], element="th", model_name="cavity")
    # parameters = dict(K=1., tau_zero=50., f=[0., 0.], element="th", model_name="opencavity")
    # parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="th", model_name="bfs")

    sim = Simulation_2D(parameters)

    if mode == 1:  # Solve problem: ONE SHOT
        u_field, p_field, d_field = solve_FE_mosek(sim, strong=False)
        # sim.save_solution(u_field, p_field, d_field, model_variant='')

        # sim.f[0] = 1.
        # u_field, p_field, d_field = solve_FE_sparse(sim, strong=False, solver_name='mosek')

    elif mode == 2:  # Solve the problem: ITERATE
        res = solve_interface_tracking(sim, max_it=10, tol_delta=1.e-4, deg=1, strong=False)
        u_field, p_field, d_field = res
        sim.save_solution(u_field, p_field, d_field, model_variant=f'')

    elif mode == 3:  # Load solution from disk
        # model, variant = "cavity", "test"
        # model, variant = "opencavity", "classic"
        # model, variant = "cylinder", "double"
        model, variant = "pipe", "classic"
        model, variant = "pipe_dense", ""
        # model, variant = "pipeneck", "smooth"
        # model, variant = "pipeneck", "sharp"

        parameters, u_field, p_field, d_field, coords = load_solution(model, variant)
        # model, variant = "rectanglerot", "bcGood"
        # parameters, u_field2, p_field2, d_field2, coords = load_solution(model, variant)
        # u_field -= u_field2
        # p_field -= p_field2

        sim = Simulation_2D(parameters, new_coords=coords)

    else:  # DUMMY solution to debug
        u_field, p_field, d_field = dummy()
        # u_field[:, 0] = (0.25 - sim.coords[:, 1] ** 2) * np.sin(np.pi * sim.coords[:, 0]) ** 2
        # u_field[:, 1] = (0.25 - sim.coords[:, 1] ** 2) * sim.coords[:, 0] * (3. - sim.coords[:, 0])

    plot_solution_2D(u_field, p_field, d_field, sim)
    # plot_1D_slice(u_field, sim, extra_name="2D_last")

    gmsh.finalize()
