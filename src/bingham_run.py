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
    dic_params = dict(K=K, tau_zero=tau_zero, f=f, element=element, model_name=model_name)

    return dic_params, u_num, p_num


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


if __name__ == "__main__":

    if len(sys.argv) == 3 and sys.argv[1] == "-mode":
        mode = int(sys.argv[2])
    else:
        mode = 1

    # 1: load previous,
    # 2: solve problem iterative,
    # 3: solve problem oneshot,
    # 4: dummy solver debug

    gmsh.initialize()

    if mode == 1:
        # parameters, u_nodes, p_field = load_solution("test", "")
        parameters, u_nodes, p_field = load_solution("rectangle", "weak")
        # parameters, u_nodes, p_field = load_solution("cylinder", "")
        # parameters, u_nodes, p_field = load_solution("cavity", "")
        # parameters, u_nodes, p_field = load_solution("bfs", "")
        pass
    elif mode in [2, 3, 4]:
        parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="mini", model_name="test")
        # parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="th", model_name="rectangle")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="th", model_name="rectangle")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="th", model_name="rect_fit")
        # parameters = dict(K=1., tau_zero=0.9, f=[1., 0.], element="th", model_name="cylinder")
        # parameters = dict(K=1., tau_zero=10., f=[0., 0.], element="th", model_name="cavity")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="th", model_name="bfs")
        pass
    else:
        raise ValueError

    sim = Simulation_2D(**parameters)
    print(sim.n_node)

    if mode == 1:
        # _, u_nodes_1, p_field_1 = load_solution("rectangle", "weak_cross")
        # _, u_nodes_2, p_field_2 = load_solution("rectangle", "strong_cross")
        u_nodes = u_nodes - get_analytical_poiseuille(sim)
        # p_field = p_field_1

    if mode == 2:  # Solve the problem: ITERATE
        u_nodes = solve_interface_tracking(sim, atol=1e-8, rtol=1e-6)

    elif mode == 3:  # Solve problem: ONE SHOT
        # for legacy only
        # u_nodes_slow = solve_FE_sparse(sim, solver_name='mosek', strong=False)
        # sim.save_solution(u_nodes, simu_nb=-1)

        u_nodes_weak, p_field_weak = solve_FE_mosek(sim, strong=False)
        # sim.save_solution(u_nodes_weak, p_field_weak, model_variant='weak')

        # u_nodes_strong, p_field_strong = solve_FE_mosek(sim, strong=True)
        # sim.save_solution(u_nodes_strong, p_field_strong, model_variant='strong_f')

        u_nodes, p_field = u_nodes_weak, p_field_weak

    elif mode == 4:  # DUMMY solution to debug
        u_nodes = np.zeros((sim.n_node, 2))
        u_nodes[:, 0] = (1. - sim.coords[:, 1]**2) / 2.
        u_nodes[:, 1] = 0 * sim.coords[:, 0] * (1. + sim.coords[:, 1])
        # x_centered = sim.coords[:, 0] - np.mean(sim.coords[:, 0])
        # y_centered = sim.coords[:, 1] - np.mean(sim.coords[:, 1])
        # u_nodes[:, 0] = +x_centered**2-y_centered**2
        # u_nodes[:, 1] = -2*x_centered*y_centered
        p_field = np.zeros(sim.primary_nodes.size - sim.nodes_singular_p.size)

    plot_solution_2D(u_nodes, p_field, sim)
    plot_1D_slice(u_nodes, sim)

    gmsh.finalize()
    # python3 bingham_run.py -mode 3
