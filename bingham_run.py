from bingham_structure import *
from bingham_fem_solver import solve_FE_sparse
from bingham_post_pro import plot_1D_slice, plot_solution_2D, plot_solution_2D_matplotlib
from bingham_tracking import solve_interface_tracking

def load_solution(res_file_name, simu_number):
    res_file_name += f"_{simu_number:d}" if simu_number >= 0 else ""
    with open(f"./res/{res_file_name:s}.txt", 'r') as file:
        K, tau_zero = float(next(file).strip('\n')), float(next(file).strip('\n')),
        f = [float(component) for component in next(file).strip('\n').split(' ')]
        element, model_name = next(file).strip('\n'), next(file).strip('\n')
        u_num = np.loadtxt(file)

    return dict(K=K, tau_zero=tau_zero, f=f, element=element, model_name=model_name), u_num



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
        # parameters, u_nodes = load_solution("rectangle", 0)
        # parameters, u_nodes = load_solution("rectangle", 1)
        # parameters, u_nodes = load_solution("rect_fit", 1)
        parameters, u_nodes = load_solution("cylinder", 0)
        # parameters, u_nodes = load_solution("cylinder", 1)
        # parameters, u_nodes = load_solution("cavity", 0)
        # parameters, u_nodes = load_solution("cavity", 1)
        # parameters, u_nodes = load_solution("bfs", 0)
        # parameters, u_nodes = load_solution("bfs", 1)
    elif mode in [2, 3, 4]:
        parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="taylor-hood", model_name="test")
        # parameters = dict(K=1., tau_zero=0.1, f=[1., 0.], element="taylor-hood", model_name="rectangle")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="taylor-hood", model_name="rectangle")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="taylor-hood", model_name="rect_fit")
        # parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="taylor-hood", model_name="cylinder")
        # parameters = dict(K=1., tau_zero=5., f=[0., 0.], element="taylor-hood", model_name="cavity")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="taylor-hood", model_name="bfs")
    else:
        raise ValueError

    sim = Simulation_2D(**parameters)
    print(sim.n_node)

    if mode == 2:  # Solve the problem: ITERATE
        u_nodes = solve_interface_tracking(sim, atol=1e-8, rtol=1e-6)

    elif mode == 3:  # Solve problem: ONE SHOT
        # u_nodes = solve_FE(sim, atol=1e-8, rtol=1e-6)
        u_nodes = solve_FE_sparse(sim, solver_name='mosek', strong=True)
        
        # from bingham_fem_mosek import impose_strong_divergence
        # impose_strong_divergence(sim)
        
        sim.save_solution(u_nodes)

    elif mode == 4:  # DUMMY solution to debug
        u_nodes = np.zeros((sim.n_node, 2))
        u_nodes[:, 0] = (1. - sim.coords[:, 1]**2) / 2.
        u_nodes[:, 1] = 0 * sim.coords[:, 0] * (1. + sim.coords[:, 1])

    plot_solution_2D(u_nodes, sim)
    # plot_1D_slice(u_nodes, sim)

    gmsh.finalize()
    # python3 bingham_run.py -mode 3
