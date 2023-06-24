from bingham_structure import *
from bingham_fem_solver import solve_FE_sparse
from bingham_fem_mosek import solve_FE_mosek
from bingham_post_pro import plot_1D_slice, plot_solution_2D, plot_solution_2D_matplotlib
from bingham_tracking import solve_interface_tracking
from utils import *


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

    dic_params = dict(K=K, tau_zero=tau_zero, f=f, elem=element, model=model_name)

    return dic_params, u_num, p_num, t_num, coords


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
        mode = 3

    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    ##########################################################################################
    #######################  -  Simulation parameters and geometry  -  #######################

    # parameters = dict(K=1., tau_zero=0.25, f=[0., 0.], elem="th", model="rectangle")
    # parameters = dict(K=1., tau_zero=0.2, elem="th", model="pipe")
    # parameters = dict(K=1., tau_zero=0.1, elem="th", model="pipeneck")
    # parameters = dict(K=1., tau_zero=10., elem="th", model="cylinder")
    # parameters = dict(K=1., tau_zero=50., element="th", model="cavity_test")
    parameters = dict(K=1., tau_zero=5., elem="th", model="cavity")
    # parameters = dict(K=1., tau_zero=500., elem="th", model="cavity_cheat")
    # parameters = dict(K=1., tau_zero=100., elem="th", model="opencavity")
    # parameters = dict(K=1., tau_zero=0., elem="th", model="bfs")
    # parameters = dict(K=1., tau_zero=t, elem="th", model="finepipe")

    sim = Simulation_2D(parameters, save_variant="")
    # sim = Simulation_2D(parameters, save_variant=f"{1e3 * parameters['tau_zero']:.0f}")

    ##########################################################################################
    #############################  -  Solve / Iterate / Load  -  #############################

    if mode == 1:  # Solve problem: ONE SHOT
        u_field, p_field, d_field = solve_FE_mosek(sim, strong=False)

    elif mode == 2:  # Solve the problem: ITERATE
        res = solve_interface_tracking(sim, max_it=10, tol_delta=1.e-8, deg=1, strong=False)
        u_field, p_field, d_field = res

    elif mode == 3:  # Load solution from disk
        # model, variant = "cavity_test", "50"
        # model, variant = "cavity", "20"
        # model, variant = "cavity_cheat", "500"
        # model, variant = "opencavity", "100"
        # model, variant = "pipe", "classic"
        # model, variant = "finepipe", "225"  # f"{1e3 * t:.0f}"
        # model, variant = "necksmooth", "default"
        # model, variant = "necksharp", "default"
        model, variant = "cylinder", "10"

        parameters, u_field, p_field, d_field, coords = load_solution(model, variant)
        sim = Simulation_2D(parameters, new_coords=coords)

    else:  # DUMMY solution to debug
        u_field, p_field, d_field = dummy()

    ##########################################################################################
    ####################################  -  Display  -  #####################################

    plot_solution_2D(u_field, p_field, d_field, sim)
    # plot_1D_slice(u_field, sim, extra_name="2D_last")

    gmsh.finalize()
