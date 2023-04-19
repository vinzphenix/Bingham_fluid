from bingham_2D_structure import *
from fem_solver_2D import *


def load_solution(res_file_name, simu_number):
    res_file_name += f"_{simu_number:d}" if simu_number >= 0 else ""
    with open(f"./res/{res_file_name:s}.txt", 'r') as file:
        K, tau_zero = float(next(file).strip('\n')), float(next(file).strip('\n')),
        f = [float(component) for component in next(file).strip('\n').split(' ')]
        element, mesh_filename = next(file).strip('\n'), next(file).strip('\n')
        u_num = np.loadtxt(file)

    return dict(K=K, tau_zero=tau_zero, f=f, element=element, mesh_filename=mesh_filename), u_num


def eval_velocity_gradient(u_local, dphi_local):
    dudx = np.dot(u_local[:, 0], dphi_local[:, 0])
    dudy = np.dot(u_local[:, 0], dphi_local[:, 1])
    dvdx = np.dot(u_local[:, 1], dphi_local[:, 0])
    dvdy = np.dot(u_local[:, 1], dphi_local[:, 1])
    return dudx, dudy, dvdx, dvdy


def compute_strain_per_elem(sim: Simulation_2D, u_num, strain_norm_avg):
    strain_norm_avg[:] = 0.
    for i in range(sim.n_elem):
        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        for g, wg in enumerate(sim.weights):
            dphi = sim.dv_shape_functions_at_v[g]  # size (n_sf, 2)
            dphi = np.dot(dphi, inv_jac) / det  # size (n_sf, 2)
            l11, l12, l21, l22 = eval_velocity_gradient(u_num[idx_local_nodes], dphi)
            strain_norm_avg[i] += (2. * wg) * np.sqrt(0.5 * l11 ** 2 + 0.5 * l22 ** 2 + 0.25 * (l12 + l21) ** 2)
            # multiplied 2 bc sum(wg) = 0.5
    return


def compute_gradient_at_nodes(sim: Simulation_2D, u_num, velocity_gradient):
    velocity_gradient[:] = 0.
    _, n_local_node, _ = velocity_gradient.shape  # n_elem, n_local, 9
    _, dsf_at_nodes, _ = gmsh.model.mesh.getBasisFunctions(sim.elem_type, sim.local_node_coords, 'GradLagrange')
    dsf_at_nodes = np.array(dsf_at_nodes).reshape((n_local_node, n_local_node, 3))[:, :, :-1]

    for i in range(sim.n_elem):
        idx_local_nodes = sim.elem_node_tags[i, :]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        for j, idx_node in enumerate(idx_local_nodes):
            dphi = dsf_at_nodes[j, :]  # dphi in reference element
            dphi = np.dot(dphi, inv_jac) / det  # dphi in physical element
            l11, l12, l21, l22 = eval_velocity_gradient(u_num[idx_local_nodes], dphi)
            velocity_gradient[i, j, np.array([0, 1, 3, 4])] = np.array([l11, l12, l21, l22])
    return


def get_neighbours_mapping(sim: Simulation_2D):
    gmsh.model.mesh.create_edges()

    edge_nodes = gmsh.model.mesh.getElementEdgeNodes(sim.elem_type, tag=-1, primary=True)
    # edge_nodes = edge_nodes.reshape((sim.n_elem, 3, 2))

    edge_tags, _ = gmsh.model.mesh.getEdges(edge_nodes)
    edge_to_elem = {}
    elem_to_edge = {}
    for i, edge_tag in enumerate(edge_tags):
        edge_to_elem.setdefault(edge_tag, []).append(i // 3)
        elem_to_edge.setdefault(i // 3, []).append(edge_tag)

    neighbours_map = {}
    for elem in range(sim.n_elem):
        neighbours_map[elem] = []
        local_edges = elem_to_edge[elem]
        for edge in local_edges:
            neighbours_map[elem] += [neigh_elem for neigh_elem in edge_to_elem[edge] if neigh_elem != elem]

    return neighbours_map


def find_neighbours_solid_regions(sim: Simulation_2D, strain_norm, neighbours_map):
    tol = 1.e-3
    solid_elements, = np.where(strain_norm < tol)
    elements_accros_interface = []
    for elem in solid_elements:
        elements_accros_interface += neighbours_map[elem]

    elements_accros_interface = np.array(elements_accros_interface)
    elements_accros_interface = np.unique(elements_accros_interface)
    elements_accros_interface = np.setdiff1d(elements_accros_interface, solid_elements)

    return elements_accros_interface


def solve_interface_tracking(sim: Simulation_2D, atol=1e-8, rtol=1e-6, max_it=20, tol_unyielded=1.e-3):

    strain_norm = np.zeros(sim.n_elem)
    neighbours_map = get_neighbours_mapping(sim)

    # Solve first time with initial mesh
    u_num = solve_FE_sparse(sim, solver_name='mosek', strong=False)

    while sim.iteration < max_it:
        print("")
        compute_strain_per_elem(sim, u_num, strain_norm)
        neighbours = find_neighbours_solid_regions(sim, strain_norm, neighbours_map)
        break

    return u_num


def plot_solution_2D(u_num, sim: Simulation_2D):

    gmsh.fltk.initialize()
    modelName = gmsh.model.list()[0]
    # tag_psi = gmsh.view.add("psi")
    # show P1 basis
    # for j, idx_node in enumerate(sim.primary_nodes):
    #     data = np.zeros(sim.primary_nodes.size)
    #     if idx_node not in np.r_[sim.nodes_zero_u, sim.nodes_zero_v, sim.nodes_with_u]:
    #         data[j] = 1.
    #     gmsh.view.addHomogeneousModelData(tag_psi, j, modelName, "NodeData", sim.primary_nodes + 1, data, time=j, numComponents=1)

    if sim.degree == 3:
        sim.dv_shape_functions_at_v = sim.dv_shape_functions_at_v[:, :-1, :]
        sim.elem_node_tags = sim.elem_node_tags[:, :-1]

    n_local_node = sim.elem_node_tags.shape[1]
    velocity = np.c_[u_num, np.zeros_like(u_num[:, 0])]
    strain_tensor = np.zeros((sim.n_elem, n_local_node, 9))
    strain_norm_avg = np.zeros(sim.n_elem)

    compute_strain_per_elem(sim, u_num, strain_norm_avg)
    compute_gradient_at_nodes(sim, u_num, strain_tensor)  # filled with grad(v) for now

    vorticity = (strain_tensor[:, :, 3] - strain_tensor[:, :, 1]).copy().flatten()
    divergence = (strain_tensor[:, :, 0] - strain_tensor[:, :, 4]).copy().flatten()
    velocity = velocity.flatten()
    strain_norm_avg = strain_norm_avg.flatten()
    strain_tensor[:, :, 1] = 0.5 * (strain_tensor[:, :, 1] + strain_tensor[:, :, 3])  # symmetrize grad(v)
    strain_tensor[:, :, 3] = strain_tensor[:, :, 1]
    strain_tensor = 2 * strain_tensor.flatten() / np.sqrt(3)  # compute 2D, and rescale because of 3/2 in Von Mises

    tag_v = gmsh.view.add("Velocity", tag=1)
    tag_strain = gmsh.view.add("Strain tensor", tag=2)
    tag_vorticity = gmsh.view.add("Vorticity", tag=3)
    tag_divergence = gmsh.view.add("Divergence", tag=4)
    tag_strain_norm_avg = gmsh.view.add("Strain norm averaged", tag=5)

    gmsh.view.addHomogeneousModelData(
        tag_v, 0, modelName, "NodeData", sim.node_tags + 1, velocity, numComponents=3)
    gmsh.view.addHomogeneousModelData(
        tag_strain, 0, modelName, "ElementNodeData", sim.elem_tags, strain_tensor, numComponents=9)
    gmsh.view.addHomogeneousModelData(
        tag_vorticity, 0, modelName, "ElementNodeData", sim.elem_tags, vorticity, numComponents=1)
    gmsh.view.addHomogeneousModelData(
        tag_divergence, 0, modelName, "ElementNodeData", sim.elem_tags, divergence, numComponents=1)
    gmsh.view.addHomogeneousModelData(
        tag_strain_norm_avg, 0, modelName, "ElementData", sim.elem_tags, strain_norm_avg, numComponents=1)

    gmsh.view.option.setNumber(tag_v, "VectorType", 6)
    gmsh.view.option.setNumber(tag_v, "DrawLines", 0)
    gmsh.view.option.setNumber(tag_v, "DrawPoints", 0)
    gmsh.view.option.setNumber(tag_v, "NormalRaise", -0.5 / np.amax(np.hypot(u_num[:, 0], u_num[:, 1])))
    gmsh.view.option.setNumber(tag_strain_norm_avg, "NormalRaise", 0.5 / np.amax(strain_norm_avg))
    for tag in [tag_v, tag_strain, tag_vorticity, tag_divergence]:
        gmsh.view.option.setNumber(tag, "AdaptVisualizationGrid", 1)
        gmsh.view.option.setNumber(tag, "TargetError", -0.0001)
        gmsh.view.option.setNumber(tag, "MaxRecursionLevel", 2)
    for tag in [tag_vorticity, tag_divergence, tag_strain_norm_avg]:
        gmsh.view.option.setNumber(tag, "Visible", 0)

    gmsh.fltk.run()
    return


def plot_solution_2D_matplotlib(u_num, sim: Simulation_2D):

    fig, ax = plt.subplots(1, 1, figsize=(10., 6.), constrained_layout=True)

    n_elem, n_node_per_elem = sim.elem_node_tags.shape
    coords = sim.coords  # size (n_nodes, 2)

    triang = mpl_tri.Triangulation(sim.coords[:, 0], sim.coords[:, 1], sim.elem_node_tags[:, :3])
    tricontourset = ax.tricontourf(triang, u_num[:sim.n_node, 0])
    ax.triplot(triang, 'ko-', alpha=0.5)
    _ = fig.colorbar(tricontourset)

    ax.quiver(coords[:, 0], coords[:, 1], u_num[:sim.n_node, 0], u_num[:sim.n_node, 1],
              angles='xy', scale_units='xy', scale=5)
    # ax.quiver(centers[:, 0], centers[:, 1], u_num[sim.n_node:, 0], u_num[sim.n_node:, 1],
    #  color='C2', angles='xy', scale_units='xy', scale=5)

    plt.show()

    return


def plot_1D_slice(u_num, sim: Simulation_2D):

    if sim.mesh_filename[:4] != "rect":
        return

    slice_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=4)
    slice_node_tags = np.array(slice_node_tags).astype(int) - 1

    slice_xy = sim.coords[slice_node_tags]
    slice_y = slice_xy[:, 1]
    slice_u = u_num[slice_node_tags, 0]  # only u component of (u, v)
    H = np.amax(slice_y)

    arg_sorted_tags = np.argsort(slice_y)
    slice_u = slice_u[arg_sorted_tags]

    if sim.degree == 3:  # MINI
        slice_y = slice_y[arg_sorted_tags]
        n_intervals = len(slice_node_tags) - 1
        deg_along_edge = 1
    else:
        slice_y = slice_y[arg_sorted_tags][::2]
        n_intervals = (len(slice_node_tags) - 1) // 2
        deg_along_edge = 2

    sim_1D = Simulation_1D(
        H=H, K=sim.K, tau_zero=sim.tau_zero, f=sim.f[0], deg=deg_along_edge,
        nElem=n_intervals, random_seed=-1, fix_interface=False, save=False
    )
    sim_1D.set_y(slice_y)
    plot_solution_1D(sim=sim_1D, u_nodes=slice_u)
    return


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
        # parameters, u_nodes = load_solution("rect_coarse", -1)
        # parameters, u_nodes = load_solution("rect_dirichlet", -1)
        # parameters, u_nodes = load_solution("hole_normal", -1)
        # parameters, u_nodes = load_solution("hole_normal", 1)
        # parameters, u_nodes = load_solution("cavity_fine", 1)
        parameters, u_nodes = load_solution("cavity_fine", -1)
        # parameters, u_nodes = load_solution("bckw_fs", -1)
    elif mode in [2, 3, 4]:
        # parameters = dict(K=1., tau_zero=0.25, f=[1., 0.], element="taylor-hood", mesh_filename="rect_coarse")
        # parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="taylor-hood", mesh_filename="rect_dirichlet")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="taylor-hood", mesh_filename="hole_normal")
        parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="taylor-hood", mesh_filename="cavity_fine")
        # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="taylor-hood", mesh_filename="bckw_fs")
        # parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="taylor-hood", mesh_filename="test")
    else:
        raise ValueError

    sim = Simulation_2D(**parameters)
    print(sim.n_node)

    if mode == 2:  # Solve the problem: ITERATE
        u_nodes = solve_interface_tracking(sim, atol=1e-8, rtol=1e-6)

    elif mode == 3:  # Solve problem: ONE SHOT
        # u_nodes = solve_FE(sim, atol=1e-8, rtol=1e-6)
        u_nodes = solve_FE_sparse(sim, solver_name='mosek', strong=False)
        sim.save_solution(u_nodes)

    elif mode == 4:  # DUMMY solution to debug
        u_nodes = np.zeros((sim.n_node, 2))
        u_nodes[:, 0] = (1. - sim.coords[:, 1]**2) / 2.
        u_nodes[:, 1] = 0 * sim.coords[:, 0] * (1. + sim.coords[:, 1])

    plot_solution_2D(u_nodes, sim)
    plot_1D_slice(u_nodes, sim)

    gmsh.finalize()
    # python3 bingham_2D_run.py -mode 3
