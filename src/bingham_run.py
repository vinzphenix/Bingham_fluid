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


def compute_streamfunction(sim: Simulation_2D, u_num, t_num):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    ###############################  -  Mass matrix  -  ###############################
    dphi_loc = sim.dv_shape_functions_at_v  # size (ng_v, nsf, 2)
    inv_jac = sim.inverse_jacobians
    mass_matrices = np.einsum(
        'g,gkm,imn,ion,gjo,i->ikj',
        sim.weights, dphi_loc, inv_jac, inv_jac, dphi_loc, 1. / sim.determinants
    )  # size (ne, nsf, nsf)
    rows = sim.elem_node_tags[:, :, np.newaxis].repeat(sim.nsf, axis=2)
    cols = sim.elem_node_tags[:, np.newaxis, :].repeat(sim.nsf, axis=1)
    rows = rows.flatten()
    cols = cols.flatten()
    mass_matrices = mass_matrices.flatten()
    
    mass_matrices[rows == 0] = 0.
    mass_matrices[(rows == 0) & (cols == 0)] = 1.
    mass_matrix = csr_matrix((mass_matrices, (rows, cols)), shape=(sim.n_node, sim.n_node))

    #############################  -  Vorticity source  -  ############################
    phi = sim.v_shape_functions
    rotated_v = u_num[sim.elem_node_tags]
    rotated_v[:, :, 0] *= -1.
    rotated_v[:, :, :] = rotated_v[:, :, ::-1]
    source = np.einsum(
        'g,ijn,gjm,imn,gk->ik',
        sim.weights, rotated_v, dphi_loc, inv_jac, phi
    )  # size (ne, nsf)
    source = source.flatten()
    rows = sim.elem_node_tags.flatten()
    cols = np.zeros(sim.elem_node_tags.size, dtype=int)
    rhs_vorticity = csr_matrix((source, (rows, cols)), shape=(sim.n_node, 1))
    rhs_vorticity = rhs_vorticity.toarray().flatten()

    #############################  -  Boundary source  -  #############################
    gmsh.model.mesh.createEdges()
    line_tag, n_pts = (1, 2) if sim.element == "mini" else (8, 3)
    edge_node_tags = gmsh.model.mesh.getElementEdgeNodes(elementType=line_tag)
    edge_node_tags = np.array(edge_node_tags).astype(int) - 1
    edge_node_tags = edge_node_tags.reshape((-1, n_pts))
    n_edge = edge_node_tags.shape[0]

    coords_org = sim.coords[edge_node_tags[:, 0]]
    coords_dst = sim.coords[edge_node_tags[:, 1]]
    length = np.linalg.norm(coords_dst - coords_org, axis=1)
    tangent = -(coords_dst - coords_org) / length[:, None]

    uvw = 0.5 + 0.5 * np.array([-1. / np.sqrt(3), 1. / np.sqrt(3)])
    ng_edge = len(uvw)
    integral_rule = "Gauss" + str(2 * (sim.element == "mini") + 4 * (sim.element == "th"))
    uvw, weights_edge = gmsh.model.mesh.getIntegrationPoints(line_tag, integral_rule)
    ng_edge = len(weights_edge)
    _, sf, _ = gmsh.model.mesh.getBasisFunctions(line_tag, uvw, 'Lagrange')
    sf_edge = np.array(sf).reshape((ng_edge, -1))

    edge_v = u_num[edge_node_tags]
    source_boundary = np.einsum(
        'g,ijn,gj,in,gk,i->ik',
        weights_edge, edge_v, sf_edge, tangent, sf_edge, length / 2.
    )  # size (ne, nsf)

    source_boundary = source_boundary.flatten()
    rows = edge_node_tags.flatten()
    cols = np.zeros(edge_node_tags.size, dtype=int)
    rhs_boundary = csr_matrix((source_boundary, (rows, cols)), shape=(sim.n_node, 1))
    rhs_boundary = rhs_boundary.toarray().flatten()

    #############################  -  System solve  -  #############################
    rhs = rhs_vorticity + rhs_boundary
    rhs[0] = 0.
    psi = spsolve(mass_matrix, rhs)

    return psi

    # perm = reverse_cuthill_mckee(mass_matrix, symmetric_mode=True)
    # plt.spy(mass_matrix.todense()[np.ix_(perm, perm)])
    # plt.show()

    gmsh.fltk.initialize()

    tag_psi = gmsh.view.add("streamfunction", tag=-1)
    gmsh.view.addHomogeneousModelData(
        tag_psi, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, psi, numComponents=1
    )
    gmsh.view.option.set_number(tag_psi, "IntervalsType", 0)
    gmsh.fltk.run()
    gmsh.fltk.finalize()



    # sb = np.zeros((n_edge, n_pts))
    # for edge in range(n_edge):
    #     org, dst = edge_node_tags[edge, 0], edge_node_tags[edge, 1]
    #     normal = np.array([sim.coords[dst, 1] - sim.coords[org, 1], sim.coords[org, 0] - sim.coords[dst, 0]])
    #     nx, ny = normal / np.hypot(*normal)

    #     for k in range(n_pts):
    #         for g, w in enumerate(weights_edge):
    #             u_loc = 0.
    #             v_loc = 0.
    #             for j in range(n_pts):
    #                 node = edge_node_tags[edge, j]
    #                 u_loc += sf_edge[g, j] * u_num[node, 0]
    #                 v_loc += sf_edge[g, j] * u_num[node, 1]
    #             sb[edge, k] += w * sf_edge[g, k] * (-v_loc * nx  + u_loc * ny) * length[edge] / 2.

    # start2 = perf_counter()
    # big_m = np.zeros((sim.n_node, sim.n_node))
    # # big_r = np.zeros((sim.n_elem, sim.nsf), dtype=float)
    # big_r = np.zeros(sim.n_node, dtype=float)
    # for i in range(sim.n_elem):
    #     mass = np.zeros((sim.nsf, sim.nsf))

    #     for k in range(sim.nsf):
    #         for g, wg in enumerate(sim.weights):

    #             dphi = sim.dv_shape_functions_at_v[g]
    #             dphi = np.dot(dphi, sim.inverse_jacobians[i]) / sim.determinants[i]

    #             for j in range(sim.nsf):
    #                 node = sim.elem_node_tags[i, j]
    #                 mass[k, j] += wg * (dphi[k, 0] * dphi[j, 0] + dphi[k, 1] *
    #                                     dphi[j, 1]) * sim.determinants[i]
    #                 big_r[sim.elem_node_tags[i, k]] += wg * \
    #                     (-u_num[node, 0] * dphi[j, 1] + u_num[node, 1] *
    #                      dphi[j, 0]) * phi[g, k] * sim.determinants[i]
    #                 # big_r[i, k] += wg * (u_num[node, 1] * dphi[j, 0] - u_num[node, 0] * dphi[j, 1]) * phi[g, k] * sim.determinants[i]

    #     big_m[np.ix_(sim.elem_node_tags[i], sim.elem_node_tags[i])] += mass
    # end2 = perf_counter()

    # print(f"Vectorized : {1.e3 * (end - start):.3f} ms")
    # print(f"Loops      : {1.e3 * (end2 - start2):.3f} ms")

    # fig, axs = plt.subplots(1, 2, figsize=(10., 5), constrained_layout=True, sharey='all', sharex='all')
    # mat_dense = mass_matrix.toarray()
    # vmax = np.amax(np.abs(mat_dense))
    # axs[0].imshow(mat_dense, cmap=plt.get_cmap('bwr'), vmin=-vmax, vmax=vmax)
    # vmax = np.amax(big_m - mat_dense)
    # axs[1].imshow(big_m - mat_dense, cmap=plt.get_cmap('bwr'), vmin=-vmax, vmax=vmax)
    # plt.show()

    # print(np.sum(big_m, axis=1))
    # print(np.amax(mass_global[i] - mass))

    # bd_nodes, _, _ = gmsh.model.mesh.getNodes(1, -1, True, False)
    # bd_nodes = np.array(bd_nodes).astype(int) - 1
    # bd_elems_list = [sim.n2e_map[sim.n2e_st[node]: sim.n2e_st[node + 1]] for node in bd_nodes]
    # bd_elems = np.unique(np.concatenate(bd_elems_list))

    # n_pts_edge = 2 if sim.element == "mini" else 3
    # mask_edge_bd = np.isin(sim.elem_node_tags[bd_elems], bd_nodes)
    # bd_elems = bd_elems[np.sum(mask_edge_bd, axis=1) == n_pts_edge]

    return psi


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
        mode = 4

    gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="th", model_name="test")
    # parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="th", model_name="rectangle")
    # parameters = dict(K=1., tau_zero=0.25, f=[1., 0.], element="th", model_name="rectangle")
    # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="th", model_name="rect_fit")
    # parameters = dict(K=1., tau_zero=0.9, f=[1., 0.], element="th", model_name="cylinder")
    # parameters = dict(K=1., tau_zero=500., f=[0., 0.], element="th", model_name="cavity")
    # parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="th", model_name="bfs")
    # parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="th", model_name="opencavity")

    sim = Simulation_2D(parameters)

    if mode == 1:  # Solve problem: ONE SHOT
        u_field, p_field, d_field = solve_FE_mosek(sim, strong=False)
        sim.save_solution(u_field, p_field, d_field, model_variant='oneshot')

    elif mode == 2:  # Solve the problem: ITERATE
        u_field, p_field, d_field = solve_interface_tracking(sim, max_it=5, tol_delta=1.e-3, deg=1)
        sim.save_solution(u_field, p_field, d_field, model_variant=f'{sim.tau_zero:.0f}')

    elif mode == 3:  # Load solution from disk
        model, variant = "opencavity", "oneshot"
        parameters, u_field, p_field, d_field, coords = load_solution(model, variant)
        sim = Simulation_2D(parameters, new_coords=coords)

    else:  # DUMMY solution to debug
        u_field, p_field, d_field = dummy()
        u_field[:, 0] = (0.25 - sim.coords[:, 1] ** 2) * np.sin(np.pi * sim.coords[:, 0]) ** 2
        u_field[:, 1] = (0.25 - sim.coords[:, 1] ** 2)
        psi = compute_streamfunction(sim, u_field, d_field)

    # plot_solution_2D(u_field, p_field, d_field, sim)
    # plot_1D_slice(u_field, sim)

    gmsh.finalize()
