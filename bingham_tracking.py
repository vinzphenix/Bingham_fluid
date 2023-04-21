from bingham_structure import *

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