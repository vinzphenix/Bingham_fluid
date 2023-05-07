from bingham_structure import *
from bingham_fem_mosek import solve_FE_mosek
from bingham_post_pro import plot_solution_2D


# def compute_strain_per_elem(sim: Simulation_2D, u_num, strain_norm_avg):
#     """
#     Evaluate the average value of the strain-rate norm '|D| = sqrt(1/2 D:D)'
#     Should be noted that in the objective function, we bound |2D|, twice the norm
#     """
#     strain_norm_avg[:] = 0.
#     for i in range(sim.n_elem):
#         idx_local_nodes = sim.elem_node_tags[i]
#         det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
#         for g, wg in enumerate(sim.weights):
#             dphi = sim.dv_shape_functions_at_v[g]  # size (n_sf, 2)
#             dphi = np.dot(dphi, inv_jac) / det  # size (n_sf, 2)
#             l11, l12, l21, l22 = eval_velocity_gradient(u_num[idx_local_nodes], dphi)
#             strain_norm_avg[i] += (2. * wg) * np.sqrt(2. * l11 ** 2 +
#                                                       2. * l22 ** 2 + (l12 + l21) ** 2)
#             # multiplied 2 bc sum(wg) = 0.5
#     return


# def get_neighbours_mapping(sim: Simulation_2D):
#     gmsh.model.mesh.create_edges()

#     edge_nodes = gmsh.model.mesh.getElementEdgeNodes(sim.elem_type, tag=-1, primary=True)
#     # edge_nodes = edge_nodes.reshape((sim.n_elem, 3, 2))

#     edge_tags, edge_orientations = gmsh.model.mesh.getEdges(edge_nodes)
#     edge_to_elem = {}
#     elem_to_edge = {}
#     edge_to_node = {}

#     edge_nodes = np.array(edge_nodes).reshape((-1, 2)) - 1

#     for i, (edge_tag, orientation) in enumerate(zip(edge_tags, edge_orientations)):
#         edge_to_elem.setdefault(edge_tag, []).append(i // 3)
#         elem_to_edge.setdefault(i // 3, [-1, -1, -1])[i % 3] = edge_tag
#         if orientation == 1:
#             edge_to_node[edge_tag] = (edge_nodes[i][0], edge_nodes[i][1])

#     neighbours_map = {}
#     for elem in range(sim.n_elem):
#         neighbours_map[elem] = []
#         local_edges = elem_to_edge[elem]
#         for edge in local_edges:
#             neighbours_map[elem] += [neigh_elem for neigh_elem in edge_to_elem[edge]
#                                      if neigh_elem != elem]

#     return neighbours_map, elem_to_edge, edge_to_node


# def get_diamond_mapping(sim: Simulation_2D):
#     node_to_elems = {}
#     for i, local_nodes in enumerate(sim.elem_node_tags[:, :3]):
#         for local_node in local_nodes:
#             node_to_elems.setdefault(local_node, []).append(i)

#     return node_to_elems


# def find_nodes_interface_old(sim: Simulation_2D, t_num, elem_to_edge, edge_to_node, node_to_elem, tol):
#     solid_elements, = np.where(np.all(t_num < tol, axis=1))

#     # solid_edges = np.empty(3 * solid_elements.size, dtype=int)
#     # for idx, elem in enumerate(solid_elements):
#     #     solid_edges[3 * idx:3 * (idx + 1)] = elem_to_edge[elem]
#     # unique, counts = np.unique(solid_edges, return_counts=True)
#     # interface_edges = unique[np.where(counts == 1)]  # inside edges are counted twice

#     # interface_nodes = [edge_to_node[edge] for edge in interface_edges]
#     # interface_nodes = np.unique(interface_nodes).astype(int)
#     # print(interface_nodes + 1)

#     ###################
#     nodes = np.unique(sim.elem_node_tags[solid_elements, :3])
#     mask_interface = np.empty_like(nodes)
#     for idx, node in enumerate(nodes):
#         elems = node_to_elem[node]
#         diamond_strains = t_num[elems]
#         if np.any(diamond_strains > tol):
#             mask_interface[idx] = True
#         else:
#             mask_interface[idx] = False

#     interface_nodes = nodes[mask_interface == True]
#     return interface_nodes


# def move_along_edges(sim: Simulation_2D, interface_nodes, diamond_map, reconstructed):
#     new_coords = sim.coords[interface_nodes].copy()
#     data = []

#     for idx, node in enumerate(interface_nodes):
#         print("")
#         print(f"Node {node+1:d}")

#         if abs(reconstructed[node]) > 1e-6:

#             neighbours = np.unique(sim.elem_node_tags[diamond_map[node], :3])
#             print("\t", neighbours + 1)

#             targets = [neigh for neigh in neighbours if reconstructed[neigh] *
#                        reconstructed[node] < 0.]
#             print("\t", np.array(targets) + 1)

#             distances = np.linalg.norm(sim.coords[targets] - sim.coords[node], axis=1)
#             target = targets[np.argmax(distances)]
#             print("\t", target + 1)

#             alpha = reconstructed[node] / (reconstructed[node] - reconstructed[target])
#             new_coords[idx] = (1. - alpha) * sim.coords[node] + alpha * sim.coords[target]
#             print("\t", alpha)

#             data += [*new_coords[idx], 0., distances[np.argmax(distances)]]
#             x_new, y_new = new_coords[idx]
#             print(f"\t ({sim.coords[node][0]:5.2f}, {sim.coords[node][1]:5.2f})")
#             print(f"\t ({x_new:5.2f}, {y_new:5.2f})")

#         else:
#             data += [*sim.coords[node], 0.]

#     # sim.tmp_data = data
#     return

# diamond_map = get_diamond_mapping(sim)

# strain_norm = np.zeros(sim.n_elem)
# neighbours_map, elem_to_edge, edge_to_node = get_neighbours_mapping(sim)
# node_to_elem = get_diamond_mapping(sim)


def find_nodes_interface(sim: Simulation_2D, t_num):

    # solid_elements, = np.where(np.all(t_num < sim.tol_yield, axis=1))
    solid_elements, = np.where(~sim.is_yielded(t_num))  # at least 2 out of 3
    solid_nodes = np.unique(sim.elem_node_tags[solid_elements, :3])
    mask_interface = np.empty_like(solid_nodes)

    for idx, node in enumerate(solid_nodes):
        elems = sim.n2e_map[sim.n2e_st[node]:sim.n2e_st[node + 1]]
        diamond_strains = t_num[elems]
        # if np.any(diamond_strains > sim.tol_yield):
        if np.any(sim.is_yielded(diamond_strains)):  # any elem has less than 2
            mask_interface[idx] = True
        else:
            mask_interface[idx] = False

    interface_nodes = solid_nodes[mask_interface == True]
    return interface_nodes


def build_approximation(sim: Simulation_2D, node, t_num):
    # Use strain information at all elements around 'node' to create a linear approximation

    # WARINING (no longer valid with new criteria)
    # At least one element is unyielded, i.e. not used for reconstruction
    # max_size = sim.ng_loc * (len(diamond_map[node]) - 1)

    max_pts_used = sim.ng_loc * (sim.n2e_st[node + 1] - sim.n2e_st[node])
    matrix = np.zeros((max_pts_used, 4))
    row = 0

    # for elem in diamond_map[node]:

    diamond_elems = sim.n2e_map[sim.n2e_st[node]: sim.n2e_st[node + 1]]
    for elem in diamond_elems:

        local_nodes = sim.elem_node_tags[elem, :3]
        corners_coords = sim.coords[local_nodes]

        # for each gauss point of this element
        for g, (xi, eta, _) in enumerate(sim.uvw):

            if t_num[elem, g] > sim.tol_yield:  # if information is useful
                gauss_coords = np.dot(np.array([1. - xi - eta, xi, eta]), corners_coords)
                matrix[row, :] = np.r_[1., gauss_coords, t_num[elem, g]]
                row += 1
            else:  # zero yield should not be used
                pass

    return matrix[:row]


def compute_target(sim: Simulation_2D, node, coefs):
    """
    Compute target location of 'node' based on linear approximation 'coefs'
    Nodes in the corners stay at their location
    Nodes on the boundary stay on the boundary
    Nodes in the bulk move to their projection on the zero level set of the linear approx
    """

    ux, uy = sim.coords[node]

    if node in sim.nodes_corner:
        target = sim.coords[node]

    elif node in sim.nodes_boundary:
        target = sim.coords[node]
        neighbours = sim.n2n_map[sim.n2n_st[node]: sim.n2n_st[node + 1]]

        for neigh in np.intersect1d(neighbours, sim.nodes_boundary):
            # Find intersection btw zero level-set and edge (node-neigh)
            # TODO: WARNING: not robust to parallel line-segment

            vx, vy = sim.coords[neigh]
            alpha = np.dot(coefs, [1., *sim.coords[node]])
            alpha /= (coefs[1] * (ux - vx) + coefs[2] * (uy - vy))

            if 0. <= alpha <= 1.:
                target = (1. - alpha) * sim.coords[node] + alpha * sim.coords[neigh]
                break

    else:
        det = (coefs[1]**2 + coefs[2]**2)
        inv_matrix = np.array([[coefs[1], -coefs[2]], [coefs[2], coefs[1]]])
        vector = np.array([-coefs[0], -coefs[2] * ux + coefs[1] * uy])
        target = 1. / det * np.dot(inv_matrix, vector)

    return target


def update_mesh(sim: Simulation_2D, interface_nodes, new_coords):

    # Update in gmsh api
    for idx, node in enumerate(interface_nodes):
        gmsh.model.mesh.set_node(node + 1, np.r_[new_coords[idx], 0.], [0., 0.])

    # Update in structure
    sim.coords[interface_nodes] = new_coords

    elements = np.array([], dtype=int)
    for node in interface_nodes:
        elements = np.r_[elements, sim.n2e_map[sim.n2e_st[node]:sim.n2e_st[node + 1]]]
    elements = np.unique(elements)

    # Update higher order nodes (otherwise, triangle is destroyed into hexagon)
    if sim.element == "th":
        for elem in elements:
            for j in range(3):
                local_node = sim.elem_node_tags[elem, 3 + j]
                local_coords = (sim.coords[sim.elem_node_tags[elem, j]] +
                                sim.coords[sim.elem_node_tags[elem, (j + 1) % 3]]) / 2.
                gmsh.model.mesh.set_node(local_node + 1, np.r_[local_coords, 0.], [0., 0.])
                sim.coords[local_node] = local_coords

    gmsh.model.set_current(gmsh.model.get_current())
    sim.update_transformation(elements)  # update jacobians...
    return


def eval_approx_diamond(sim: Simulation_2D, node, coefs, dic_node_approx):
    # # diamond_nodes = np.unique(sim.elem_node_tags[diamond_map[node], :3])
    neighbours = sim.n2n_map[sim.n2n_st[node]: sim.n2n_st[node + 1]]
    for diamond_node in np.r_[neighbours, node]:
        x_node, y_node = sim.coords[diamond_node]
        strain_apxm = np.dot(coefs, np.array([1., x_node, y_node]))
        dic_node_approx.setdefault(diamond_node, []).append(strain_apxm)
        # print(f"\t{diamond_node+1} : {strain_apxm}")
    return


def reconstruct(sim: Simulation_2D, t_num, interface_nodes):

    # Mapping from node to list of approximated strain values
    # Each value of the list corresponds to the approximation
    # around a specific interface node
    dic_node_approx = {}

    nodes_unchanged = []
    new_coords = sim.coords[interface_nodes].copy()
    coefs = np.empty((interface_nodes.size, 3))

    # for each node at interface
    for idx, node in enumerate(interface_nodes):
        # print(f"Node {node+1:d}")

        matrix = build_approximation(sim, node, t_num)

        if 3 <= matrix.shape[0]:  # system has a solution
            A, b = matrix[:, :3], matrix[:, 3]  # matrix, vector
            coefs[idx, :] = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))  # z = a + bx + cy
            # Evaluate linear approximation at all nodes of the diamond
            eval_approx_diamond(sim, node, coefs[idx], dic_node_approx)
            target = compute_target(sim, node, coefs[idx])
            new_coords[idx] = target
        else:
            nodes_unchanged.append(node)

    if len(nodes_unchanged) > 0:
        msg = """
        Not enough information to move some 'interface nodes'.
        They are surrounded by many unyielded elements,
        and are likely to become such during next iteration
        """
        print(msg, np.array(nodes_unchanged) + 1, "\n")

    node_strain_rec = dict((key, np.mean(strains)) for key, strains in dic_node_approx.items())
    return new_coords, node_strain_rec, coefs


def solve_interface_tracking(sim: Simulation_2D, max_it=5, tol_delta=1.e-3):

    # Solve first time with initial mesh
    u_num, p_num, t_num = solve_FE_mosek(sim, strong=False)

    while sim.iteration < max_it:

        interface_nodes = find_nodes_interface(sim, t_num)
        new_coords, strain_rec, coefs = reconstruct(sim, t_num, interface_nodes)

        extra = [interface_nodes, new_coords, strain_rec, coefs]
        plot_solution_2D(u_num, p_num, t_num, sim, extra)

        # Check if nodes moved enough to require new 'fem solve'
        delta = np.linalg.norm(sim.coords[interface_nodes] - new_coords, axis=1)
        if np.amax(delta) < tol_delta:
            break

        # Change node positions, jacobians...
        update_mesh(sim, interface_nodes, new_coords)

        # Solve the problem, slightly modified
        u_num, p_num, t_num = solve_FE_mosek(sim, strong=False)

        sim.iteration += 1

    return u_num, p_num, t_num
