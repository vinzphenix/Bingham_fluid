from bingham_structure import *
from bingham_fem_mosek import solve_FE_mosek
from bingham_post_pro import plot_solution_2D, plot_1D_slice


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


def build_approximation(sim: Simulation_2D, node, t_num, interface_nodes, deg):
    # Use strain information at all elements around 'node' to create a linear approximation

    # WARINING (no longer valid with new criteria)
    # At least one element is unyielded, i.e. not used for reconstruction
    # max_size = sim.ng_loc * (len(diamond_map[node]) - 1)

    # diamond_elems = sim.n2e_map[sim.n2e_st[node]: sim.n2e_st[node + 1]]
    support_elems = sim.get_support_approx(node)
    # print(
    #     f"singular support [{node+1:d}] ?", 
    #     f"{np.amin(sim.determinants[support_elems]):8.3g} >?<",
    #     f"{sim.min_det / 100:8.3g}"
    # )
    support_elems = support_elems[sim.determinants[support_elems] > sim.min_det / 100]
    support_nodes = np.array([], dtype=int)

    max_pts_used = sim.ng_loc * support_elems.size
    matrix = np.zeros((max_pts_used, (deg + 1) * (deg + 2) // 2 + 1))
    matrix[:, 0] = 1.  # constant coefficient

    row = 0

    for elem in support_elems:

        local_nodes = sim.elem_node_tags[elem, :3]
        corners_coords = sim.coords[local_nodes]
        support_nodes = np.r_[support_nodes, local_nodes]

        # for each gauss point of this element
        for g, (xi, eta, _) in enumerate(sim.uvw):

            if t_num[elem, g] > sim.tol_yield * 1.:  # if information is useful
                gauss_coords = np.dot(np.array([1. - xi - eta, xi, eta]), corners_coords)
                matrix[row, [1, 2]] = gauss_coords
                if deg == 2:
                    matrix[row, [3, 4]] = gauss_coords ** 2
                    matrix[row, 5] = gauss_coords[0] * gauss_coords[1]
                matrix[row, -1] = t_num[elem, g]
                row += 1
            else:  # zero yield should not be used
                pass

    return matrix[:row], np.unique(support_nodes)


# def compute_target(sim: Simulation_2D, node, coefs):
#     """
#     Compute target location of 'node' based on linear approximation 'coefs'
#     Nodes in the corners stay at their location
#     Nodes on the boundary stay on the boundary
#     Nodes in the bulk move to their projection on the zero level set of the linear approx
#     """

#     ux, uy = sim.coords[node]

#     if node in sim.nodes_corner:
#         target = sim.coords[node]

#     elif node in sim.nodes_boundary:
#         target = sim.coords[node]
#         neighbours = sim.n2n_map[sim.n2n_st[node]: sim.n2n_st[node + 1]]

#         for neigh in np.intersect1d(neighbours, sim.nodes_boundary):
#             # Find intersection btw zero level-set and edge (node-neigh)
#             # TODO: WARNING: not robust to parallel line-segment

#             vx, vy = sim.coords[neigh]
#             alpha = np.dot(coefs, [1., *sim.coords[node]])
#             alpha /= (coefs[1] * (ux - vx) + coefs[2] * (uy - vy))

#             if 0. <= alpha <= 1.:
#                 target = (1. - alpha) * sim.coords[node] + alpha * sim.coords[neigh]
#                 break

#     else:
#         det = (coefs[1]**2 + coefs[2]**2)
#         inv_matrix = np.array([[coefs[1], -coefs[2]], [coefs[2], coefs[1]]])
#         vector = np.array([-coefs[0], -coefs[2] * ux + coefs[1] * uy])
#         target = 1. / det * np.dot(inv_matrix, vector)

#     return target


def update_mesh(sim: Simulation_2D, nodes_to_move, new_coords, nodes_to_relax, init_coords_relax):

    # Relax 
    new_coords_bis = init_coords_relax * 0.2 + sim.coords[nodes_to_relax] * 0.8
    nodes_to_move = np.r_[nodes_to_move, nodes_to_relax]
    new_coords = np.r_[new_coords, new_coords_bis]

    # Update in gmsh api
    for idx, node in enumerate(nodes_to_move):
        gmsh.model.mesh.setNode(node + 1, np.r_[new_coords[idx], 0.], [0., 0.])

    # Update in structure
    sim.coords[nodes_to_move] = new_coords

    elements = np.array([], dtype=int)
    for node in nodes_to_move:
        elements = np.r_[elements, sim.n2e_map[sim.n2e_st[node]:sim.n2e_st[node + 1]]]
    elements = np.unique(elements)

    # Update higher order nodes (otherwise, triangle is destroyed into hexagon)
    if sim.element == "th":
        for elem in elements:
            for j in range(3):
                local_node = sim.elem_node_tags[elem, 3 + j]
                local_coords = (sim.coords[sim.elem_node_tags[elem, j]] +
                                sim.coords[sim.elem_node_tags[elem, (j + 1) % 3]]) / 2.
                gmsh.model.mesh.setNode(local_node + 1, np.r_[local_coords, 0.], [0., 0.])
                sim.coords[local_node] = local_coords

    gmsh.model.set_current(gmsh.model.get_current())
    sim.update_transformation(elements)  # update jacobians...
    return


def eval_approx_diamond(sim: Simulation_2D, node, coefs, supp_nodes, dic_node_approx, deg):
    def f1(x, y): return np.array([1., x, y])
    def f2(x, y): return np.array([1., x, y, x * x, y * y, x * y])
    f = f1 if deg == 1 else f2

    # # diamond_nodes = np.unique(sim.elem_node_tags[diamond_map[node], :3])
    # neighbours = sim.n2n_map[sim.n2n_st[node]: sim.n2n_st[node + 1]]
    for neigh in supp_nodes:
        x_node, y_node = sim.coords[neigh]
        strain_apxm = np.dot(coefs, f(x_node, y_node))
        dic_node_approx.setdefault(neigh, []).append(strain_apxm)
        # print(f"\t{diamond_node+1} : {strain_apxm}")
    return


def reconstruct(sim: Simulation_2D, t_num, interface_nodes, deg):

    # Mapping from node to list of approximated strain values
    # Each value of the list corresponds to the approximation
    # around a specific interface node
    dic_node_approx = {}

    nodes_unchanged = []
    new_coords = sim.coords[interface_nodes].copy()
    coefs = np.zeros((interface_nodes.size, (deg + 1) * (deg + 2) // 2))

    # Approximation with at least twice more data than coefficients
    min_pts_required = coefs.shape[1] * 2

    # for each node at interface
    for idx, node in enumerate(interface_nodes):
        # print(f"Node {node+1:d}")

        system, supp_nodes = build_approximation(sim, node, t_num, interface_nodes, deg)

        if min_pts_required <= system.shape[0]:  # system with enough data
            A, b = system[:, :-1], system[:, -1]  # matrix, vector
            coefs[idx, :] = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

            # Evaluate linear approximation at all nodes of the diamond
            eval_approx_diamond(sim, node, coefs[idx], supp_nodes, dic_node_approx, deg)

            # target = compute_target(sim, node, coefs[idx])
            # new_coords[idx] = target

        else:
            nodes_unchanged.append(node)

    # if len(nodes_unchanged) > 0:
    #     msg = """
    #     Not enough information to move some 'interface nodes'.
    #     They are surrounded by many unyielded elements,
    #     and are likely to become such during next iteration
    #     """
    #     print(msg, np.array(nodes_unchanged) + 1, "\n")

    node_strain_rec = dict((key, np.mean(strains)) for key, strains in dic_node_approx.items())
    return new_coords, node_strain_rec, coefs


def select_nodes(sim: Simulation_2D, edges, strain_map: dict):

    mask_org_pos = np.array([strain_map[edge[0]] > 0 for edge in edges])
    mask_dst_pos = np.array([strain_map[edge[1]] > 0 for edge in edges])
    cut_edges = edges[mask_org_pos != mask_dst_pos]

    corners_org = np.intersect1d(sim.nodes_corner, cut_edges[:, 0])  # always the case with gmsh
    corners_dst = np.intersect1d(sim.nodes_corner, cut_edges[:, 1])  # never the case with gmsh

    # Choose node closer to the interface as target
    strains_org = np.array([strain_map[cut_edge[0]] for cut_edge in cut_edges])
    strains_dst = np.array([strain_map[cut_edge[1]] for cut_edge in cut_edges])
    mask_move_dst = np.abs(strains_org) > np.abs(strains_dst)

    # Handle corners nodes that should not move
    mask_move_dst[corners_org] = True
    mask_move_dst[corners_dst] = False

    # # Handle boundary nodes that cannot move inside the domain
    # idxs_edge_bd_bulk = np.where(
    #     np.logical_xor(
    #         np.isin(cut_edges[:, 0], sim.nodes_boundary),
    #         np.isin(cut_edges[:, 1], sim.nodes_boundary)
    #     )
    # )
    # mask_move_dst[idxs_edge_bd_bulk] = np.isin(cut_edges[idxs_edge_bd_bulk, 0], sim.nodes_boundary)

    tomove = np.where(mask_move_dst, cut_edges[:, 1], cut_edges[:, 0])
    tomove = np.unique(tomove)

    return tomove


def move_along_edges(sim: Simulation_2D, node, strain_map: dict, boundary, edges_moved: dict):

    neighbours = sim.n2n_map[sim.n2n_st[node]: sim.n2n_st[node + 1]]
    neighbours = np.intersect1d(neighbours, np.fromiter(strain_map.keys(), dtype=int))

    if boundary:
        neighbours = np.intersect1d(neighbours, sim.nodes_boundary)

    f_node = strain_map[node]
    f_neighbours = np.array([strain_map[neigh] for neigh in neighbours])
    opposite_neigh = neighbours[(f_neighbours > 0.) != (f_node > 0.)]
    if opposite_neigh.size == 0:
        print(f"cannot move node {node:d}!")
        return sim.coords[node]

    f_opposite = np.array([strain_map[neigh] for neigh in opposite_neigh])
    dist = np.linalg.norm(sim.coords[opposite_neigh] - sim.coords[node], axis=1)
    closest = np.argmin((f_opposite - f_node) / dist * np.sign(f_node))
    target = opposite_neigh[closest]
    alpha = f_node / (f_node - strain_map[target])

    # print(node, target)
    # print("\t", edges_moved.get(node, -1))
    # print("\t", edges_moved.get(target, -1))
    # print("")
    if target in edges_moved.get(node, []):
        return sim.coords[node]
    else:
        edges_moved.setdefault(target, []).append(node)
        return (1. - alpha) * sim.coords[node] + alpha * sim.coords[target]


def move_front(sim: Simulation_2D, interface_nodes, rec_strains_map):
    nodes = np.fromiter(rec_strains_map.keys(), dtype=int)
    values = np.fromiter(rec_strains_map.values(), dtype=float)

    edges_moved = {}

    # _, nodes_ind, nodes_inv = np.unique(nodes, return_index=True, return_inverse=True)

    elems = [sim.n2e_map[sim.n2e_st[node]: sim.n2e_st[node + 1]] for node in interface_nodes]
    edges = sim.get_edges(np.unique(np.concatenate(elems)))

    nodes_unchanged = np.setdiff1d(edges.flatten(), nodes)
    edges = edges[~np.any(np.isin(edges, nodes_unchanged), axis=1)]

    tomove = select_nodes(sim, edges, rec_strains_map)
    boundary_mask = np.isin(tomove, sim.nodes_boundary)
    new_coords = np.empty((tomove.size, 2))

    for idx, (node, flag_boundary) in enumerate(zip(tomove, boundary_mask)):
        new_coords[idx] = move_along_edges(sim, node, rec_strains_map, flag_boundary, edges_moved)

    return tomove, new_coords


def solve_interface_tracking(sim: Simulation_2D, max_it=5, tol_delta=1.e-3, deg: int = 1, strong=False):

    # Solve first time with initial mesh
    u_num, p_num, t_num = solve_FE_mosek(sim, strong=strong)
    # plot_1D_slice(u_num, sim, extra_name="2D_init")
    init_coords = np.copy(sim.coords)
    moved_once = np.array([], dtype=int)

    while (sim.iteration < max_it) and (sim.tau_zero > 0.):

        interface_nodes = find_nodes_interface(sim, t_num)
        _, strain_rec, coefs = reconstruct(sim, t_num, interface_nodes, deg)
        nodes_to_move, new_coords = move_front(sim, interface_nodes, strain_rec)
        moved_once = np.union1d(moved_once, nodes_to_move)
        nodes_to_relax = np.setdiff1d(moved_once, nodes_to_move)

        delta = np.linalg.norm(sim.coords[nodes_to_move] - new_coords, axis=1)

        rm = np.setdiff1d(interface_nodes, nodes_to_move).size
        ad = np.setdiff1d(nodes_to_move, interface_nodes).size
        print(f"Interface location - predictor (Gauss points) : {interface_nodes.size:d} nodes")
        print(f"Interface location - corrector (Linear apprx) : {rm:d} removed // {ad:d} added")
        print(f"Maximum node displacement : {np.amax(delta):.2e}")
        print(sim.coords[nodes_to_move[np.argmax(delta)]])

        extra = [interface_nodes, strain_rec, coefs, nodes_to_move, new_coords]
        plot_solution_2D(u_num, p_num, t_num, sim, extra)

        # Check if nodes moved enough to require new 'fem solve'
        # res_input = input("Iteratate again to improve the mesh ? [y/n]\n")
        # if (np.amax(delta) < tol_delta) or (res_input == 'n'):
        if (np.amax(delta) < tol_delta):
            break

        # Change node positions, jacobians...
        update_mesh(sim, nodes_to_move, new_coords, nodes_to_relax, init_coords[nodes_to_relax])

        # Solve the problem, slightly modified
        u_num, p_num, t_num = solve_FE_mosek(sim, strong=False)
        # sim.tol_yield = min(2 * sim.tol_yield, 1.e-4)

        sim.iteration += 1

    return u_num, p_num, t_num
