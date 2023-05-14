from bingham_structure import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def eval_velocity_gradient(u_local, dphi_local):
    dudx = np.dot(u_local[:, 0], dphi_local[:, 0])
    dudy = np.dot(u_local[:, 0], dphi_local[:, 1])
    dvdx = np.dot(u_local[:, 1], dphi_local[:, 0])
    dvdy = np.dot(u_local[:, 1], dphi_local[:, 1])
    return dudx, dudy, dvdx, dvdy


def compute_gradient_at_nodes(sim: Simulation_2D, u_num, velocity_gradient):
    velocity_gradient[:] = 0.
    elem_type, n_local_node, local_coords = sim.elem_type, sim.n_local_node, sim.local_node_coords
    _, dsf_at_nodes, _ = gmsh.model.mesh.getBasisFunctions(elem_type, local_coords, 'GradLagrange')
    dsf_at_nodes = np.array(dsf_at_nodes).reshape((n_local_node, n_local_node, 3))[:, :, :-1]

    if sim.element == "mini":
        # eval bubble derivatives at nodes
        xi_eta = local_coords.reshape(n_local_node, 3)[:, :-1].T
        bubble_dsf = np.c_[DPHI_DXI(*xi_eta), DPHI_DETA(*xi_eta)]
        dsf_at_nodes = np.append(dsf_at_nodes, bubble_dsf[:, np.newaxis, :], axis=1)

    for i in range(sim.n_elem):
        idx_local_nodes = sim.elem_node_tags[i, :]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        for j, idx_node in enumerate(idx_local_nodes[:3]):  # don't evaluate at bubble node
            dphi = dsf_at_nodes[j, :, :]  # dphi in reference element
            dphi = np.dot(dphi, inv_jac) / det  # dphi in physical element
            l11, l12, l21, l22 = eval_velocity_gradient(u_num[idx_local_nodes], dphi)
            velocity_gradient[i, j, np.array([0, 1, 3, 4])] = np.array([l11, l12, l21, l22])
    return


def plot_1D_slice(u_num, sim: Simulation_2D):

    if sim.model_name[:4] not in ["rect", "test"]:
        return

    # slice_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=4)
    slice_node_tags = sim.nodes_cut

    slice_xy = sim.coords[slice_node_tags]
    slice_y = slice_xy[:, 1]
    slice_u = u_num[slice_node_tags, 0]  # only u component of (u, v)
    H = np.amax(slice_y) - np.amin(slice_y)

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

    params = dict(H=H, K=sim.K, tau_zero=sim.tau_zero, f=sim.f[0], degree=deg_along_edge,
                  n_elem=n_intervals, random_seed=-1, fix_interface=False,
                  save=False, plot_density=25, dimensions=True)

    sim_1D = Simulation_1D(params)
    sim_1D.set_y(slice_y)
    plot_solution_1D(sim=sim_1D, u_nodes=slice_u)
    return


def compute_streamfunction(sim: Simulation_2D, u_num):

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
    # Pff... easier to deal with Dirichlet conditions
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

    # from scipy.sparse.csgraph import reverse_cuthill_mckee
    # perm = reverse_cuthill_mckee(mass_matrix, symmetric_mode=True)
    # plt.spy(mass_matrix.todense()[np.ix_(perm, perm)])
    # plt.show()

    return psi


def add_streamfunction(sim: Simulation_2D, u_num):

    psi = compute_streamfunction(sim, u_num)

    tag_psi = gmsh.view.add("Streamfunction", tag=-1)
    gmsh.view.addHomogeneousModelData(
        tag_psi, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, psi, numComponents=1
    )
    gmsh.view.option.set_number(tag_psi, "NbIso", 20)
    gmsh.view.option.set_number(tag_psi, "IntervalsType", 0)
    
    return [tag_psi]


def add_streamlines(sim: Simulation_2D, u_num, tag_v):

    # option_names = ['View', 'X0', 'Y0', 'X1', 'Y1', 'DT', 'NumPointsU', 'MaxIter']
    option_names = [
        'X0', 'Y0', 'X1', 'Y1', 'X2', 'Y2', 
        'NumPointsU', 'NumPointsV', 
        'DT', 'MaxIter'
    ]

    if sim.model_name == "bfs":
        option_values_list = [
            [0, 0., 0., 0., 1., 0.2, 20, 500],
            [0, 1., -0.5, 1.1, -0.4, 2., 5, 500],
            [0, 1.005, -0.05, 1.06, -0.05, 0.5, 5, 500]
        ]
    elif sim.model_name == "cavity":
        option_values_list = [
            [0, 0.5, -0.25, 0.5, -0.9, 0.1, 20, 400],
            # [0, 1., -1., 0.97, -0.97, .5, 5, 5000],
        ]
    elif sim.model_name == "opencavity":
        option_values_list = [
            # [0, 0., -0.01, 0., -1./3., 0.1, 20, 200],
            # [0, 0., -1./3., 0.4, -1./3., 0.1, 20, 400],
            [0.02, -0.02, 0.02, -0.98, 0.98, -0.02, 20, 20, 0.01, 100],
        ]
    elif sim.model_name in ["cylinder", "rectangle"]:
        option_values_list = [
            [0, 0., -0.95, 0., 0.95, 0.1, 30, 400],
        ]
    else:
        return

    tag_speed = gmsh.view.add("speed", tag=-1)
    gmsh.view.addHomogeneousModelData(
        tag_speed, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, np.linalg.norm(u_num, axis=1), numComponents=1
    )
    gmsh.view.option.set_number(tag_speed, "Visible", 0)

    tags_streamlines = []
    option_names += ['View', 'OtherView']

    for option_values in option_values_list:

        option_values += [tag_v - 1, tag_speed - 1]
        for option_name, option_value in zip(option_names, option_values):
            gmsh.plugin.set_number('StreamLines', option=option_name, value=option_value)

        tag_streamlines = gmsh.plugin.run('StreamLines')
        gmsh.view.option.set_number(tag_streamlines, "TimeStep", -1)
        gmsh.view.option.set_number(tag_streamlines, "ShowScale", 1)
        tags_streamlines += [tag_streamlines]

    return tags_streamlines


def add_unstrained_zone(sim: Simulation_2D, t_num):
    if sim.tau_zero < 1.e-10:
        return []

    # mask = np.all(t_num < sim.tol_yield, axis=1)
    mask = np.where(~sim.is_yielded(t_num))
    strain_norm_avg = np.mean(t_num, axis=1)
    percentile_95 = -np.partition(-strain_norm_avg, sim.n_elem // 20)[sim.n_elem // 20]
    tag_unstrained = gmsh.view.add("Unstrained_zone", tag=-1)

    if np.any(mask):
        gmsh.view.addHomogeneousModelData(
            tag_unstrained, 0, sim.model_name, "ElementData",
            sim.elem_tags[mask], strain_norm_avg[mask], numComponents=1
        )
        gmsh.view.option.setNumber(tag_unstrained, "RangeType", 2)
        gmsh.view.option.setNumber(tag_unstrained, "CustomMin", 0.)
        gmsh.view.option.setNumber(tag_unstrained, "CustomMax", percentile_95)
        gmsh.view.option.set_number(tag_unstrained, "ShowScale", 0)
        gmsh.view.option.set_number(tag_unstrained, "ColormapAlpha", 0.25)
        gmsh.view.option.set_number(tag_unstrained, "OffsetZ", -1.e-5)

    return [tag_unstrained]


def add_velocity_views(sim: Simulation_2D, u_num, strain_tensor, strain_norm):

    vorticity = (strain_tensor[:, :, 3] - strain_tensor[:, :, 1]).flatten()
    divergence = (strain_tensor[:, :, 0] + strain_tensor[:, :, 4]).flatten()
    velocity = np.c_[u_num, np.zeros(sim.n_node)].flatten()
    strain_norm = strain_norm.flatten()

    # Compute strain-rate matrix by symmetrizing grad(v)
    strain_tensor[:, :, 1] = 0.5 * (strain_tensor[:, :, 1] + strain_tensor[:, :, 3])
    strain_tensor[:, :, 3] = strain_tensor[:, :, 1]

    # Rescale because of 3/2 factor in Von Mises expression
    # Von Mises is equivalent up to a factor sqrt(3) to sqrt(1/2 D:D) because trace(D) = 0
    # or to sqrt(3)/2 wrt sqrt(2 D:D) = ||2D||
    strain_tensor = 2. / np.sqrt(3) * strain_tensor.flatten()

    tag_v = gmsh.view.add("Velocity", tag=-1)
    tag_strain = gmsh.view.add("Strain tensor", tag=-1)
    tag_strain_norm_avg = gmsh.view.add("Strain norm avg", tag=-1)
    tag_vorticity = gmsh.view.add("Vorticity", tag=-1)
    tag_divergence = gmsh.view.add("Divergence", tag=-1)
    tags = [tag_v, tag_strain, tag_strain_norm_avg, tag_vorticity, tag_divergence]

    gmsh.view.addHomogeneousModelData(
        tag_v, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, velocity, numComponents=3
    )
    gmsh.view.addHomogeneousModelData(
        tag_strain, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, strain_tensor, numComponents=9
    )
    gmsh.view.addHomogeneousModelData(
        tag_strain_norm_avg, 0, sim.model_name, "ElementData",
        sim.elem_tags, strain_norm, numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_vorticity, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, vorticity, numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_divergence, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, divergence, numComponents=1
    )

    v_normal_raise = 0.5 / np.amax(np.hypot(u_num[:, 0], u_num[:, 1]))
    strain_normal_raise = 0.5 / np.amax(strain_norm)
    gmsh.view.option.setNumber(tag_v, "VectorType", 6)
    gmsh.view.option.setNumber(tag_v, "DrawLines", 0)
    gmsh.view.option.setNumber(tag_v, "DrawPoints", 0)
    # gmsh.view.option.setNumber(tag_v, "Sampling", 5)
    # gmsh.view.option.setNumber(tag_v, "NormalRaise", v_normal_raise)
    gmsh.view.option.setNumber(tag_v, "ArrowSizeMax", 120)
    gmsh.view.option.setNumber(tag_strain_norm_avg, "NormalRaise", strain_normal_raise)
    # for tag in [tag_v, tag_strain, tag_vorticity, tag_divergence]:
    #     gmsh.view.option.setNumber(tag, "AdaptVisualizationGrid", 1)
    #     gmsh.view.option.setNumber(tag, "TargetError", -0.0001)
    #     gmsh.view.option.setNumber(tag, "MaxRecursionLevel", 2)

    return tags


def add_pressure_view(sim: Simulation_2D, p_num):

    weak_pressure_nodes = np.setdiff1d(sim.primary_nodes, sim.nodes_singular_p)
    tag_pressure = gmsh.view.add("Pressure", tag=-1)

    if p_num.size == weak_pressure_nodes.size:  # weak incompressibility condition
        gmsh.view.addHomogeneousModelData(
            tag_pressure, 0, sim.model_name, "NodeData",
            weak_pressure_nodes + 1, p_num, numComponents=1
        )
    else:  # strong incompressibility condition -> avg p of each gauss pt over the elem
        # deg = (sim.degree - 1) * 2
        # uvw = gmsh.model.mesh.getIntegrationPoints(2, "Gauss" + str(deg))
        p_avg = np.mean(p_num.reshape((sim.n_elem, sim.ng_loc)), axis=1)
        gmsh.view.addHomogeneousModelData(
            tag_pressure, 0, sim.model_name, "ElementData",
            sim.elem_tags, p_avg, numComponents=1
        )

    return [tag_pressure]


def add_reconstruction(sim: Simulation_2D, extra):
    interface_nodes, strain_rec, coefs_matrix, moved_nodes, new_coords = extra

    view_rec = gmsh.view.add("Reconstructed", -1)
    # gmsh.view.option.setNumber(view_rec, 'LineWidth', 3.)
    gmsh.view.option.setNumber(view_rec, 'NbIso', 11)
    gmsh.view.option.setNumber(view_rec, 'IntervalsType', 3)
    gmsh.view.option.setNumber(view_rec, 'RangeType', 3)
    gmsh.view.option.setNumber(view_rec, 'GlyphLocation', 2)
    # gmsh.view.option.setNumber(view_rec, 'CustomMin', -1.)
    # gmsh.view.option.setNumber(view_rec, 'CustomMax', 1.)
    # gmsh.view.option.setNumber(view_rec, 'ColormapNumber', 0)
    # gmsh.view.option.setNumber(view_rec, 'ShowScale', 0)
    gmsh.view.option.setNumber(view_rec, "ColormapAlpha", 0.75)
    # gmsh.view.option.setNumber(view_rec, "RaiseZ", 1.)

    for step, (node, coefs) in enumerate(zip(interface_nodes, coefs_matrix)):
        if np.linalg.norm(coefs) < 1.e-9:
            continue  # node were unable to reconstruct

        support_elems = sim.get_support_approx(node)
        support_nodes = [sim.elem_node_tags[elem] for elem in support_elems]
        support_nodes = np.unique(np.concatenate(support_nodes))

        # neighbours = sim.n2n_map[sim.n2n_st[node]: sim.n2n_st[node+1]]
        # diamond_nodes = np.r_[node, neighbours]

        data_3 = np.c_[np.ones(support_nodes.size), sim.coords[support_nodes]]
        if coefs.size == 6:
            data_3 = np.c_[
                data_3,
                sim.coords[support_nodes]**2,
                sim.coords[support_nodes, 0] * sim.coords[support_nodes, 1],
            ]

        data_3 = np.dot(data_3, coefs)
        gmsh.view.addHomogeneousModelData(
            view_rec, step, sim.model_name, "NodeData",
            support_nodes + 1, data_3, numComponents=1
        )

    #########################

    nodes_rec = np.fromiter(strain_rec.keys(), dtype='int') + 1
    value_rec = np.fromiter(strain_rec.values(), dtype='float')
    view_avg_rec = gmsh.view.add("Avg reconstr.", -1)
    gmsh.view.addHomogeneousModelData(
        view_avg_rec, 0, sim.sim.model_name, "NodeData",
        nodes_rec, value_rec, numComponents=1
    )
    gmsh.view.option.setNumber(view_avg_rec, 'IntervalsType', 1)
    gmsh.view.option.setNumber(view_avg_rec, 'NbIso', 1)
    gmsh.view.option.setNumber(view_avg_rec, 'RangeType', 2)
    gmsh.view.option.setNumber(view_avg_rec, "CustomMin", 0.)
    gmsh.view.option.setNumber(view_avg_rec, "CustomMax", 0.)
    gmsh.view.option.setNumber(view_avg_rec, "LineWidth", 10.)

    # gmsh.view.option.setNumber(view_avg_rec, 'PointSize', 7.5)
    # gmsh.view.option.setNumber(view_avg_rec, 'Boundary', 3)
    # gmsh.view.option.setNumber(view_avg_rec, 'GlyphLocation', 2)

    #########################

    n = moved_nodes.size
    data_2 = np.zeros((n, 3 + 3 + 2))
    data_2[:, [0, 2]] = sim.coords[moved_nodes]
    data_2[:, [1, 3]] = new_coords
    data_2[:, 7] = np.linalg.norm(sim.coords[moved_nodes] - new_coords, axis=1)
    view_targets = gmsh.view.add("Targets", tag=-1)
    gmsh.view.addListData(tag=view_targets, dataType="SL", numEle=n, data=data_2.flatten())
    gmsh.view.option.setNumber(view_targets, 'LineWidth', 3.)
    gmsh.view.option.setNumber(view_targets, 'ShowScale', 0)

    return [view_rec, view_avg_rec, view_targets]


def add_gauss_points(sim: Simulation_2D, t_num):
    data, data_zero = [], []
    counter = 0
    uvw, _ = gmsh.model.mesh.getIntegrationPoints(2, "Gauss2")
    xi_eta_list = np.array(uvw).reshape(3, 3)[:, :-1]
    for i in range(sim.n_elem):
        local_nodes = sim.elem_node_tags[i, :3]
        node_coords = sim.coords[local_nodes]
        for g, (xi, eta) in enumerate(xi_eta_list):
            gauss_coords = np.dot(np.array([1. - xi - eta, xi, eta]), node_coords)
            if t_num[i, g] < sim.tol_yield:
                data_zero += [*gauss_coords, 0., t_num[i, g]]
                counter += 1
            else:
                data += [*gauss_coords, 0., t_num[i, g]]

    view_gauss_1 = gmsh.view.add("Gauss", tag=-1)
    gmsh.view.addListData(tag=view_gauss_1, dataType="SP", numEle=sim.ng_all - counter, data=data)
    gmsh.view.option.setNumber(view_gauss_1, 'RaiseZ', 1.)
    gmsh.view.option.setNumber(view_gauss_1, 'PointSize', 5.)
    gmsh.view.option.setNumber(view_gauss_1, 'ColormapNumber', 11)

    view_gauss_2 = gmsh.view.add("Gauss_zero", tag=-1)
    gmsh.view.addListData(tag=view_gauss_2, dataType="SP", numEle=counter, data=data_zero)
    gmsh.view.option.setNumber(view_gauss_2, 'PointSize', 5.)
    gmsh.view.option.setNumber(view_gauss_2, 'ColormapNumber', 9)

    return [view_gauss_1, view_gauss_2]


def plot_solution_2D(u_num, p_num, t_num, sim: Simulation_2D, extra=None, run=True):

    gmsh.fltk.initialize()

    strain_norm = np.mean(t_num, axis=1)  # filled with |2D| (cst / elem)
    strain_tensor = np.zeros((sim.n_elem, sim.n_local_node, 9))
    compute_gradient_at_nodes(sim, u_num, strain_tensor)  # filled with grad(v) matrix

    if sim.degree == 3:  # remove bubble stuff
        sim.dv_shape_functions_at_v = sim.dv_shape_functions_at_v[:, :-1, :]
        sim.elem_node_tags = sim.elem_node_tags[:, :-1]
        u_num = u_num[:sim.n_node]

    tags_velocities = add_velocity_views(sim, u_num, strain_tensor, strain_norm)
    tags_unstrained = add_unstrained_zone(sim, t_num)
    tags_pressure = add_pressure_view(sim, p_num)
    tags_gauss = add_gauss_points(sim, t_num)
    # tags_steamlines = add_streamlines(sim, u_num, tags_velocities[0])
    tags_steamlines = add_streamfunction(sim, u_num)

    tags_invisible = tags_velocities[1:] + tags_pressure + tags_gauss

    if extra is not None:
        tags_reconstructed = add_reconstruction(sim, extra)
        tag_all_steps = gmsh.view.addAlias(tags_reconstructed[0], copyOptions=True, tag=-1)
        gmsh.view.remove(tags_reconstructed[0])
        tags_invisible += [tag_all_steps]

    for tag in tags_invisible:
        gmsh.view.option.setNumber(tag, "Visible", 0)

    gmsh.option.set_number("Mesh.SurfaceEdges", 0)
    gmsh.option.set_number("Mesh.Lines", 1)
    gmsh.option.set_number("Mesh.Points", 0)
    # gmsh.option.set_number("Mesh.ColorCarousel", 0)
    gmsh.option.set_number("Mesh.NodeLabels", 0)

    # for yi in [-sim.tau_zero/sim.f[0], sim.tau_zero/sim.f[0]]:
    #     gmsh.plugin.set_string('CutParametric', option="X", value="u")
    #     gmsh.plugin.set_string('CutParametric', option="Y", value=f"{yi}")
    #     gmsh.plugin.set_string('CutParametric', option="Z", value=f"{0}")
    #     gmsh.plugin.set_number('CutParametric', option="MaxU", value=np.amax(sim.coords[:, 0]))
    #     gmsh.plugin.set_number('CutParametric', option="View", value=1)
    #     gmsh.plugin.run('CutParametric')

    if run:
        gmsh.fltk.run()

    gmsh.fltk.finalize()
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
