from bingham_structure import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from analyze_pipe import save_profiles


def plot_1D_slice(u_num, sim: Simulation_2D, extra_name=""):

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

    info = sim.get_edge_node_tags('')
    edge_node_tags, length, tangent, normal = info
    gn = np.zeros(edge_node_tags.shape)
    sim.eval_gn(sim.coords[edge_node_tags], gn)
    dpdx = sim.f[0] + (np.amax(gn) - np.amin(gn)) / 2.  # to modify if channel length changed

    params = dict(H=H, K=sim.K, tau_zero=sim.tau_zero, f=dpdx, degree=deg_along_edge,
                  n_elem=n_intervals, random_seed=-1, fix_interface=False,
                  save=True, plot_density=25, dimensions=False)

    sim_1D = Simulation_1D(params)
    sim_1D.set_y(slice_y)
    plot_solution_1D(sim=sim_1D, u_nodes=slice_u, extra_name=extra_name)
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
        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        # don't evaluate at bubble node 
        for j, idx_node in enumerate(idx_local_nodes[:sim.n_local_node]):  
            dphi = dsf_at_nodes[j, :, :]  # dphi in reference element
            dphi = np.dot(dphi, inv_jac) / det  # dphi in physical element
            l11, l12, l21, l22 = eval_velocity_gradient(u_num[idx_local_nodes], dphi)
            velocity_gradient[i, j, np.array([0, 1, 3, 4])] = np.array([l11, l12, l21, l22])
    return


def get_stress_boundary(sim: Simulation_2D, strain_full):
    """
    Compute the stress tensor on the boundary, and where tau_zero < tau
    using the evaluation of the strain tensor at those nodes.
    Generates a ListData for gmsh: a force vector at each node of every boundary line segment
    """

    edge_node_tags, length, tangent, normal = sim.get_edge_node_tags("", exclude=[5, 6])
    mid_x = 0.5 * (np.amin(sim.coords[:, 0]) + np.amax(sim.coords[:, 0]))
    mask =(sim.coords[edge_node_tags, 0] - mid_x) ** 2 + sim.coords[edge_node_tags, 1] ** 2 < 1.
    mask = np.all(mask, axis=1)
    edge_node_tags, length, tangent, normal = edge_node_tags[mask], length[mask], tangent[mask], normal[mask]
    n_edge, n_pts = edge_node_tags.shape

    nodes_bd = np.unique(edge_node_tags[:, :])  # primary nodes only
    shear_force = np.zeros((n_edge, n_pts, 2))
    data = []

    # Loop over the edges on the boundary
    simpson = np.array([1., 4., 1.]) / 6.
    resultant = np.array([0., 0.])
    for i, (org, dst) in enumerate(edge_node_tags[:, :2]):
        # locate the element on which is the current edge (using neighbours: fast)
        elems_org = sim.n2e_map[sim.n2e_st[org]: sim.n2e_st[org + 1]]
        elems_dst = sim.n2e_map[sim.n2e_st[dst]: sim.n2e_st[dst + 1]]
        elem = np.intersect1d(elems_org, elems_dst)[0]  # always one and only one
        edge_nb = np.argwhere(sim.elem_node_tags[elem, :3] == org)[0, 0]

        # Mask that indicates which nodes of the element are on the boundary edge
        nodes_on_edge = np.in1d(sim.elem_node_tags[elem, :sim.n_local_node], edge_node_tags[i])
        indices = np.arange(sim.n_local_node)[nodes_on_edge]
        for j, elem_node_idx in enumerate(indices):
            local_strain = strain_full[elem, elem_node_idx, [[0, 1], [3, 4]]]
            local_norm = 2. * local_strain[0, 0]**2 + 2. * \
                local_strain[1, 1]**2 + 4. * local_strain[1, 0]**2

            # Shear is indefinite when the fluid is unyielded
            local_strain = np.where(
                local_norm > sim.tol_yield,
                (sim.K + sim.tau_zero / local_norm) * 2. * local_strain,
                0. * local_strain
            )
            shear_force[i, j] = np.dot(normal[i], local_strain)
            resultant[:] += simpson[j] * shear_force[i, j] * length[i]

    data = np.zeros((n_edge, 2 + 2 + 2 + 3 + 3 + 3 * (n_pts == 3)))
    data[:, 0] = sim.coords[edge_node_tags[:, 0], 0]  # x coord node 1
    data[:, 1] = sim.coords[edge_node_tags[:, 1], 0]  # x coord node 2
    data[:, 2] = sim.coords[edge_node_tags[:, 0], 1]  # y coord node 1
    data[:, 3] = sim.coords[edge_node_tags[:, 1], 1]  # y coord node 2
    data[:, [6, 7, 9, 10]] = np.c_[shear_force[:, 0, :], shear_force[:, 1, :]]
    if n_pts == 3:  # line with 3 nodes (vertex - inner - vertex)
        data[:, [12, 13]] = shear_force[:, 2, :]
    else:  # n_pts = 2
        pass

    print(f"Force resultant = ({resultant[0]:.3e}, {resultant[1]:.3e})")
    with open("../cylinder_force.txt", "a") as f:
        l = np.amax(sim.coords[:, 0]) - np.amin(sim.coords[:, 0])
        h = np.amax(sim.coords[:, 1]) - np.amin(sim.coords[:, 1])
        f.write(f"{l:.3e} {h:.3e} {resultant[0]:.3e} {resultant[1]:.3e}\n")

    return n_edge, data


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

    # The streamfunction is defined up to a constant
    # -> impose psi=0 at some arbitray location
    mass_matrices[rows == 0] = 0.
    mass_matrices[(rows == 0) & (cols == 0)] = 1.

    n_node = sim.n_node + sim.n_elem * (sim.element == "mini")
    mass_matrix = csr_matrix((mass_matrices, (rows, cols)), shape=(n_node, n_node))

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
    rhs_vorticity = csr_matrix((source, (rows, cols)), shape=(n_node, 1))
    rhs_vorticity = rhs_vorticity.toarray().flatten()

    #############################  -  Boundary source  -  #############################

    edge_node_tags, length, tangent, normal = sim.get_edge_node_tags("", exclude=[5, 6])
    edge_v = u_num[edge_node_tags]
    source_boundary = np.einsum(
        'g,ijn,gj,in,gk,i->ik',
        sim.weights_edge, edge_v, sim.sf_edge, -tangent, sim.sf_edge, length / 2.
    )  # size (ne, nsf)

    source_boundary = source_boundary.flatten()
    rows = edge_node_tags.flatten()
    cols = np.zeros(edge_node_tags.size, dtype=int)
    rhs_boundary = csr_matrix((source_boundary, (rows, cols)), shape=(n_node, 1))
    rhs_boundary = rhs_boundary.toarray().flatten()

    #############################  -  System solve  -  #############################
    rhs = rhs_vorticity + rhs_boundary
    rhs[0] = 0.
    psi = spsolve(mass_matrix, rhs)

    # Nice node renumbering
    # from scipy.sparse.csgraph import reverse_cuthill_mckee
    # perm = reverse_cuthill_mckee(mass_matrix, symmetric_mode=True)
    # plt.spy(mass_matrix.todense()[np.ix_(perm, perm)])
    # plt.show()

    return psi[:sim.n_node]


def add_streamfunction(sim: Simulation_2D, u_num):

    psi = compute_streamfunction(sim, u_num)

    tag_psi = gmsh.view.add("Streamfunction", tag=sim.tag)
    sim.tag += 1
    gmsh.view.addHomogeneousModelData(
        tag_psi, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, psi, numComponents=1
    )
    gmsh.view.option.set_number(tag_psi, "NbIso", 20)
    gmsh.view.option.set_number(tag_psi, "IntervalsType", 0)
    gmsh.view.option.set_number(tag_psi, "ShowScale", 0)
    gmsh.view.option.set_number(tag_psi, "LineWidth", 1.5)
    gmsh.view.option.set_number(tag_psi, "ColormapAlpha", 0.75)
    gmsh.view.option.set_number(tag_psi, "ColormapNumber", 0)
    gmsh.view.option.set_number(tag_psi, "ColormapInvert", 1)

    return 0*[tag_psi]


def add_unstrained_zone(sim: Simulation_2D, t_num):
    if sim.tau_zero < 1.e-10:
        return []

    # mask = np.all(t_num < sim.tol_yield, axis=1)
    mask = np.where(~sim.is_yielded(t_num))
    strain_norm_avg = np.mean(t_num, axis=1)
    percentile_95 = -np.partition(-strain_norm_avg, sim.n_elem // 20)[sim.n_elem // 20]
    tag_unstrained = gmsh.view.add("Unstrained_zone", tag=sim.tag)
    sim.tag += 1

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
    velocity = np.c_[u_num[:sim.n_node], np.zeros(sim.n_node)].flatten()
    strain_norm = strain_norm.flatten()

    # Compute strain-rate matrix by symmetrizing grad(v)
    strain_tensor[:, :, 1] = 0.5 * (strain_tensor[:, :, 1] + strain_tensor[:, :, 3])
    strain_tensor[:, :, 3] = strain_tensor[:, :, 1]

    n_edge, data_bd_shear = get_stress_boundary(sim, strain_tensor)

    tag_velocity = gmsh.view.add("Velocity", tag=sim.tag)

    tag_u = gmsh.view.add("Velocity - x", tag=sim.tag + 1)
    tag_v = gmsh.view.add("Velocity - y", tag=sim.tag + 2)

    tag_strain_xx = gmsh.view.add("Strain xx", tag=sim.tag + 3)
    tag_strain_xy = gmsh.view.add("Strain xy", tag=sim.tag + 4)
    tag_strain_yy = gmsh.view.add("Strain yy", tag=sim.tag + 5)

    tag_strain = gmsh.view.add("Strain rate tensor"*0, tag=sim.tag + 6)
    tag_strain_norm_avg = gmsh.view.add("Strain norm avg", tag=sim.tag + 7)
    tag_vorticity = gmsh.view.add("Vorticity", tag=sim.tag + 8)
    tag_divergence = gmsh.view.add("Divergence", tag=sim.tag + 9)

    tag_bd_shear = gmsh.view.add("Shear force", tag=sim.tag + 10)

    tags = [
        tag_velocity, tag_u, tag_v, tag_strain_xx, tag_strain_xy, tag_strain_yy,
        tag_strain, tag_strain_norm_avg, 
        tag_vorticity, tag_divergence, tag_bd_shear
    ]
    sim.tag += 11

    gmsh.view.addHomogeneousModelData(
        tag_velocity, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, velocity, numComponents=3
    )

    gmsh.view.addHomogeneousModelData(
        tag_u, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, u_num[:, 0], numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_v, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, u_num[:, 1], numComponents=1
    )

    gmsh.view.addHomogeneousModelData(
        tag_strain_xx, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, strain_tensor[:, :, 0].flatten(), numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_strain_xy, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, strain_tensor[:, :, 1].flatten(), numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_strain_yy, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, strain_tensor[:, :, 4].flatten(), numComponents=1
    )

    # Rescale because of 3/2 factor in Von Mises expression
    # Von Mises is equivalent up to a factor sqrt(3) to sqrt(1/2 D:D) because trace(D) = 0
    # or to sqrt(3)/2 wrt sqrt(2 D:D) = ||2D||
    gmsh.view.addHomogeneousModelData(
        tag_strain, 0, sim.model_name, "ElementNodeData",
        sim.elem_tags, 2. / np.sqrt(3) * strain_tensor.flatten(), numComponents=9
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

    gmsh.view.addListData(tag_bd_shear, "VL", n_edge, data_bd_shear.flatten())
    gmsh.view.option.setNumber(tag_bd_shear, "VectorType", 2)
    gmsh.view.option.setNumber(tag_bd_shear, "Sampling", 2)
    gmsh.view.option.setNumber(tag_bd_shear, "LineWidth", 4.)
    gmsh.view.option.setNumber(tag_bd_shear, "ColormapNumber", 23)
    gmsh.view.option.setNumber(tag_bd_shear, "ArrowSizeMax", 60)
    # gmsh.view.option.setNumber(tag_bd_shear, "RangeType", 2)
    # gmsh.view.option.setNumber(tag_bd_shear, "CustomMin", 0.)
    # gmsh.view.option.setNumber(tag_bd_shear, "CustomMax", 0.5)

    v_normal_raise = 0.5 / np.amax(np.hypot(u_num[:, 0], u_num[:, 1]))
    strain_normal_raise = 0.5 / np.amax(strain_norm)
    gmsh.view.option.setNumber(tag_velocity, "VectorType", 6)
    gmsh.view.option.setNumber(tag_velocity, "DrawLines", 0)
    gmsh.view.option.setNumber(tag_velocity, "DrawPoints", 0)
    gmsh.view.option.setNumber(tag_velocity, "NormalRaise", v_normal_raise)
    gmsh.view.option.setNumber(tag_velocity, "ArrowSizeMax", 50)
    gmsh.view.option.setNumber(tag_velocity, "LineWidth", 0.7)
    gmsh.view.option.setNumber(tag_velocity, "PointSize", 2.5)
    gmsh.view.option.setNumber(tag_velocity, "ShowScale", 1)
    gmsh.view.option.setNumber(tag_velocity, "Sampling", 1)

    for tag in [tag_u, tag_v]:
        gmsh.view.option.setNumber(tag, "ShowScale", 1)
        gmsh.view.option.setNumber(tag, "DrawTriangles", 0)
        gmsh.view.option.setNumber(tag, "NormalRaise", 10. / 6.)
        # gmsh.view.option.setNumber(tag, "ShowScale", 0)

    gmsh.view.option.setNumber(tag_strain_norm_avg, "NormalRaise", strain_normal_raise)
    for tag in [tag_strain_xx, tag_strain_xy, tag_strain_yy]:
        gmsh.view.option.setNumber(tag, "ColormapAlpha", 0.75)
        gmsh.view.option.setNumber(tag, "RangeType", 2)
        gmsh.view.option.setNumber(tag, "CustomMin", -0.25)
        gmsh.view.option.setNumber(tag, "CustomMax", 0.25)

    gmsh.view.option.setNumber(tag_strain, "RangeType", 2)
    gmsh.view.option.setNumber(tag_strain, "CustomMin", 1e-11)
    gmsh.view.option.setNumber(tag_strain, "CustomMax", 1e+1)
    gmsh.view.option.setNumber(tag_strain, "IntervalsType", 3)
    gmsh.view.option.setNumber(tag_strain, "NbIso", 12)
    gmsh.view.option.setNumber(tag_strain, "ColormapAlpha", 1.)
    gmsh.view.option.setNumber(tag_strain, "ColormapNumber", 23)
    gmsh.view.option.setNumber(tag_strain, "ScaleType", 2)
    gmsh.view.option.setString(tag_strain, "Format", r"%.0e")

    return tags


def add_pressure_view(sim: Simulation_2D, p_num):
    if p_num.size == 0:
        return []
    if sim.model_name in ["cavity", "opencavity", "cylinder"]:
        # n_values_keep = p_num.size * 19 // 20
        # p_small_partition = np.argpartition(p_num, n_values_keep)
        # p_large_partition = np.argpartition(-p_num, n_values_keep)
        # p_intmd_partition = np.intersect1d(p_small_partition, p_large_partition)
        # p_intmd_partition = p_num[p_intmd_partition]
        # pmax = p_num[p_small_partition[n_values_keep]]
        # pmin = p_num[p_large_partition[n_values_keep]]
        # p_num -= np.mean(p_intmd_partition)
        # pmax = max(abs(pmax), abs(pmin))

        pmax, pmin = np.amax(p_num), np.amin(p_num)
        p_avg = np.mean(p_num)
        pmax = max(pmax - p_avg, p_avg - pmin)
        p_num -= p_avg
    else:
        p_num -= np.amin(p_num)

    weak_pressure_nodes = np.setdiff1d(sim.primary_nodes, sim.nodes_singular_p)
    tag_pressure = gmsh.view.add("Pressure", tag=sim.tag)
    sim.tag += 1

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

    gmsh.view.option.setNumber(tag_pressure, "ColormapNumber", 24)
    gmsh.view.option.setNumber(tag_pressure, "IntervalsType", 3)
    if sim.model_name in ["cavity", "opencavity", "cylinder"]:
        pmax = np.amax(np.abs([pmax, pmin]))
        gmsh.view.option.setNumber(tag_pressure, "RangeType", 2)
        gmsh.view.option.setNumber(tag_pressure, "CustomMin", -pmax)
        gmsh.view.option.setNumber(tag_pressure, "CustomMax", pmax)

    return [tag_pressure]


def add_reconstruction(sim: Simulation_2D, extra):
    interface_nodes, strain_rec, coefs_matrix, moved_nodes, new_coords = extra

    view_rec = gmsh.view.add("Reconstructed", sim.tag)
    sim.tag += 1
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
    view_avg_rec = gmsh.view.add("Avg reconstr.", sim.tag)
    sim.tag += 1
    gmsh.view.addHomogeneousModelData(
        view_avg_rec, 0, sim.model_name, "NodeData",
        nodes_rec, value_rec, numComponents=1
    )
    gmsh.view.option.setNumber(view_avg_rec, 'IntervalsType', 1)
    gmsh.view.option.setNumber(view_avg_rec, 'NbIso', 1)
    gmsh.view.option.setNumber(view_avg_rec, 'RangeType', 2)
    gmsh.view.option.setNumber(view_avg_rec, "CustomMin", 0.)
    gmsh.view.option.setNumber(view_avg_rec, "CustomMax", 0.)
    gmsh.view.option.setNumber(view_avg_rec, "LineWidth", 2.)
    gmsh.view.option.setNumber(view_avg_rec, "ColormapNumber", 21)
    gmsh.view.option.setNumber(view_avg_rec, "ShowScale", 0)

    # gmsh.view.option.setNumber(view_avg_rec, 'PointSize', 7.5)
    # gmsh.view.option.setNumber(view_avg_rec, 'Boundary', 3)
    # gmsh.view.option.setNumber(view_avg_rec, 'GlyphLocation', 2)

    #########################

    n = moved_nodes.size
    data_2 = np.zeros((n, 3 + 3 + 2))
    data_2[:, [0, 2]] = sim.coords[moved_nodes]
    data_2[:, [1, 3]] = new_coords
    data_2[:, 7] = np.linalg.norm(sim.coords[moved_nodes] - new_coords, axis=1)
    view_targets = gmsh.view.add("Targets", tag=sim.tag)
    sim.tag += 1
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

    view_gauss_1 = gmsh.view.add("Gauss", tag=sim.tag)
    sim.tag += 1
    gmsh.view.addListData(tag=view_gauss_1, dataType="SP", numEle=sim.ng_all - counter, data=data)
    gmsh.view.option.setNumber(view_gauss_1, 'RaiseZ', 1.)
    gmsh.view.option.setNumber(view_gauss_1, 'PointSize', 5.)
    gmsh.view.option.setNumber(view_gauss_1, 'ColormapNumber', 11)

    view_gauss_2 = gmsh.view.add("Gauss_zero", tag=sim.tag)
    sim.tag += 1
    gmsh.view.addListData(tag=view_gauss_2, dataType="SP", numEle=counter, data=data_zero)
    gmsh.view.option.setNumber(view_gauss_2, 'PointSize', 5.)
    gmsh.view.option.setNumber(view_gauss_2, 'ColormapNumber', 9)

    return [view_gauss_1, view_gauss_2]


def add_exact_interface(sim: Simulation_2D):
    if (sim.tau_zero < sim.tol_yield) or (sim.model_name not in ["rectangle", "rectanglerot"]):
        return

    view_interface = gmsh.view.add("Exact interface", tag=sim.tag)
    sim.tag += 1

    beta = np.pi / 6. * (sim.model_name == "rectanglerot")
    rot_matrix = np.array([
        [np.cos(beta), -np.sin(beta)],
        [np.sin(beta), np.cos(beta)]]
    )
    length = 2.  # hardcoded, but (max-min) not valid when rotated
    yi = sim.tau_zero / 1.  # hardcoded, bc should otherwise compute dp/dx when Neumann
    pts = np.c_[[0., yi], [length, yi], [0., -yi], [length, -yi]]
    pts = np.dot(rot_matrix, pts).T
    data = [
        pts[0, 0], pts[1, 0], pts[0, 1], pts[1, 1], 0., 0., 0., 0.,
        pts[2, 0], pts[3, 0], pts[2, 1], pts[3, 1], 0., 0., 0., 0.,
    ]
    gmsh.view.addListData(tag=view_interface, dataType="SL", numEle=2, data=data)
    gmsh.view.option.setNumber(view_interface, 'ShowScale', 0)
    gmsh.view.option.setNumber(view_interface, 'LineWidth', 5.)
    gmsh.view.option.setNumber(view_interface, 'ColormapAlpha', 0.75)
    gmsh.view.option.setNumber(view_interface, 'ColormapNumber', 23)

    return [view_interface]


def plot_solution_2D(u_num, p_num, t_num, sim: Simulation_2D, extra=None):

    gmsh.fltk.initialize()

    strain_norm = np.mean(t_num, axis=1)  # filled with |2D| (cst / elem)
    strain_norm = np.sqrt(strain_norm) if sim.tau_zero < sim.tol_yield else strain_norm
    strain_tensor = np.zeros((sim.n_elem, sim.n_local_node, 9))
    compute_gradient_at_nodes(sim, u_num, strain_tensor)  # filled with grad(v) matrix

    sim.tag = 1
    tags_velocities = add_velocity_views(sim, u_num, strain_tensor, strain_norm)
    tags_unstrained = add_unstrained_zone(sim, t_num)
    tags_pressure = add_pressure_view(sim, p_num)
    tags_gauss = add_gauss_points(sim, t_num)
    # tags_steamlines = add_streamlines(sim, u_num, tags_velocities[0])
    tags_steamlines = add_streamfunction(sim, u_num)

    tags_invisible = tags_velocities[:] + tags_pressure + tags_gauss + tags_steamlines

    if extra is not None:
        tags_reconstructed = add_reconstruction(sim, extra)
        tag_all_steps = gmsh.view.addAlias(tags_reconstructed[0], copyOptions=True, tag=sim.tag)
        gmsh.view.remove(tags_reconstructed[0])
        tags_invisible += [tag_all_steps]
        sim.tag += 1

    tags_exact_interface = add_exact_interface(sim)

    for tag in tags_invisible:
        gmsh.view.option.setNumber(tag, "Visible", 0)

    gmsh.option.set_number("Mesh.SurfaceEdges", 0)
    gmsh.option.set_number("Mesh.Lines", 1)
    gmsh.option.set_number("Mesh.Points", 0)
    gmsh.option.set_number("Mesh.ColorCarousel", 2)
    gmsh.option.set_number("Mesh.NodeLabels", 0)
    gmsh.option.set_number("Geometry.Points", 0)
    gmsh.option.set_number("General.GraphicsFontSize", 24)

    # if extra is None:  # used to show pipe velocity profile
    #     tags = np.array(tags_velocities)[[0, 6, 8]]  # velocity, strain, vorticity
    #     save_profiles(tags, "../figures/pipe_profiles", n_pts=150)

    gmsh.fltk.run()
    gmsh.fltk.finalize()

    return
