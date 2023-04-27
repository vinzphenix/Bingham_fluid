from bingham_structure import *
from bingham_tracking import compute_strain_per_elem, compute_gradient_at_nodes


def plot_1D_slice(u_num, sim: Simulation_2D):

    if sim.model_name[:4] not in ["rect", "test"]:
        return

    slice_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=4)
    slice_node_tags = np.array(slice_node_tags).astype(int) - 1

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


def add_streamlines(sim: Simulation_2D):
    # n_time_steps = 1000
    # dic = {
    #     'View': 0, 'X0': 1., 'Y0': -0.5, 'X1': 1.2, 'Y1': -0.3,
    #     'DT': 1., 'NumPointsU': 30, 'MaxIter': 1000
    # }
    
    option_names = ['View', 'X0', 'Y0', 'X1', 'Y1', 'DT', 'NumPointsU', 'MaxIter']
    
    model_name = sim.model_name.split("_")[0]
    if model_name == "bfs":
        option_values_list = [
            [0, 0., 0., 0., 1., 0.2, 20, 500],
            [0, 1., -0.5, 1.1, -0.4, 2., 5, 500],
            [0, 1.005, -0.05, 1.06, -0.05, 0.5, 5, 500]
        ]
    elif model_name == "cavity":
        option_values_list = [
            [0, 0.5, -0.25, 0.5, -0.9, 0.1, 20, 400],
            # [0, 1., -1., 0.97, -0.97, .5, 5, 5000],
        ]
    elif model_name in ["cylinder", "rectangle"]:
        option_values_list = [
            [0, 0., -0.95, 0., 0.95, 0.1, 30, 400],
        ]
    else:
        return
    
    tags_streamlines = []
    for option_values in option_values_list:
        for option_name, option_value in zip(option_names, option_values):
            gmsh.plugin.set_number('StreamLines', option=option_name, value=option_value)
        tag_streamlines = gmsh.plugin.run('StreamLines')
        gmsh.view.option.set_number(tag_streamlines, "TimeStep", -1)
        gmsh.view.option.set_number(tag_streamlines, "ShowScale", 0)
        tags_streamlines += [tag_streamlines]

    return tags_streamlines


def add_unstrained_zone(sim: Simulation_2D, strain_norm_avg, model_name, tol=1e-3):
    mask = strain_norm_avg <= tol
    percentile_95 = -np.partition(-strain_norm_avg, sim.n_elem//20)[sim.n_elem//20]
    tag_unstrained = gmsh.view.add("Unstrained_zone", tag=-1)

    if np.any(mask):
        gmsh.view.addHomogeneousModelData(
            tag_unstrained, 0, model_name, "ElementData", 
            sim.elem_tags[mask], strain_norm_avg[mask], numComponents=1
        )
        gmsh.view.option.setNumber(tag_unstrained, "RangeType", 2)
        gmsh.view.option.setNumber(tag_unstrained, "CustomMin", 0.)
        gmsh.view.option.setNumber(tag_unstrained, "CustomMax", percentile_95)
        gmsh.view.option.set_number(tag_unstrained, "ShowScale", 0)

    return [tag_unstrained]


def add_velocity_views(sim: Simulation_2D, u_num, strain_tensor, strain_norm, model_name):

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

    tag_v = gmsh.view.add("Velocity", tag=1)
    tag_strain = gmsh.view.add("Strain tensor", tag=2)
    tag_vorticity = gmsh.view.add("Vorticity", tag=3)
    tag_divergence = gmsh.view.add("Divergence", tag=4)
    tag_strain_norm_avg = gmsh.view.add("Strain norm averaged", tag=5)
    tags = [tag_v, tag_strain, tag_vorticity, tag_divergence, tag_strain_norm_avg]

    gmsh.view.addHomogeneousModelData(
        tag_v, 0, model_name, "NodeData", 
        sim.node_tags + 1, velocity, numComponents=3
    )
    gmsh.view.addHomogeneousModelData(
        tag_strain, 0, model_name, "ElementNodeData", 
        sim.elem_tags, strain_tensor, numComponents=9
    )
    gmsh.view.addHomogeneousModelData(
        tag_vorticity, 0, model_name, "ElementNodeData", 
        sim.elem_tags, vorticity, numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_divergence, 0, model_name, "ElementNodeData",
        sim.elem_tags, divergence, numComponents=1
    )
    gmsh.view.addHomogeneousModelData(
        tag_strain_norm_avg, 0, model_name, "ElementData", 
    sim.elem_tags, strain_norm, numComponents=1
    )

    v_normal_raise = 0.5 / np.amax(np.hypot(u_num[:, 0], u_num[:, 1]))
    strain_normal_raise = 0.5 / np.amax(strain_norm)
    gmsh.view.option.setNumber(tag_v, "VectorType", 6)
    gmsh.view.option.setNumber(tag_v, "DrawLines", 0)
    gmsh.view.option.setNumber(tag_v, "DrawPoints", 0)
    gmsh.view.option.setNumber(tag_v, "Sampling", 5)
    gmsh.view.option.setNumber(tag_v, "NormalRaise", v_normal_raise)
    gmsh.view.option.setNumber(tag_strain_norm_avg, "NormalRaise", strain_normal_raise)
    # for tag in [tag_v, tag_strain, tag_vorticity, tag_divergence]:
    #     gmsh.view.option.setNumber(tag, "AdaptVisualizationGrid", 1)
    #     gmsh.view.option.setNumber(tag, "TargetError", -0.0001)
    #     gmsh.view.option.setNumber(tag, "MaxRecursionLevel", 2)

    return tags


def add_pressure_view(sim: Simulation_2D, p_num, model_name):
    
    weak_pressure_nodes = np.setdiff1d(sim.primary_nodes, sim.nodes_singular_p)
    tag_pressure = gmsh.view.add("Pressure", tag=-1)

    if p_num.size == weak_pressure_nodes.size:  # weak incompressibility condition
        gmsh.view.addHomogeneousModelData(
            tag_pressure, 0, model_name, "NodeData",
            weak_pressure_nodes + 1, p_num, numComponents=1
        )
    else:  # strong incompressibility condition -> avg p of each gauss pt over the elem
        # deg = (sim.degree - 1) * 2
        # uvw = gmsh.model.mesh.getIntegrationPoints(2, "Gauss" + str(deg))
        p_avg = np.mean(p_num.reshape((sim.n_elem, sim.ng_loc)), axis=1)
        gmsh.view.addHomogeneousModelData(
            tag_pressure, 0, model_name, "ElementData",
            sim.elem_tags, p_avg, numComponents=1
        )


    return [tag_pressure]


def plot_solution_2D(u_num, p_num, sim: Simulation_2D):

    gmsh.fltk.initialize()
    model_name = gmsh.model.list()[0]
    # tag_psi = gmsh.view.add("psi")
    # show P1 basis
    # for j, idx_node in enumerate(sim.primary_nodes):
    #     data = np.zeros(sim.primary_nodes.size)
    #     if idx_node not in np.r_[sim.nodes_zero_u, sim.nodes_zero_v, sim.nodes_with_u]:
    #         data[j] = 1.
    #     gmsh.view.addHomogeneousModelData(
    #         tag_psi, j, model_name, "NodeData",
    #         sim.primary_nodes + 1, data, time=j, numComponents=1
    #     )

    strain_tensor = np.zeros((sim.n_elem, sim.n_local_node, 9))
    strain_norm = np.zeros(sim.n_elem)
    compute_strain_per_elem(sim, u_num, strain_norm)  # filled with |D| (cst / elem)
    compute_gradient_at_nodes(sim, u_num, strain_tensor)  # filled with grad(v) matrix

    if sim.degree == 3:  # remove bubble stuff
        sim.dv_shape_functions_at_v = sim.dv_shape_functions_at_v[:, :-1, :]
        sim.elem_node_tags = sim.elem_node_tags[:, :-1]
        u_num = u_num[:sim.n_node]

    tags_velocities = add_velocity_views(sim, u_num, strain_tensor, strain_norm, model_name)
    tags_unstrained = add_unstrained_zone(sim, strain_norm, model_name, tol=1e-3)
    tags_pressure = add_pressure_view(sim, p_num, model_name)
    tags_steamlines = add_streamlines(sim)

    for tag in tags_velocities[1:] + tags_pressure + tags_unstrained:
        gmsh.view.option.setNumber(tag, "Visible", 0)

    gmsh.option.set_number("Mesh.SurfaceEdges", 0)
    gmsh.option.set_number("Mesh.Lines", 1)
    gmsh.option.set_number("Mesh.Points", 0)
    gmsh.option.set_number("Mesh.ColorCarousel", 0)

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