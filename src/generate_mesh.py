import sys
import numpy as np
import gmsh


def display_elem_edge_node():
    _, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    gmsh.model.mesh.create_edges()
    edge_tags, _ = gmsh.model.mesh.getAllEdges()
    print("N elem = ", len(elem_tags[0]))
    print("N node = ", len(node_tags))
    print("N edge = ", len(edge_tags))
    return


def create_rectangle(
    filename, width, height, elemSizeRatio, size_field=False,
    y_zero=0., angle=0., fit=False, cut=False
):

    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh

    lc = elemSizeRatio * height
    h = y_zero if lc < y_zero < height / 2. - lc else height / 4.
    c = width / 2.

    # Geometry
    angle = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    coords = np.array([
        [0., -h],
        [c, -h],
        [c, +h],
        [0., +h],
        [0., -height / 2.],
        [+c, -height / 2.],
        [+c, +height / 2.],
        [0., +height / 2.],
        [width, -h],
        [width, +h],
        [width, -height / 2.],
        [width, +height / 2.]
    ]).T
    coords = np.dot(rot_matrix, coords)

    point_tags = []
    surfc_tags = []
    pt_2_ln_map = {}
    phys_2_ln_map = dict(inflow=[], outflow=[], noslip=[], inside=[], cut=[])

    line_pt_map = dict(
        noslip=[(5, 6), (6, 11), (12, 7), (7, 8)],
        inflow=[(8, 4), (4, 1), (1, 5)],
        outflow=[(11, 9), (9, 10), (10, 12)],
        cut=[(6, 2), (2, 3), (3, 7)],              # remove for bad bc 
        inside=[(1, 2), (2, 9), (10, 3), (3, 4)],  # remove for bad bc
    )

    if cut and fit:
        curve_loops = [
            (1, 2, 3, 4), (1, 5, 6, 2), (3, 7, 8, 4),
            (3, 2, 9, 10), (2, 6, 11, 9), (3, 10, 12, 7),
        ]
    elif (not cut) and fit:
        curve_loops = [
            (1, 5, 6, 11, 9, 2), (1, 2, 9, 10, 3, 4), (4, 3, 10, 12, 7, 8)
        ]
    elif cut and (not fit):
        curve_loops = [
            (1, 5, 6, 2, 3, 7, 8, 4), (2, 6, 11, 9, 10, 12, 7, 3),
        ]
    else:
        curve_loops = [
            (1, 5, 6, 11, 9, 10, 12, 7, 8, 4)
        ]

    for label, lines in line_pt_map.items():
        for org, dst in lines:
            if org not in point_tags:
                factory.addPoint(*coords[:, org - 1], 0., lc, org)
                point_tags += [org]
            if dst not in point_tags:
                factory.addPoint(*coords[:, dst - 1], 0., lc, dst)
                point_tags += [dst]

            exists_fwd = pt_2_ln_map.get((org, dst), -1)
            exists_bwd = pt_2_ln_map.get((dst, org), -1)

            if (exists_fwd == -1) and (exists_bwd == -1):
                line_tag = factory.addLine(org, dst)
                pt_2_ln_map[(org, dst)] = line_tag
                phys_2_ln_map[label].append(line_tag)

    for pt_seq in curve_loops:
        this_curve_loop = []
        for pt, next_pt in zip(pt_seq, np.roll(pt_seq, -1)):
            exists_fwd = pt_2_ln_map.get((pt, next_pt), -1)
            exists_bwd = pt_2_ln_map.get((next_pt, pt), -1)
            if (exists_fwd != -1):
                this_curve_loop += [exists_fwd]
            else:
                this_curve_loop += [-exists_bwd]

        curve_loop_tag = factory.addCurveLoop(this_curve_loop)
        surfc_tags.append(factory.addPlaneSurface([curve_loop_tag]))

    factory.synchronize()

    # Physical groups for boundary conditions
    lines_cut = phys_2_ln_map["cut"] if cut else phys_2_ln_map["outflow"]
    lines_bd = phys_2_ln_map["noslip"] + phys_2_ln_map["inflow"] + phys_2_ln_map["outflow"]
    lines_set_vn = phys_2_ln_map["noslip"]
    lines_set_vt = phys_2_ln_map["noslip"] + phys_2_ln_map["inflow"] + phys_2_ln_map["outflow"] * 1
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")
    gmsh.model.addPhysicalGroup(1, lines_cut, tag=5, name="cut")
    # gmsh.model.addPhysicalGroup(1, phys_2_ln_map["inside"], tag=6, name="others")
    tag_bulk_2d = gmsh.model.addPhysicalGroup(2, surfc_tags, tag=-1, name="bulk")

    factory.synchronize()

    if size_field:
        gmsh.model.mesh.field.add("Distance", tag=1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", phys_2_ln_map["inside"])
        # gmsh.model.mesh.field.setNumbers(1, "CurvesList", phys_2_ln_map["outflow"])  # bad bc
        gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

        size_min, size_max, dist_min, dist_max = lc / 3., lc, height / 50., height / 20.
        # size_min, size_max, dist_min, dist_max = lc / 3., lc, width / 30., width / 4.  # bad bc
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", size_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", size_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    meshFact.generate(2)
    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()
    return


def create_cylinder(filename, elemSizeRatio, radial=False, sharp=False, multiple=False):

    if radial and multiple:
        print("Radial and multiple not possible at same time")
        return

    gmsh.initialize()
    gmsh.model.add("cylinder")

    radius = 0.5  # unitary diameter
    height = 7.5 * radius if radial else 10. * radius
    width = 20. * radius
    bottom = 0. if radial else -height / 2.

    rect = gmsh.model.occ.add_rectangle(0., bottom, 0., width, height, 0)

    if multiple:
        centers = [(0.4 * width, -0.5 * radius), (0.6 * width, +0.5 * radius)]
    else:
        centers = [(0.5 * width, 0.)]

    tools = []
    for center_x, center_y in centers:
        if sharp:
            args = (center_x - radius / 2., center_y - radius, 0., radius, 2. * radius)
            tools += [gmsh.model.occ.add_rectangle(*args)]
        else:
            tools += [gmsh.model.occ.add_disk(center_x, center_y, 0., radius, radius)]

    res_cut = gmsh.model.occ.cut([(2, rect)], [(2, tool) for tool in tools])

    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()

    ln_inflow, ln_outflow, ln_tool, ln_lateral = [], [], [], []

    if radial:
        if sharp:
            pts_rect = [1, 2, 3, 4]
            pts_tool = [5, 6, 7, 8]
            ln_tool = [5, 6, 7]
            ln_inflow, ln_outflow, ln_lateral = [1], [3], [2, 4, 8]
        else:
            pts_rect = [3, 4, 5, 6]
            pts_tool = [1, 2]
            ln_tool = [1]
            ln_inflow, ln_outflow, ln_lateral = [3], [5], [2, 4, 6]
    else:
        if sharp:
            pts_rect = [1, 2, 3, 4]
            pts_tool = [5, 6, 7, 8, 9, 10, 11, 12] if multiple else [5, 6, 7, 8]
            ln_tool = [5, 6, 7, 8, 9, 10, 11, 12] if multiple else [5, 6, 7, 8]
            ln_inflow, ln_outflow, ln_lateral = [2], [3], [1, 4]
        else:
            pts_rect = [1, 2, 3, 4]
            pts_tool = [5, 6] if multiple else [5]
            ln_tool = [5, 6] if multiple else [5]
            ln_inflow, ln_outflow, ln_lateral = [2], [3], [1, 4]

    # Size field
    lc = elemSizeRatio * height

    gmsh.model.mesh.field.add("Distance", tag=1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", ln_tool)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 8)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.30)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2.50)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    lines_bd = ln_inflow + ln_outflow + ln_tool + ln_lateral
    lines_set_vn = ln_inflow + ln_lateral + ln_tool
    lines_set_vt = ln_inflow + ln_outflow + ln_tool
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)  # []
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)  # []

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(1, ln_outflow, tag=5, name="cut")
    gmsh.model.addPhysicalGroup(2, [0], tag=-1, name="bulk")

    # for pt in pts_rect:
    #     gmsh.model.mesh.setSize([(0, pt)], elemSizeRatio * height)
    # for pt in pts_tool:
    #     gmsh.model.mesh.setSize([(0, pt)], elemSizeRatio * height * 0.1)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    # gmsh.model.mesh.setOrder(2)
    # gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()


def create_cavity(filename, elemSizeRatio, cut=True, size_field=False, cheat=False):
    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh

    width, height = 1., 1.
    lc = elemSizeRatio * width
    refinement_factor_surface = 1. if size_field else 4.

    p1 = factory.addPoint(0., 0., 0., lc / refinement_factor_surface)
    p2 = factory.addPoint(0., -height, 0., lc)
    p3 = factory.addPoint(width, -height, 0., lc)
    p4 = factory.addPoint(width, 0., 0., lc / refinement_factor_surface)

    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3)
    l3 = factory.addLine(p3, p4)
    l4 = factory.addLine(p4, p1)
    lengths = [height, width, height, width]
    lines = [l1, l2, l3, l4]
    lines_bd = [l1, l2, l3, l4]

    cl1 = factory.addCurveLoop([l1, l2, l3, l4])
    s1 = factory.addPlaneSurface([cl1])
    srfs = [s1]

    factory.synchronize()

    if size_field:
        fields = []

        gmsh.model.mesh.field.add("Distance", tag=1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", [l4])
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc / 4.)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", lc / 1.)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.10)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.50)

        gmsh.model.mesh.field.add("Distance", tag=3)
        gmsh.model.mesh.field.setNumbers(3, "PointsList", [p1, p4])
        gmsh.model.mesh.field.setNumber(3, "Sampling", 100)
        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "InField", 3)
        gmsh.model.mesh.field.setNumber(4, "SizeMin", lc / 6.)
        gmsh.model.mesh.field.setNumber(4, "SizeMax", lc / 1.)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 0.10)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 0.25)
        
        fields += [2, 4]
    
        if cheat:
            # Set mesh size field
            p_solid_lf = factory.addPoint(0., -0.1767, 0.)
            p_solid_rg = factory.addPoint(1., -0.1767, 0.)
            p_solid_center = factory.addPoint(0.5, +0.05, 0.)
            arc = factory.addCircleArc(p_solid_lf, p_solid_center, p_solid_rg)
            # p_center_top = factory.addPoint(0.50, 0., 0.)
            # diag_lf = factory.addLine(p1, p_solid_lf)
            # diag_rg = factory.addLine(p4, p_solid_rg)
            # line_solid = factory.addLine(p_solid_lf, p_solid_rg)
            factory.synchronize()

            gmsh.model.mesh.field.add("Distance", tag=5)
            gmsh.model.mesh.field.setNumbers(5, "CurvesList", [arc])
            gmsh.model.mesh.field.setNumber(5, "Sampling", 100)
            gmsh.model.mesh.field.add("Threshold", 6)
            gmsh.model.mesh.field.setNumber(6, "InField", 5)
            gmsh.model.mesh.field.setNumber(6, "SizeMin", lc / 3)
            gmsh.model.mesh.field.setNumber(6, "SizeMax", lc)
            gmsh.model.mesh.field.setNumber(6, "DistMin", 0.08)
            gmsh.model.mesh.field.setNumber(6, "DistMax", 0.15)
            
            fields += [6]

        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

    # Physical groups for boundary conditions

    lines_set_vn = lines_bd
    lines_set_vt = lines_bd
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)  # []
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)  # []

    gmsh.model.addPhysicalGroup(0, [p1, p4], tag=5, name="singular")
    # gmsh.model.addPhysicalGroup(0, [], tag=5, name="singular")

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(1, [], tag=5, name="cut")

    tag_bulk_2d = gmsh.model.addPhysicalGroup(2, srfs, tag=-1, name="bulk")

    # Meshing
    # for li, length in zip(lines[6:], lengths[6:]):
    #     n_nodes = int(np.ceil(length / lc))
    #     meshFact.setTransfiniteCurve(li, numNodes=n_nodes)
    # for si in srfs:
    #     meshFact.setTransfiniteSurface(si)

    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    meshFact.generate(2)
    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()
    return


def create_open_cavity(filename, elemSizeRatio):
    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh

    width, height = 1., 1.
    opening_height = height / 3.
    lc = elemSizeRatio * width
    refinement_factor_surface = 4.

    p1 = factory.addPoint(0., 0., 0., lc / refinement_factor_surface)
    p2 = factory.addPoint(0., -opening_height, 0., lc)
    p3 = factory.addPoint(0., -height, 0., lc)
    p4 = factory.addPoint(width, -height, 0., lc)
    p5 = factory.addPoint(width, -height + opening_height, 0., lc)
    p6 = factory.addPoint(width, 0., 0., 0.5 * lc / refinement_factor_surface)

    lines_u_zero, lines_v_zero, lines_u_one, lines_cut = [], [], [], []
    lines, lengths = [], []

    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3)
    l3 = factory.addLine(p3, p4)
    l4 = factory.addLine(p4, p5)
    l5 = factory.addLine(p5, p6)
    l6 = factory.addLine(p6, p1)
    lines_u_zero += [l2, l3, l5]
    lines_v_zero += [l2, l3, l5, l6]
    lines_u_one += [l6]
    lines += [l1, l2, l3, l4, l5, l6]
    lengths += [
        opening_height, height - opening_height, width,
        opening_height, height - opening_height, width
    ]

    cl1 = factory.addCurveLoop([l1, l2, l3, l4, l5, l6])
    s1 = factory.addPlaneSurface([cl1])
    srfs = [s1]

    factory.synchronize()

    lines_bd = [1, 2, 3, 4, 5, 6]
    lines_set_vn = [2, 3, 5, 6]
    lines_set_vt = [1, 2, 3, 4, 5, 6]
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)

    gmsh.model.addPhysicalGroup(0, [p5, p6], tag=5, name="singular")

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(2, [1], tag=-1, name="bulk")

    # Meshing
    # for li, length in zip(lines[6:], lengths[6:]):
    #     n_nodes = int(np.ceil(length / lc))
    #     meshFact.setTransfiniteCurve(li, numNodes=n_nodes)
    # for si in srfs:
    #     meshFact.setTransfiniteSurface(si)

    meshFact.generate(2)
    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()
    return


def create_backward_facing_step(filename, elemSizeRatio):
    """
    4 ----------- 5
    |             |
    3 ---- 2      |
           |      |
           1 ---- 6
    """
    gmsh.initialize()
    gmsh.model.add("bfs")

    width, height = 3., 1.  # width and height before step
    step_size = height / 2.
    step_loc = width / 3.

    rect = gmsh.model.occ.add_rectangle(0., -step_size, 0., width, height + step_size, 0)
    rect_to_remove = gmsh.model.occ.add_rectangle(0., -step_size, 0., step_loc, step_size, 1000)
    res_cut = gmsh.model.occ.cut([(2, rect)], [(2, rect_to_remove)])

    gmsh.model.occ.synchronize()

    lines_bd = [1, 2, 3, 4, 5, 6]
    lines_set_vn = [1, 2, 4, 6]
    lines_set_vt = [1, 2, 3, 4, 5, 6]
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(2, [0], tag=-1, name="bulk")

    # gmsh.model.mesh.set_size_callback(lambda *args: elemSizeRatio * height)
    gmsh.model.mesh.setSize([(0, 2), (0, 1)], elemSizeRatio * height / 2.)
    for i in [3, 4, 5, 6]:
        gmsh.model.mesh.setSize([(0, i)], elemSizeRatio * height)

    gmsh.model.occ.synchronize()
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()
    return


def create_pipe(filename, elemSizeRatio, l1, l2, width, radius, theta):
    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh
    gmsh.model.add("pipe")

    theta = min(theta, 180. - 1.e-10)
    r_in, r_out = radius, radius + width
    rot_matrix = np.array([
        [+np.cos(np.radians(theta)), +np.sin(np.radians(theta))],
        [-np.sin(np.radians(theta)), +np.cos(np.radians(theta))]
    ])
    coords = np.array([
        [l1, 0.],
        [l1, width],
        [l1 + l2, 0.],
        [l1 + l2, width]
    ])
    center = np.array([l1, -radius])[None, :]
    coords = center + np.dot(rot_matrix, (coords - center).T).T

    mesh_size = elemSizeRatio * width
    mesh_size_fine = mesh_size * 0.7
    mesh_size_very_fine = mesh_size * 0.6

    p1 = factory.addPoint(0., 0., 0., mesh_size)
    p2 = factory.addPoint(l1, 0., 0., mesh_size_very_fine)
    p3 = factory.addPoint(l1, width, 0., mesh_size_fine)
    p4 = factory.addPoint(0., width, 0., mesh_size)
    p5 = factory.addPoint(*coords[0], 0., mesh_size_very_fine)
    p6 = factory.addPoint(*coords[1], 0., mesh_size_fine)
    p7 = factory.addPoint(*coords[2], 0., mesh_size)
    p8 = factory.addPoint(*coords[3], 0., mesh_size)
    pc = factory.addPoint(*center[0], 0., mesh_size)

    c1 = factory.addLine(p1, p2)
    c2 = factory.addCircleArc(p2, pc, p5)
    c3 = factory.addLine(p5, p7)
    c4 = factory.addLine(p7, p8)
    c5 = factory.addLine(p8, p6)
    c6 = factory.addCircleArc(p6, pc, p3)
    c7 = factory.addLine(p3, p4)
    c8 = factory.addLine(p4, p1)

    c_cut1 = factory.addLine(p5, p6)
    c_cut2 = factory.addLine(p3, p2)

    if l1 > 0. and l2 > 0.:
        lines = [c1, c2, c3, c4, c5, c6, c7, c8]
        ln_in_out = [c8, c4]
    elif l1 > 0.:  # l2 == 0
        lines = [c1, c2, c_cut1, c6, c7, c8]
        ln_in_out = [c8, c_cut1]
    elif l2 > 0.:
        lines = [c2, c3, c4, c5, c6, c_cut2]
        ln_in_out = [c_cut2, c4]
    else:
        lines = [c2, c_cut1, c6, c_cut2]
        ln_in_out = [c_cut1, c_cut2]
    cl = factory.addCurveLoop(lines)
    s1 = factory.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Mesh refinement near yield transition
    gmsh.model.mesh.field.add("Distance", tag=1)
    pp_1 = factory.addPoint(2.20, 0.44, 0.)
    pp_2 = factory.addPoint(2.80, 0.35, 0.)
    ll = factory.addLine(pp_1, pp_2)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [pp_1, pp_2])
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [ll])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    # gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size * 0.33)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size * 0.10)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", width/30.)
    gmsh.model.mesh.field.setNumber(2, "DistMax", width/4.)

    # Mesh refinement near boundaries
    gmsh.model.mesh.field.add("Distance", tag=3)
    gmsh.model.mesh.field.setNumbers(3, "CurvesList", lines)
    gmsh.model.mesh.field.setNumber(3, "Sampling", 200)
    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", mesh_size * 0.66)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(4, "DistMin", width/10.)
    gmsh.model.mesh.field.setNumber(4, "DistMax", width/5.)
    
    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    lines_set_vn = np.setdiff1d(lines, ln_in_out)
    lines_set_vt = lines
    lines_set_gn = np.setdiff1d(lines, lines_set_vn)
    lines_set_gt = np.setdiff1d(lines, lines_set_vt)

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(2, [1], tag=-1, name="bulk")

    # gmsh.model.mesh.setSize([(2, 1)], elemSizeRatio * width)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()
    return


def create_pipe_contraction(filename, elemSizeRatio, l1, l2, w1, w2, delta, sharp=0):
    # sharp = 0 --> bezier curve
    # sharp = 1 --> oblique line
    # sharp = 2 --> vertical line

    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh
    lc = elemSizeRatio * min(w1, w2)

    if sharp == 2:
        l1 += delta / 2.
        l2 += delta / 2.
        delta = 0.
        n_nodes = int(np.ceil(4 * (max(w1, w2) / 2. - min(w1, w2) / 2.) / lc))
    else:
        n_nodes = int(np.ceil(4 * delta / lc))

    # Geometry
    p1 = factory.addPoint(0., -w1 / 2., 0.)
    p2 = factory.addPoint(l1, -w1 / 2., 0.)
    p3 = factory.addPoint(l1, +w1 / 2., 0.)
    p4 = factory.addPoint(0., +w1 / 2., 0.)
    p5 = factory.addPoint(l1 + delta + 0., -w2 / 2., 0.)
    p6 = factory.addPoint(l1 + delta + l2, -w2 / 2., 0.)
    p7 = factory.addPoint(l1 + delta + l2, +w2 / 2., 0.)
    p8 = factory.addPoint(l1 + delta + 0., +w2 / 2., 0.)
    q1 = factory.addPoint(l1 + delta / 3., -w1 / 2., 0.)
    q2 = factory.addPoint(l1 + delta / 3., -w2 / 2., 0.)
    q3 = factory.addPoint(l1 + delta / 3., +w1 / 2., 0.)
    q4 = factory.addPoint(l1 + delta / 3., +w2 / 2., 0.)

    if sharp == 0:
        links = np.array([
            factory.addBezier([p2, q1, q2, p5]),
            factory.addBezier([p8, q4, q3, p3]),
        ])
    else:
        links = np.array([
            factory.addLine(p2, p5),
            factory.addLine(p8, p3),
        ])

    lines_bd = np.array([
        factory.addLine(p1, p2),
        links[0],
        factory.addLine(p5, p6),
        factory.addLine(p6, p7),  # outflow
        factory.addLine(p7, p8),
        links[1],
        factory.addLine(p3, p4),
        factory.addLine(p4, p1),  # inflow
    ])

    cl1 = factory.addCurveLoop(lines_bd)
    s1 = factory.addPlaneSurface([cl1])
    srfs = [s1]

    factory.synchronize()

    ln_in_out = lines_bd[[3, 7]]
    lines_set_vn = np.setdiff1d(lines_bd, ln_in_out)
    lines_set_vt = lines_bd
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")
    gmsh.model.addPhysicalGroup(2, [1], tag=-1, name="bulk")
    gmsh.model.geo.synchronize()

    if sharp == 0:
        meshFact.setTransfiniteCurve(lines_bd[1], numNodes=n_nodes)
        meshFact.setTransfiniteCurve(lines_bd[5], numNodes=n_nodes)

    # Meshing
    gmsh.model.mesh.field.add("Distance", tag=1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", lines_bd)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.5 * lc)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 1. * lc)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0 * max(w1, w2))
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * max(w1, w2))
    if sharp == 2:
        gmsh.model.mesh.field.add("Distance", tag=3)
        gmsh.model.mesh.field.setNumbers(3, "PointsList", [p5, p8])
        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "InField", 3)
        gmsh.model.mesh.field.setNumber(4, "SizeMin", 0.15 * lc)
        gmsh.model.mesh.field.setNumber(4, "SizeMax", 1. * lc)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 0.5 * (w1 - w2) / 2.)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 1.5 * (w1 - w2) / 2.)
        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    meshFact.generate(2)
    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()
    return


if __name__ == "__main__":
    path_to_dir = "../mesh/"

    # create_rectangle(
    #     path_to_dir + "rectangle.msh", width=2., height=1., elemSizeRatio=1. / 28.,
    #     size_field=False, y_zero=0.35, angle=0., fit=False, cut=False,
    # )
    # create_rectangle(
    #     path_to_dir + "rectanglerot.msh", width=2., height=1.,
    #     elemSizeRatio=1. / 20., y_zero=0.0, cut=False, angle=np.pi / 6.
    # )

    # create_cylinder(
    #     path_to_dir + "cylinder_new.msh", elemSizeRatio=1./18.,
    #     radial=True, sharp=False, multiple=False
    # )

    # to keep for all simulations with increasing Bn
    # create_cavity(path_to_dir + "cavity.msh", elemSizeRatio=1./35., cut=False, size_field=False)

    create_cavity(path_to_dir + "cavity_cheat.msh", elemSizeRatio=1./35., size_field=True, cheat=True)
    # create_open_cavity(path_to_dir + "opencavity.msh", elemSizeRatio=1./35.)

    # create_backward_facing_step(path_to_dir + "bfs.msh", elemSizeRatio=1./25.)
    # create_pipe(path_to_dir + "pipe.msh", 1./28., l1=2.5, l2=0., width=1., radius=1., theta=90.)
    # create_pipe(path_to_dir + "pipe_dense.msh", 1./28., l1=2.5, l2=0., width=1., radius=1., theta=90.)

    # create_pipe_contraction(path_to_dir + "pipeneck.msh", 1./10., 1.5, 1.5, 1., 0.5, 0.5, sharp=0)
    # create_pipe_contraction(path_to_dir + "pipeneck.msh", 1./9., 1.5, 1.5, 1., 0.5, 0.5, sharp=2)