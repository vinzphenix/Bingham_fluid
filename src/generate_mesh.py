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


def create_split_rectangle(filename, width, height, elemSizeRatio, y_zero=0., cut=False, angle=0.):
    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh

    # width, height = 1., 2.
    lc = elemSizeRatio * height

    fit = 0. < y_zero < height / 2.
    h = y_zero if fit else height / 2.
    c = width / 2. if cut else width

    # Geometry
    # points
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

    p1 = factory.addPoint(*coords[:, 0], 0., lc)
    p2 = factory.addPoint(*coords[:, 1], 0., lc)
    p3 = factory.addPoint(*coords[:, 2], 0., lc)
    p4 = factory.addPoint(*coords[:, 3], 0., lc)
    pts_no_slip = [p1, p2, p3, p4]
    pts_inflow = [p1, p4]
    pts_outflow = [p2, p3]

    # lines
    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3)
    l3 = factory.addLine(p3, p4)
    l4 = factory.addLine(p4, p1)
    lines = [l1, l2, l3, l4]
    ln_inflow = [l4]
    ln_outflow = [l2]
    ln_no_slip = [l1, l3]

    # curve loops
    cl1 = factory.addCurveLoop([l1, l2, l3, l4])

    # surfaces
    s1 = factory.addPlaneSurface([cl1])
    srfs = [s1]

    p5, p6, p7, p8, p9, p10, p11, p12 = [-1] * 8
    ldr, lur = [-1] * 2

    if fit:
        p5 = factory.addPoint(*coords[:, 4], 0., lc)
        p6 = factory.addPoint(*coords[:, 5], 0., lc)
        p7 = factory.addPoint(*coords[:, 6], 0., lc)
        p8 = factory.addPoint(*coords[:, 7], 0., lc)
        pts_no_slip = [p5, p6, p7, p8]
        pts_inflow += [p5, p8]
        pts_outflow += [p6, p7]

        ldl = factory.addLine(p1, p5)
        ldd = factory.addLine(p5, p6)
        ldr = factory.addLine(p6, p2, tag=202)
        lur = factory.addLine(p3, p7, tag=201)
        luu = factory.addLine(p7, p8)
        lul = factory.addLine(p8, p4)
        lines += [ldl, ldd, ldr, lur, luu, lul]
        ln_inflow += [ldl, lul]
        ln_outflow += [ldr, lur]
        ln_no_slip = [ldd, luu]

        cl2 = factory.addCurveLoop([ldl, ldd, ldr, -l1])
        cl3 = factory.addCurveLoop([lur, luu, lul, -l3])

        s2 = factory.addPlaneSurface([cl2])
        s3 = factory.addPlaneSurface([cl3])
        srfs += [s2, s3]

    pts_cut = pts_outflow
    ln_cut = ln_outflow
    if cut:
        p9 = factory.addPoint(*coords[:, 8], 0., lc)
        p10 = factory.addPoint(*coords[:, 9], 0., lc)
        pts_no_slip += [] if fit else [p9, p10]
        pts_outflow = [p9, p10]

        ldw = factory.addLine(p2, p9)
        lrg = factory.addLine(p9, p10)
        lup = factory.addLine(p10, p3)
        lines += [ldw, lrg, lup]
        ln_outflow = [lrg]
        ln_no_slip += [] if fit else [ldw, lup]

        cl4 = factory.addCurveLoop([-l2, ldw, lrg, lup])
        s4 = factory.addPlaneSurface([cl4])
        srfs += [s4]

        if fit:
            p11 = factory.addPoint(*coords[:, 10], 0., lc)
            p12 = factory.addPoint(*coords[:, 11], 0., lc)
            pts_no_slip += [p11, p12]
            pts_outflow += [p11, p12]

            ldd_ = factory.addLine(p6, p11)
            ldr_ = factory.addLine(p11, p9)
            lur_ = factory.addLine(p10, p12)
            luu_ = factory.addLine(p12, p7)
            lines += [ldd_, ldr_, lur_, luu_]
            ln_outflow += [ldr_, lur_]
            ln_no_slip += [ldd_, luu_]

            cl5 = factory.addCurveLoop([-ldr, ldd_, ldr_, -ldw])
            cl6 = factory.addCurveLoop([-lup, lur_, luu_, -lur])
            s5 = factory.addPlaneSurface([cl5])
            s6 = factory.addPlaneSurface([cl6])
            srfs += [s5, s6]

    factory.synchronize()

    # Physical groups for boundary conditions
    ln_others = np.setdiff1d(lines, ln_cut + ln_inflow + ln_outflow + ln_no_slip)

    lines_bd = ln_no_slip + ln_inflow + ln_outflow
    lines_set_vn = ln_no_slip
    lines_set_vt = ln_no_slip + ln_inflow + ln_outflow
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(1, ln_cut, tag=5, name="cut")
    gmsh.model.addPhysicalGroup(1, ln_others, tag=6, name="others")

    tag_bulk_2d = gmsh.model.addPhysicalGroup(2, srfs, tag=-1, name="bulk")

    # Meshing

    # for li, length in zip(lines, [width, height, width, height]):
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


def create_cylinder(filename, elemSizeRatio, radial=False, sharp=False):
    gmsh.initialize()
    gmsh.model.add("cylinder")

    width, height = 3., 2.
    radius = height / 5. if radial else height / 10.
    bottom = 0. if radial else -height / 2.

    rect = gmsh.model.occ.add_rectangle(0., bottom, 0., width, height, 0)
    if sharp:
        tool = gmsh.model.occ.add_rectangle(
            width / 2. - radius / 2., -radius, 0., radius, 2. * radius, 1000
        )
    else:
        tool = gmsh.model.occ.add_disk(width / 2., 0., 0., radius, radius, 1000)
    res_cut = gmsh.model.occ.cut([(2, rect)], [(2, tool)])

    gmsh.model.occ.synchronize()

    ln_inflow, ln_outflow, ln_tool, ln_lateral = [], [], [], []

    if radial:
        if sharp:
            pts_rect = [1, 2, 3, 4]
            pts_tool = [5, 6, 7, 8]
            ln_inflow, ln_outflow, ln_tool, ln_lateral = [1], [3], [5, 6, 7], [2, 4, 8]
        else:
            pts_rect = [3, 4, 5, 6]
            pts_tool = [1, 2]
            ln_inflow, ln_outflow, ln_tool, ln_lateral = [3], [5], [1], [2, 4, 6]
    else:
        if sharp:
            pts_rect = [1, 2, 3, 4]
            pts_tool = [5, 6, 7, 8]
            ln_inflow, ln_outflow, ln_tool, ln_lateral = [2], [3], [5, 6, 7, 8], [1, 4]
        else:
            pts_rect = [1, 2, 3, 4]
            pts_tool = [5]
            ln_inflow, ln_outflow, ln_tool, ln_lateral = [2], [3], [5], [1, 4]

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

    for pt in pts_rect:
        gmsh.model.mesh.setSize([(0, pt)], elemSizeRatio * height)
    for pt in pts_tool:
        gmsh.model.mesh.setSize([(0, pt)], elemSizeRatio * height * 0.2)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    # gmsh.model.mesh.setOrder(2)
    # gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    display_elem_edge_node()
    gmsh.fltk.run()
    gmsh.finalize()


def create_cavity(filename, elemSizeRatio, cut=True, size_field=False):
    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh

    width, height = 1., 1.
    lc = elemSizeRatio * width
    refinement_factor_surface = 1. if size_field else 10.

    c = width / 2. if cut else width

    p1 = factory.addPoint(0., 0., 0., lc / refinement_factor_surface)
    p2 = factory.addPoint(0., -height, 0., lc)
    p3 = factory.addPoint(width, -height, 0., lc)
    p4 = factory.addPoint(width, 0., 0., lc / refinement_factor_surface)
    # pts_u_zero = [p2, p3, p1, p4]  # p1, p4 corners --> can be zero / one
    # pts_v_zero = [p1, p2, p3, p4]
    # pts_u_one = [p1, p4]
    # pts_cut = []
    # lines_u_zero, lines_v_zero, lines_u_one, lines_cut = [], [], [], []
    lengths, lines, lines_bd, lines_cut = [], [], [], []

    if cut:
        p5 = factory.addPoint(c, -height, 0., lc)
        p6 = factory.addPoint(c, 0., 0., lc / refinement_factor_surface)
        # pts_u_zero += [p5]
        # pts_v_zero += [p5, p6]
        # pts_u_one += [p6]
        # pts_cut += [p5, p6]

        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p5)
        l3 = factory.addLine(p5, p3)
        l4 = factory.addLine(p3, p4)
        l5 = factory.addLine(p4, p6)
        l6 = factory.addLine(p6, p1)
        l7 = factory.addLine(p5, p6)
        # lines_u_zero += [l1, l2, l3, l4]
        # lines_v_zero += [l1, l2, l3, l4, l5, l6]
        # lines_u_one += [l5, l6]
        lengths += [height, width / 2., width / 2., height, width / 2., width / 2., height]
        lines += [l1, l2, l3, l4, l5, l6, l7]
        lines_bd += [l1, l2, l3, l4, l5, l6]
        lines_cut += [l7]

        cl1 = factory.addCurveLoop([l1, l2, l7, l6])
        cl2 = factory.addCurveLoop([l3, l4, l5, -l7])
        s1 = factory.addPlaneSurface([cl1])
        s2 = factory.addPlaneSurface([cl2])
        srfs = [s1, s2]

    else:
        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p3)
        l3 = factory.addLine(p3, p4)
        l4 = factory.addLine(p4, p1)
        # lines_u_zero += [l1, l2, l3]
        # lines_v_zero += [l1, l2, l3, l4]
        # lines_u_one += [l4]
        lengths += [height, width, height, width]
        lines += [l1, l2, l3, l4]
        lines_bd += [l1, l2, l3, l4]

        cl1 = factory.addCurveLoop([l1, l2, l3, l4])
        s1 = factory.addPlaneSurface([cl1])
        srfs = [s1]

    factory.synchronize()

    if (size_field) and (not cut):
        # Set mesh size field
        p_solid_lf = factory.addPoint(0.3, -0.28, 0.)
        p_solid_rg = factory.addPoint(0.7, -0.28, 0.)
        diag_lf = factory.addLine(p1, p_solid_lf)
        diag_rg = factory.addLine(p4, p_solid_rg)
        line_solid = factory.addLine(p_solid_lf, p_solid_rg)
        factory.synchronize()

        gmsh.model.mesh.field.add("Distance", tag=1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", [diag_lf, diag_rg, l4])
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

        gmsh.model.mesh.field.add("Distance", tag=2)
        gmsh.model.mesh.field.setNumbers(2, "PointsList", [p1, p4])
        gmsh.model.mesh.field.setNumbers(2, "CurvesList", [line_solid])
        gmsh.model.mesh.field.setNumber(2, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 3)
        gmsh.model.mesh.field.setNumber(3, "InField", 1)
        gmsh.model.mesh.field.setNumber(3, "SizeMin", lc / 12)
        gmsh.model.mesh.field.setNumber(3, "SizeMax", lc)
        gmsh.model.mesh.field.setNumber(3, "DistMin", 0.05)
        gmsh.model.mesh.field.setNumber(3, "DistMax", 0.10)

        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "InField", 2)
        gmsh.model.mesh.field.setNumber(4, "SizeMin", lc / 8)
        gmsh.model.mesh.field.setNumber(4, "SizeMax", lc)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 0.10)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 0.30)

        gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [3, 4])
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        gmsh.option.setNumber("Mesh.Algorithm", 5)


    # Physical groups for boundary conditions

    lines_set_vn = lines_bd
    lines_set_vt = lines_bd
    lines_set_gn = np.setdiff1d(lines_bd, lines_set_vn)  # []
    lines_set_gt = np.setdiff1d(lines_bd, lines_set_vt)  # []

    gmsh.model.addPhysicalGroup(0, [p1, p4], tag=5, name="singular")

    gmsh.model.addPhysicalGroup(1, lines_set_vn, tag=1, name="setNormalFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_vt, tag=2, name="setTangentFlow")
    gmsh.model.addPhysicalGroup(1, lines_set_gn, tag=3, name="setNormalForce")
    gmsh.model.addPhysicalGroup(1, lines_set_gt, tag=4, name="setTangentForce")

    gmsh.model.addPhysicalGroup(1, lines_cut, tag=5, name="cut")

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
    p6 = factory.addPoint(width, 0., 0., lc / refinement_factor_surface)
    pts_u_zero = [p2, p3, p4, p5]  # p1, p4 corners --> can be zero / one
    pts_v_zero = [p1, p2, p3, p4, p5, p6]
    pts_u_one = [p1, p6]
    pts_cut = []
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
        [l1+l2, 0.],
        [l1+l2, width]
    ])
    print(coords)
    center = np.array([l1, -radius])[None, :]
    coords = center + np.dot(rot_matrix, (coords - center).T).T

    mesh_size = elemSizeRatio * width
    mesh_size_fine = mesh_size / 2.

    p1 = factory.addPoint(0., 0., 0., mesh_size)
    p2 = factory.addPoint(l1, 0., 0., mesh_size_fine)
    p3 = factory.addPoint(l1, width, 0., mesh_size_fine)
    p4 = factory.addPoint(0., width, 0., mesh_size)
    p5 = factory.addPoint(*coords[0], 0., mesh_size_fine)
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

    lines = [c1, c2, c3, c4, c5, c6, c7, c8]
    cl = factory.addCurveLoop(lines)
    s1 = factory.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    ln_in_out = [4, 8]
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


if __name__ == "__main__":
    path_to_dir = "../mesh/"

    # create_split_rectangle(path_to_dir + "rectangle.msh", width=3., height=2.,
    #                        elemSizeRatio=1. / 20., y_zero=0.2, cut=False)
    # create_split_rectangle(path_to_dir + "rectanglerot.msh", width=2., height=1.,
    #                        elemSizeRatio=1. / 20., y_zero=0.0, cut=False, angle=np.pi / 6.)

    # create_cylinder(path_to_dir + "cylinder.msh", elemSizeRatio=1./30., radial=False, sharp=False)
    create_cavity(path_to_dir + "cavity.msh", elemSizeRatio=1./12., cut=False, size_field=True)

    # create_open_cavity(path_to_dir + "opencavity.msh", elemSizeRatio=1./35.)
    # create_backward_facing_step(path_to_dir + "bfs.msh", elemSizeRatio=1./25.)
    # create_pipe(path_to_dir + "pipe.msh", 1./20., l1=2., l2=1., width=1., radius=0.5, theta=120.)
