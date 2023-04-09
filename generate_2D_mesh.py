import sys
import numpy as np
import gmsh

def create_split_rectangle(filename, elemSizeRatio, y_zero=0., cut=True):
    gmsh.initialize()
    factory = gmsh.model.geo
    meshFact = gmsh.model.mesh

    width, height = 3., 2.
    lc = elemSizeRatio * width

    fit = 0. < y_zero < height / 2.
    h = y_zero if fit else height / 2.
    c = width / 2. if cut else width

    # Geometry
    # points
    p1 = factory.addPoint(0., -h, 0., lc)
    p2 = factory.addPoint(+c, -h, 0., lc)
    p3 = factory.addPoint(+c, +h, 0., lc)
    p4 = factory.addPoint(0., +h, 0., lc)
    pts_no_slip = [p1, p2, p3, p4]
    pts_inflow = [p1, p4]
    pts_outflow = [p2, p3]

    # lines
    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3, tag=100)
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

    if fit:
        p5 = factory.addPoint(0., -height / 2., 0., lc)
        p6 = factory.addPoint(+c, -height / 2., 0., lc)
        p7 = factory.addPoint(+c, +height / 2., 0., lc)
        p8 = factory.addPoint(0., +height / 2., 0., lc)
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
        p9 = factory.addPoint(width, -h, 0., lc)
        p10 = factory.addPoint(width, +h, 0., lc)
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
            p11 = factory.addPoint(width, -height / 2., 0., lc)
            p12 = factory.addPoint(width, +height / 2., 0., lc)
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

    tag_pts_cut = gmsh.model.addPhysicalGroup(0, pts_cut, tag=4, name="cut")
    tag_pts_cut = gmsh.model.addPhysicalGroup(0, pts_outflow, tag=2, name="outflow")
    tag_pts_no_slip = gmsh.model.addPhysicalGroup(0, pts_inflow, tag=1, name="inflow")
    tag_pts_no_slip = gmsh.model.addPhysicalGroup(0, pts_no_slip, tag=3, name="no-slip")
    
    tag_other = gmsh.model.addPhysicalGroup(1, ln_others, tag=5, name="others")
    tag_inflow = gmsh.model.addPhysicalGroup(1, ln_inflow, tag=1, name="inflow")
    tag_outflow = gmsh.model.addPhysicalGroup(1, ln_outflow, tag=2, name="outflow")
    tag_noslip = gmsh.model.addPhysicalGroup(1, ln_no_slip, tag=3, name="no-slip")
    tag_cut = gmsh.model.addPhysicalGroup(1, ln_cut, tag=4, name="cut")
    
    tag_bulk_2d = gmsh.model.addPhysicalGroup(2, srfs, tag=-1, name="bulk")

    # Meshing
    
    # n_nodes = int(np.ceil(height / lc))
    # for li in lines:
    #     meshFact.setTransfiniteCurve(li, numNodes=n_nodes)
    # for si in srfs:
    #     meshFact.setTransfiniteSurface(si)

    meshFact.generate(2)

    gmsh.write(filename)
    gmsh.fltk.run()
    gmsh.finalize()
    return


def create_hole(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("hole")

    width, height = 3., 2.
    radius = height / 5.
    rect = gmsh.model.occ.add_rectangle(0., -height/2., 0., width, height, 0)
    disk1 = gmsh.model.occ.add_disk(width/2., 0., 0., radius, radius, 1000)
    res_cut = gmsh.model.occ.cut([(2, rect)], [(2, disk1)])
    # disk2 = gmsh.model.occ.add_disk(width, 0., 0., radius, radius, 1001)
    # res_cut = gmsh.model.occ.cut([(2, rect)], [(2, disk1), (2, disk2)], tag=1)
    
    gmsh.model.occ.synchronize()

    tag = gmsh.model.addPhysicalGroup(0, [1, 2, 3, 4, 5], tag=3, name="points")
    tag = gmsh.model.addPhysicalGroup(1, [2], tag=1, name="inflow")
    tag = gmsh.model.addPhysicalGroup(1, [3], tag=2, name="outflow")
    tag = gmsh.model.addPhysicalGroup(1, [1, 4, 5], tag=3, name="no-slip")
    tag = gmsh.model.addPhysicalGroup(2, [0], tag=-1, name="bulk")

    gmsh.model.mesh.set_size_callback(lambda *args: elemSizeRatio * width)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    # gmsh.model.mesh.setOrder(2)

    gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()



if __name__ == "__main__":
    path_to_dir = "./mesh/"

    create_split_rectangle(path_to_dir + "rect_coarse.msh", elemSizeRatio=1./10., y_zero=0.0, cut=False)
    # create_hole(path_to_dir + "hole_fine.msh", elemSizeRatio=1./20.)
