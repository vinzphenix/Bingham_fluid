import gmsh
import sys


def create_square(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("t1")
    # lc = 0.1
    lc = elemSizeRatio / 1.
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, p4, 3)
    gmsh.model.geo.addLine(4, 1, p4)
    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def create_circle(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("t1")
    lc = elemSizeRatio * 100.
    # lc = 5.

    gmsh.model.geo.addPoint(50., 0., 0, lc, 1)
    gmsh.model.geo.addPoint(100., 50., 0, lc, 2)
    gmsh.model.geo.addPoint(50., 100., 0, lc, 3)
    gmsh.model.geo.addPoint(0., 50., 0, lc, 4)
    gmsh.model.geo.addPoint(50., 50., 0, lc, 5)

    gmsh.model.geo.addCircleArc(1, 5, 3, 1)
    gmsh.model.geo.addCircleArc(3, 5, 1, 2)

    gmsh.model.geo.addCurveLoop([1, 2], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def create_hole(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("hole")

    rect = gmsh.model.occ.add_rectangle(0, 0, 0, 1.5, 1., 0)
    # disk1 = gmsh.model.occ.add_disk(0.5, 0.5, 0, 0.2, 0.2, 1000)
    # disk2 = gmsh.model.occ.add_disk(2., 0.5, 0, 0.75, 0.75, 1001)
    # mm = gmsh.model.occ.cut([(2, rect)], [(2, disk1), (2, disk2)])
    
    gmsh.model.occ.synchronize()
    # tag = gmsh.model.addPhysicalGroup(0, [1, 2, 3, 4, 5], tag=0, name="points")
    # tag = gmsh.model.addPhysicalGroup(1, [2, 4, 5], tag=0, name="no-slip")
    # tag = gmsh.model.addPhysicalGroup(1, [1], tag=1, name="inflow")
    # tag = gmsh.model.addPhysicalGroup(1, [3], tag=2, name="outflow")
    # tag = gmsh.model.addPhysicalGroup(2, [0], tag=-1, name="bulk")

    tag = gmsh.model.addPhysicalGroup(0, [1, 2, 3, 4], tag=0, name="points")
    tag = gmsh.model.addPhysicalGroup(1, [1, 3], tag=0, name="no-slip")
    tag = gmsh.model.addPhysicalGroup(1, [4], tag=1, name="inflow")
    tag = gmsh.model.addPhysicalGroup(1, [2], tag=2, name="outflow")
    tag = gmsh.model.addPhysicalGroup(2, [0], tag=-1, name="bulk")


    gmsh.model.mesh.set_size_callback(lambda *args: elemSizeRatio * 1.5)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    # gmsh.model.mesh.setOrder(2)

    gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


def create_rectangle(filename, elemSizeRatio):
    gmsh.initialize()
    gmsh.model.add("rectangle")

    X = 1.0
    Y = 1.0
    # lc = 0.03 * (X * X + Y * Y) ** 0.5

    rect = gmsh.model.occ.add_rectangle(0, 0, 0, X, Y, 0)
    gmsh.model.mesh.set_size_callback(lambda *args: elemSizeRatio / 1.)
    gmsh.model.occ.synchronize()

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("Mesh.Nodes", 1)
    gmsh.write(filename)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    path_to_dir = "./mesh/"

    # create_square(path_to_dir + "square_best.msh", elemSizeRatio=1./50.)
    # create_circle(path_to_dir + "circle_h8.msh", elemSizeRatio=8./100.)
    create_hole(path_to_dir + "hole.msh", elemSizeRatio=1./7.)
    # create_rectangle(path_to_dir + "rectangle.msh", elemSizeRatio=1./5.)
