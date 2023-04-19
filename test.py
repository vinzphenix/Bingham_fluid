import numpy as np
# from cvxopt import matrix, solvers

# c = matrix([-6., -4., -5.])

# G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.],
#                 [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.],
#                 [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.]])

# G = np.array(G)
# G = matrix(G)

# h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.] )

# dims = {'l': 2, 'q': [4 for i in range(2)], 's': []}

# # P = np.zeros()
# sol = solvers.conelp(c, G, h, dims)

# class MyClass:
#     def __init__(self):
#         self.attr = "my attribute"
#         self.add_new_attribute("newwwww !")
#     def add_new_attribute(self, s_input):
#         self.s = s_input


# myclass = MyClass()
# # myclass.add_new_attribute("newwwww !")
# print(myclass.s)

"""
Python example on using gmsh [1] on extruded transfinite meshes.

Author: Breno Vincenzo de Almeida
Date: 1 June 2022

References
----------
[1] C. Geuzaine and J.-F. Remacle. Gmsh: a three-dimensional finite
element mesh generator with built-in pre- and post-processing
facilities. International Journal for Numerical Methods in Engineering
79(11), pp. 1309-1331, 2009

"""

import gmsh
import math


def main1():
    gmsh.initialize()

    # alias to facilitate code writing
    factory = gmsh.model.geo

    # default mesh size (not necessary, since we are using transfinite curves
    # and setting a certain number of points in all curves)
    lc = 1.

    # Geometry
    # points
    p1 = factory.addPoint(0., 0., 0., lc)
    p2 = factory.addPoint(10., 0., 0., lc)
    p3 = factory.addPoint(0., 10., 0., lc)
    p4 = factory.addPoint(4., 0., 0., lc)
    p5 = factory.addPoint(0., 4., 0., lc)
    p6 = factory.addPoint(4., 4., 0., lc)
    angle = math.pi/4.
    p7 = factory.addPoint(10*math.cos(angle), 10*math.sin(angle), 0., lc)

    # lines
    l1 = factory.addLine(p5, p6)
    l2 = factory.addLine(p6, p4)
    l3 = factory.addLine(p4, p1)
    l4 = factory.addLine(p1, p5)
    l5 = factory.addLine(p4, p2)
    l6 = factory.addLine(p5, p3)
    l7 = factory.addLine(p6, p7)
    l8 = factory.addCircleArc(p2, p1, p7)
    l9 = factory.addCircleArc(p7, p1, p3)

    # curve loops
    cl1 = factory.addCurveLoop([l3, l4, l1, l2])
    cl2 = factory.addCurveLoop([l7, l9, -l6, l1])
    cl3 = factory.addCurveLoop([l5, l8, -l7, l2])

    # surfaces
    s1 = factory.addPlaneSurface([cl1])
    s2 = factory.addPlaneSurface([cl2])
    s3 = factory.addPlaneSurface([cl3])

    # extrusions
    # dx = 5.
    # num_els_z = 10
    # factory.extrude([(2, s1), (2, s2), (2, s3)], 0., 0., dx,
    #                 numElements=[num_els_z], recombine=True)

    factory.synchronize()

    # Meshing
    meshFact = gmsh.model.mesh

    # transfinite curves
    n_nodes = 10
    # "Progression" 1 is default
    meshFact.setTransfiniteCurve(l1, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l2, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l3, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l4, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l5, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l6, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l7, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l8, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l9, numNodes=n_nodes)
    # transfinite surfaces
    meshFact.setTransfiniteSurface(s1)
    meshFact.setTransfiniteSurface(s2)
    meshFact.setTransfiniteSurface(s3)

    # mesh
    meshFact.generate(2)
    # meshFact.recombine()
    # meshFact.generate(3)

    gmsh.fltk.run()

    gmsh.finalize()


def main2():
    import gmsh
    import sys

    gmsh.initialize(sys.argv)

    # Copied from discrete.py...
    gmsh.model.add("test")
    gmsh.model.addDiscreteEntity(2, 1)
    gmsh.model.mesh.addNodes(2, 1, [1, 2, 3, 4],
                            [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.])
    gmsh.model.mesh.addElements(2, 1, [2], [[1, 2]], [[1, 2, 3, 1, 3, 4]])
    # ... end of copy

    # Create a new post-processing view
    t = gmsh.view.add("some data")

    # add 10 steps of model-based data, on the nodes of the mesh
    for step in range(0, 10):
        gmsh.view.addModelData(
            t,
            step,
            "test",
            "NodeData",
            [1, 2, 3, 4],  # tags of nodes
            [[10.], [10.], [12. + step], [13. + step]])  # data, per node

    # gmsh.view.write(t, "data.msh")

    gmsh.fltk.initialize()
    gmsh.fltk.run()

    gmsh.finalize()

def main3():
    import gmsh
    import sys

    gmsh.initialize(sys.argv)

    t2 = gmsh.view.add("Second order triangle")

    pt1 = gmsh.model.geo.addPoint(0., 0., 0., 1.)
    pt2 = gmsh.model.geo.addPoint(1., 0., 0., 1.)
    pt3 = gmsh.model.geo.addPoint(1., 1., 0., 1.)
    pt4 = gmsh.model.geo.addPoint(0., 1., 0., 1.)
    ln1 = gmsh.model.geo.addLine(pt1, pt2)
    ln2 = gmsh.model.geo.addLine(pt2, pt3)
    ln3 = gmsh.model.geo.addLine(pt3, pt4)
    ln4 = gmsh.model.geo.addLine(pt4, pt1)
    cl1 = gmsh.model.geo.addCurveLoop([ln1, ln2, ln3, ln4])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(2)

    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    elementTypes, elementTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)

    modelName = gmsh.model.list()[0]
    print(nodeTags)
    print(elemNodeTags)
    # data_nodes = [0., 1., -1., 2.,
    #               0., 0., 0., 0.,
    #               0., 0., 0., 0.,
    #              -2., 0., 0., 0.,
    #               0., 0., 0., 0.,
    #               0., 0., 0., 0.,
    #               0.]
                
    
    # data_nodes = [0., 0., 0., 0.,
    #               0., 0., 0., 0.,
    #               0., 0., 0., 0.,
    #               0., 0., 0., 0.,
    #               0., -0.2, 0., 0.,
    #               0.5, 0., 0., 0.2,
    #               0.8]

    data_nodes = [0., 1., -1., 2.,
                  0., 0., 0., 0.,
                  -2., 0., 0., 0.,
                   0.]
    # data_nodes = [0., 1., -1., 2., -2.]
    data = data_nodes
    
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    print(len(data_nodes))
    print(len(nodeTags))

    gmsh.view.addHomogeneousModelData(t2, 0, modelName, "NodeData", nodeTags, data, numComponents=1)


    # gmsh.view.setInterpolationMatrices(t2, "Triangle", 10,
    #                                    [1., -1., -1., 0., 0., 0., 0., 0., 0., 0.,  # node 1
    #                                     0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,    # node 2
    #                                     0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,    # node 3
    #                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 3 nodes
    #                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 3 nodes
    #                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 2 nodes
    #                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 2 nodes
    #                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 1 nodes
    #                                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 1 nodes
    #                                     0., 0., 0., -27., -27., 27., 0., 0., 0., 0.],  # face node
    #                                    [0, 0, 0,
    #                                     1, 0, 0,
    #                                     0, 1, 0,
    #                                     2, 1, 0,
    #                                     1, 2, 0,
    #                                     1, 1, 0,
    #                                     0, 0, 0,
    #                                     0, 0, 0,
    #                                     0, 0, 0,
    #                                     0, 0, 0])

    # gmsh.view.setInterpolationMatrices(t2, "Triangle", 6,
    #                                    [1., -1., -1., 0., 0., 0.,  # node 1
    #                                     0., 1., 0., 0., 0., 0.,    # node 2
    #                                     0., 0., 1., 0., 0., 0.,    # node 3
    #                                     0., 0., 0., 0., 0., 0.,    # edge 3 nodes
    #                                     0., 0., 0., 0., 0., 0.,    # edge 3 nodes
    #                                     0., 0., 0., 0., 0., 0.,    # edge 2 nodes
    #                                     ],
    #                                    [0, 0, 0,
    #                                     1, 0, 0,
    #                                     0, 1, 0,
    #                                     2, 0, 0,
    #                                     1, 1, 0,
    #                                     0, 2, 0])

    # gmsh.view.setInterpolationMatrices(t2, "Triangle", 3,
    #                                    [1., -1., -1.,  # node 1
    #                                     0., 1., 0.,    # node 2
    #                                     0., 0., 1.,],  # node 3
    #                                    [0, 0, 0,
    #                                     1, 0, 0,
    #                                     0, 1, 0])


    gmsh.view.setInterpolationMatrices(t2, "Quadrangle", 9,
                                   [0, 0, 0.25, 0, 0, -0.25, -0.25, 0, 0.25,
                                    0, 0, 0.25, 0, 0, -0.25, 0.25, 0, -0.25,
                                    0, 0, 0.25, 0, 0, 0.25, 0.25, 0, 0.25,
                                    0, 0, 0.25, 0, 0, 0.25, -0.25, 0, -0.25,
                                    0, 0, -0.5, 0.5, 0, 0.5, 0, -0.5, 0,
                                    0, 0.5, -0.5, 0, 0.5, 0, -0.5, 0, 0,
                                    0, 0, -0.5, 0.5, 0, -0.5, 0, 0.5, 0,
                                    0, 0.5, -0.5, 0, -0.5, 0, 0.5, 0, 0,
                                    1, -1, 1, -1, 0, 0, 0, 0, 0],
                                   [0, 0, 0,
                                    2, 0, 0,
                                    2, 2, 0,
                                    0, 2, 0,
                                    1, 0, 0,
                                    2, 1, 0,
                                    1, 2, 0,
                                    0, 1, 0,
                                    1, 1, 0])


    # gmsh.view.addListData(t2, "SQ", 1, quad)

    # # adaptive visualization
    # gmsh.view.option.setNumber(t2, "AdaptVisualizationGrid", 1)
    # gmsh.view.option.setNumber(t2, "TargetError", 1e-2)
    # gmsh.view.option.setNumber(t2, "MaxRecursionLevel", 6)

    # # get adaptive visualization data
    # dataType, numElements, data = gmsh.view.getListData(t2)

    # # create discrete surface
    # surf = gmsh.model.addDiscreteEntity(2)

    # # create nodes and elements and add them to the surface
    # N = 1
    # for t in range(0, len(dataType)):
    #     if dataType[t] == 'SQ': # quad
    #         coord = []
    #         tags = []
    #         ele = []
    #         for q in range(0, numElements[t]):
    #             coord.extend([data[t][16*q+0], data[t][16*q+4], data[t][16*q+8]])
    #             coord.extend([data[t][16*q+1], data[t][16*q+5], data[t][16*q+9]])
    #             coord.extend([data[t][16*q+2], data[t][16*q+6], data[t][16*q+10]])
    #             coord.extend([data[t][16*q+3], data[t][16*q+7], data[t][16*q+11]])
    #             tags.extend([N, N+1, N+2, N+3])
    #             ele.extend([N, N+1, N+2, N+3])
    #             N = N+4
    #         gmsh.model.mesh.addNodes(2, 1, tags, coord)
    #         gmsh.model.mesh.addElementsByType(surf, 3, [], ele)

    # # remove duplicate nodes
    # gmsh.model.mesh.removeDuplicateNodes()

    # # save mesh
    # gmsh.write('test.msh')

    # gmsh.option.setNumber("View.AdaptVisualizationGrid", 1)
    gmsh.option.setNumber("View.MaxRecursionLevel", 6)
    gmsh.option.setNumber("View.TargetError", 0.)
    gmsh.option.setNumber("View.VectorType", 6)
    gmsh.option.setNumber("View[0].NormalRaise", -0.2)
    gmsh.option.setNumber("View[0].DrawLines", 0)

    # Launch the GUI to see the results:
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


def main4():
    # -----------------------------------------------------------------------------
    #
    #  Gmsh Python extended tutorial 3
    #
    #  Post-processing data import: list-based
    #
    # -----------------------------------------------------------------------------

    import gmsh
    import sys

    gmsh.initialize(sys.argv)

    # Add a new view:
    t2 = gmsh.view.add("Bubble element")

    p1 = gmsh.model.geo.addPoint(0., 0., 0., 1.)
    p2 = gmsh.model.geo.addPoint(1., 0., 0., 1.)
    p3 = gmsh.model.geo.addPoint(1., 1., 0., 1.)
    p4 = gmsh.model.geo.addPoint(0., 1., 0., 1.)
    ln1 = gmsh.model.geo.addLine(p1, p2)
    ln2 = gmsh.model.geo.addLine(p2, p3)
    ln3 = gmsh.model.geo.addLine(p3, p4)
    ln4 = gmsh.model.geo.addLine(p4, p1)
    cl1 = gmsh.model.geo.addCurveLoop([ln1, ln2, ln3, ln4])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.setOrder(3)

    gmsh.view.setInterpolationMatrices(t2, "Triangle", 10,
                                       [1., -1., -1., 0., 0., 0., 0., 0., 0., 0.,  # node 1
                                        0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,    # node 2
                                        0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,    # node 3
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 3 nodes
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 3 nodes
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 2 nodes
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 2 nodes
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 1 nodes
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,    # edge 1 nodes
                                        0., 0., 0., -27., -27., 27., 0., 0., 0., 0.],  # face node
                                       [0, 0, 0,
                                        1, 0, 0,
                                        0, 1, 0,
                                        2, 1, 0,
                                        1, 2, 0,
                                        1, 1, 0,
                                        0, 0, 0,
                                        0, 0, 0,
                                        0, 0, 0,
                                        0, 0, 0])
    
    # gmsh.view.addListData(t2, "SQ", 1, quad)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords).reshape((-1, 3))
    elementTypes, elementTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    elementTags = elementTags[0]
    elemNodeTags = elemNodeTags[0].reshape((elementTags.size, -1))
    print(elementTags)
    print(elemNodeTags)
    values = np.array([
        1., -1., -0, 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
        1., 0., 0., 0.,
        0., 0.3, 0., 0.,
        0.1, 0., 0., 0.1,
        0.2
    ])

    data = np.array([])
    data_bis = np.empty((len(elementTags), 10))
    for i, idx_nodes in enumerate(elemNodeTags):
        # output_values = np.c_[values[idx_nodes-1], np.ones(10), np.zeros(10)].flatten()
        output_values = values[idx_nodes-1]
        data = np.r_[data, coords[idx_nodes[:3]-1, 0], coords[idx_nodes[:3]-1, 1], coords[idx_nodes[:3]-1, 2],
                     output_values]
        data_bis[i] = values[idx_nodes-1]

    gmsh.view.addListData(t2, "ST", 4, data)
    # gmsh.view.addListData(t2, "VT", 4, data)

    # print(gmsh.model.mesh.getNodes())
    # print(gmsh.model.mesh.getElements())
    # values = np.array([1., 2., 0., 4., 0., 0., 0., 0., 0.])
    # values = values.reshape((-1, 1))
    # gmsh.view.addModelData(t2, 0, gmsh.model.list()[0], "ElementNodeData", elementTags, data_bis)

    gmsh.view.option.setNumber(t2, "AdaptVisualizationGrid", 1)
    gmsh.view.option.setNumber(t2, "TargetError", -0.0001)
    gmsh.view.option.setNumber(t2, "MaxRecursionLevel", 4)
    gmsh.option.setNumber("View.NormalRaise", -0.2)
    gmsh.option.setNumber("View.DrawLines", 0)

    # Launch the GUI to see the results:
    gmsh.fltk.run()
    gmsh.finalize()

def main5():

    import gmsh
    import sys

    gmsh.initialize(sys.argv)

    # Add a new view:
    t2 = gmsh.view.add("Second order triangle")

    p1 = gmsh.model.geo.addPoint(0., 0., 0., 1.)
    p2 = gmsh.model.geo.addPoint(1., 0., 0., 1.)
    p3 = gmsh.model.geo.addPoint(1., 1., 0., 1.)
    p4 = gmsh.model.geo.addPoint(0., 1., 0., 1.)
    ln1 = gmsh.model.geo.addLine(p1, p2)
    ln2 = gmsh.model.geo.addLine(p2, p3)
    ln3 = gmsh.model.geo.addLine(p3, p4)
    ln4 = gmsh.model.geo.addLine(p4, p1)
    cl1 = gmsh.model.geo.addCurveLoop([ln1, ln2, ln3, ln4])
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.setOrder(2)

    gmsh.view.setInterpolationMatrices(t2, "Triangle", 6,
                                       [1., -3., -3., 2., 4., 2.,  # node 1
                                        0., -1., 0., 2., 0., 0.,    # node 2
                                        0., 0., -1., 0., 0., 2.,    # node 3
                                        0., 4., 0., -4., -4., 0.,    # edge 2 nodes
                                        0., 0., 0., 0., 4., 0.,    # edge 3 nodes
                                        0., 0., 4., 0., -4., -4.,    # edge 3 nodes
                                        ],
                                       [0, 0, 0,
                                        1, 0, 0,
                                        0, 1, 0,
                                        2, 0, 0,
                                        1, 1, 0,
                                        0, 2, 0])
    
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords).reshape((-1, 3))
    elementTypes, elementTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    elementTags = elementTags[0]
    elemNodeTags = elemNodeTags[0].reshape((elementTags.size, -1))
    print(elementTags)
    print(elemNodeTags)
    values = np.array([
        0., 1., 2., 1.,
        .5, 1.5, 1.5, 0.5,
        1., 1., .5, 1.,
        1.5
    ])
    values_v = np.array([
        2., 1., 0., 1.,
        1.5, .5, .5, 1.5,
        1., 1., 1.5, 1.,
        0.5
    ])

    data = np.array([])
    data_bis = np.empty((len(elementTags), 6))
    for i, idx_nodes in enumerate(elemNodeTags):
        output_values = np.c_[values[idx_nodes-1], values_v[idx_nodes-1], np.zeros(6)].flatten()
        # output_values = values[idx_nodes-1]
        data = np.r_[data, coords[idx_nodes[:3]-1, 0], coords[idx_nodes[:3]-1, 1], coords[idx_nodes[:3]-1, 2],
                     output_values]
        data_bis[i] = values[idx_nodes-1]

    gmsh.view.addListData(t2, "VT", 4, data)
    # gmsh.view.addListData(t2, "ST", 4, data)

    # print(gmsh.model.mesh.getNodes())
    # print(gmsh.model.mesh.getElements())
    # values = np.array([1., 2., 0., 4., 0., 0., 0., 0., 0.])
    # values = values.reshape((-1, 1))
    # gmsh.view.addModelData(t2, 0, gmsh.model.list()[0], "ElementNodeData", elementTags, data_bis)

    gmsh.view.option.setNumber(t2, "AdaptVisualizationGrid", 1)
    gmsh.view.option.setNumber(t2, "TargetError", -0.0001)
    gmsh.view.option.setNumber(t2, "MaxRecursionLevel", 4)
    gmsh.option.setNumber("View.NormalRaise", -0.2)
    gmsh.option.setNumber("View.DrawLines", 0)
    gmsh.option.setNumber("View.GlyphLocation", 2)

    # Launch the GUI to see the results:
    gmsh.fltk.run()
    gmsh.finalize()


import mosek
import cvxopt
from cvxopt import matrix, solvers, spmatrix
from time import perf_counter
def main6():


    c = matrix([-2., 1., 5.])
    G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
    G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
    # h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
    
    print(G[0])

    G, h = [], []

    data = [12., 6., -5., 13., -3., -5., 12., -12., 6.]
    rows = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cols = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    G += [ spmatrix(data, rows, cols, size=(3, 3))]
    h += [ matrix( np.array([-12., -3., -2.]) ) ]

    # data = np.array([3., -6., 10., 3., -6., -2., -1., -9., -2., 1., 19., -3.])
    data = np.array([3., -6., -1., -9., 1., 19., 3., -6.])
    rows = np.array([1, 1, 2, 2, 3, 3, 0, 0])
    cols = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    G += [ spmatrix(data, rows, cols, size=(4, 3))]
    h += [ matrix( np.array([27., 0., 3., -42.]) ) ]


    # Gl = matrix( np.array([[-1., 0., 0.], [+1., 0., 0.]]) )
    hl = matrix( np.array([+4., -4.]) )

    data = np.array([-1., 1.])
    rows = np.array([0, 1])
    cols = np.array([0, 0])
    Gl = spmatrix(data, rows, cols, size=(2, 3))

    print(Gl)
    print(hl)

    A = matrix(np.array([[1., 0., 0.]]))
    b = matrix(np.array([-4.]))
    # x = -4  --> x >= -4  &  x <= -4
    # -1*x + s = +4
    # +1*x + s = -4


    cvxopt.solvers.options['show_progress'] = False
    solvers.options['mosek'] = {mosek.iparam.log: 0}

    start = perf_counter()
    sol = solvers.socp(c, Gl=Gl, hl=hl, Gq=G, hq=h, solver="mosek", )
    # sol = solvers.socp(c, Gq=G, hq=h, A=A, b=b, solver="conelp", )
    end = perf_counter()

    sol['status']
    print(sol['x'])
    print(f"Elapsed time = {(end-start)*1e3:.3f} ms")


if __name__ == "__main__":
    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    main6()
