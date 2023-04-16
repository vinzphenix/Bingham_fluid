import sys
sys.path.append('../')
import numpy as np
import os
import ctypes
import subprocess
# import xmesh2d
import gmsh

gmsh.initialize()

class Mesh:
    def __init__(self):
        _, x, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
        _, tri = gmsh.model.mesh.get_elements_by_type(2)
        self.x = x.reshape(-1,3)
        self.el = (tri-1).reshape(-1,3).astype(int)
        t = self.el
        hedges = np.hstack([
            [t[:,0],t[:,1]],[t[:,1],t[:,0]],
            [t[:,1],t[:,2]],[t[:,2],t[:,1]],
            [t[:,2],t[:,0]],[t[:,0],t[:,2]]]).T
        hedges = np.unique(hedges,axis=0).astype(np.int64)
        self.neigh_start = np.cumsum(np.hstack([[0],np.bincount(hedges[:,0])]))
        self.neigh = hedges[:,1].copy()
        edges = np.hstack([[t[:,0],t[:,1]], [t[:,1],t[:,2]], [t[:,2],t[:,0]]]).T
        edges = np.sort(edges, axis=1)
        self.edges = np.unique(edges,axis=0)
        bnd, _, _= gmsh.model.mesh.get_nodes(1, includeBoundary=True, returnParametricCoord=False)
        self.boundary_nodes = np.unique(bnd).astype(int)-1
        corner, _, _= gmsh.model.mesh.get_nodes(0, returnParametricCoord=False)
        self.corner_nodes = np.unique(corner).astype(int)-1
        hel = np.column_stack((self.el.flat, np.repeat(np.arange(self.el.shape[0]), 3)))
        hel = hel[np.argsort(hel[:,0])]
        self.node_el_start = np.cumsum(np.hstack([[0],np.bincount(hel[:,0])]))
        self.node_el = hel[:,1]


selection_mode = "nearest"

def select_nodes(mesh, f_old, f, front_prev):
    edges = mesh.edges
    cut_edges = edges[(f[edges[:,0]]>0) != (f[edges[:,1]]>0)]
    front_next = front_prev | ((f_old > 0) != (f > 0))
    
    corner0 = (np.isin(cut_edges[:,0], mesh.corner_nodes))
    corner1 = (np.isin(cut_edges[:,1], mesh.corner_nodes))
    nearer1 = np.abs(f[cut_edges[:,0]]) > np.abs(f[cut_edges[:,1]])
    if selection_mode == "nearest":
        tomove = np.where(np.logical_or(np.logical_and(nearer1,np.logical_not(corner1)), corner0), cut_edges[:,1], cut_edges[:,0])
    elif selection_mode == "front":
        tomove = np.where(front_next[cut_edges[:,0]], cut_edges[:,0], cut_edges[:,1])
    elif selection_mode == "back":
        tomove = np.where(front_next[cut_edges[:,0]], cut_edges[:,1], cut_edges[:,0])
    else:
        print(f"Unknown mode '{mode}'.")
        exit(0)

    return np.unique(tomove)

def move_node_along_edges(mesh, x, i, f):
    neigh = mesh.neigh[mesh.neigh_start[i]:mesh.neigh_start[i+1]]
    if i in mesh.boundary_nodes:
        neigh = np.intersect1d(neigh, mesh.boundary_nodes)
    opposite_neigh = neigh[(f[neigh]>0) != (f[i]>0)]
    if opposite_neigh.size == 0:
        print("cannot move node !")
        return x[i]
    dist = np.linalg.norm(x[opposite_neigh]- x[i], axis=1)
    closest = np.argmin((f[opposite_neigh]-f[i])/dist*np.sign(f[i]))
    target = opposite_neigh[closest]
    alpha = f[i]/(f[i]-f[target])
    return alpha*x[target]+(1-alpha)*x[i]

def move_front(mesh, x, f_old, f, front_prev):
    x_next = x.copy()
    f_next = f.copy()
    moved = []
    tomove = select_nodes(mesh, f_old, f, front_prev)
    front_next = np.full(x.shape[0], False)
    for i in tomove:
        x_next[i] = move_node_along_edges(mesh, x, i, f)
        f_next[i] = 0
        front_next[i] = True
        moved.append((x[i,0],x_next[i,0],x[i,1],x_next[i,1],0,0))
    return x_next, f_next, front_next, tomove, np.array(moved)


def circle(x):
    dist = np.sqrt((x[:,0]-1)**2 + (x[:,1]-1)**2)-0.5
    return dist

def gen_rectangular_mesh(X,Y,lc):
    rect = gmsh.model.occ.add_rectangle(0,0,0,X,Y,0)
    #gmsh.option.set_number("Mesh.Algorithm", 1)
    gmsh.model.mesh.set_size_callback(lambda *args: lc)
    gmsh.model.occ.synchronize()

    bottom =gmsh.model.get_entities_in_bounding_box( -0.1,  -0.1, -0.1, X+0.1, 0.1,   0.1,1)[0][1]
    top =   gmsh.model.get_entities_in_bounding_box( -0.1, Y-0.1, -0.1, X+0.1, Y+0.1, 0.1,1)[0][1]
    left =  gmsh.model.get_entities_in_bounding_box( -0.1,  -0.1, -0.1,   0.1, Y+0.1, 0.1,1)[0][1]
    right = gmsh.model.get_entities_in_bounding_box(X-0.1,  -0.1, -0.1, X+0.1, Y+0.1, 0.1,1)[0][1]
    N= 0
    if N != 0:
        gmsh.model.mesh.setTransfiniteCurve(bottom, N)
        gmsh.model.mesh.setTransfiniteCurve(top, N)
        gmsh.model.mesh.setTransfiniteCurve(left, N)
        gmsh.model.mesh.setTransfiniteCurve(right, N)
        gmsh.model.mesh.setTransfiniteSurface(rect)
    gmsh.model.mesh.generate(2)
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[left]),"left")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[right]),"right")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[bottom]),"bottom")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[top]),"top")
    gmsh.model.set_physical_name(2,gmsh.model.add_physical_group(2,[rect]),"domain")

class TrackerUI:

    def  __init__(self, pause=False):
        gmsh.option.set_number("General.Antialiasing",1)
        gmsh.option.set_number("General.SmallAxes",0)
        gmsh.option.set_number("Print.Background",1)
        gmsh.option.set_number("Mesh.LineWidth",2)
        gmsh.option.set_color("Mesh.One",0,0,0)
        gmsh.option.set_number("Mesh.Algorithm",2)
        self.view = gmsh.view.add("front")
        tag = gmsh.view.get_index(self.view)
        gmsh.option.set_number(f"View[{tag}].ShowScale",0)
        gmsh.option.set_number(f"View[{tag}].PointType",1)
        gmsh.option.set_number(f"View[{tag}].PointSize",5)
        gmsh.option.set_number(f"View[{tag}].ColormapBias",-0.15)
        pause = 1 if pause else 0
        gmsh.onelab.set(f"""[
                {{ "type":"number", "name":"Pause", "values":[{pause}], "choices":[0, 1] }},
                {{ "type":"number", "name":"Record", "values":[0], "choices":[0, 1] }}
                ]""")
        self.recid = 0
        gmsh.fltk.initialize()


    def update(self, x, front):
        for tn in range(x.shape[0]):
            gmsh.model.mesh.set_node(tn+1, x[tn], [0,0])
        self.pause = gmsh.onelab.get_number("Pause") == 1
        frontnodes = np.where(front)[0]
        vs = np.zeros((frontnodes.size, 4))
        vs[:,:3] = x[np.asarray(front).astype(bool)]
        gmsh.view.addListData(self.view, "SP", vs.shape[0], vs.reshape([-1]))

        gmsh.graphics.draw()
        gmsh.model.set_current(gmsh.model.get_current()) #force redraw
        if gmsh.onelab.get_number("Record")[0] == 1 and not self.pause:
            #gmsh.write(f"fig-{self.recid:05}.msh")
            #gmsh.view.write(self.view, f"fig-{self.recid:05}.pos")
            gmsh.write(f"fig-{self.recid:05}.png")
            self.recid += 1


class FieldEUI:

    def __init__(self, name, eid, **opts):
        self.view = gmsh.view.add(name)
        self.elements = eid
        index = gmsh.view.get_index(self.view)
        for key, value in opts.items():
            gmsh.option.set_number(f"View[{index}].{key}",value)

    def set(self, sol):
        nodes = np.arange(1, sol.shape[0]+1)
        gmsh.view.add_model_data(self.view, 0, "", "ElementNodeData", self.elements, sol)

class FieldUI:

    def __init__(self, name, **opts):
        self.view = gmsh.view.add(name)
        index = gmsh.view.get_index(self.view)
        for key, value in opts.items():
            gmsh.option.set_number(f"View[{index}].{key}",value)

    def set(self, sol):
        nodes = np.arange(1, sol.shape[0]+1)
        gmsh.view.add_model_data(self.view, 0, "", "NodeData", nodes, sol[:,None])


gen_rectangular_mesh(2,2,0.1)

mesh = Mesh()
ui = TrackerUI()
solui = FieldUI("solution")
front = np.full(mesh.x.shape[0], False)

sol_init = lambda var: circle(var)
x, _, front, _, _ = move_front(mesh, mesh.x[:], np.zeros(mesh.x.shape[0]), sol_init(mesh.x), front)

solui.set(sol_init(x))
ui.update(x, front)

gmsh.fltk.run()
