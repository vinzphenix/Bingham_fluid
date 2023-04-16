import sys
sys.path.append('../')
import gmsh
import numpy as np
import itertools
import re
import ctypes
import os
import fluid as fl

def np2c(a,dtype=float,order="C") :
    tmp = np.require(a,dtype,order)
    r = ctypes.c_void_p(tmp.ctypes.data)
    r.tmp = tmp
    return r

lib = ctypes.CDLL(os.path.dirname(__file__)+"/build/libxmesh.so")

gmsh.initialize()

def process_events():
    gmsh.graphics.draw()
    while gmsh.onelab.get_number("Pause") == 1:
        gmsh.graphics.draw()


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

class ConcentrationUI:

    def __init__(self, name, mesh, **opts):
        self.view = gmsh.view.add(name)
        self.mesh = mesh
        for key, value in opts.items():
            gmsh.option.set_number(f"View[{self.view}].{key}",value)

    def set(self, data):
        gmsh.view.add_model_data(self.view, 0, "", "ElementData", self.mesh.eltags, data[:,None])

CLAMPCB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
class clampcb2c:
    def __init__(self, cb):
        self.cb = cb
    def __call__(self, n, xptr):
        x = np.frombuffer(ctypes.cast(xptr, ctypes.POINTER(int(n)*3*ctypes.c_double)).contents)
        self.cb(x.reshape(-1,3))

class Mesh():
    class MeshC(ctypes.Structure):
        _fields_ = [
                ("dimension", ctypes.c_int),
                ("n_elements", ctypes.c_int),
                ("elements", ctypes.POINTER(ctypes.c_int)),
                ("n_edges", ctypes.c_int),
                ("edges", ctypes.POINTER(ctypes.c_int)),
                ("n_nodes", ctypes.c_int),
                ("n_boundaries", ctypes.c_int),
                ("boundary_tag", ctypes.POINTER(ctypes.c_char_p)),
                ("n_boundary_nodes", ctypes.POINTER(ctypes.c_int)),
                ("boundary_nodes", ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
                ("neighbour_start", ctypes.POINTER(ctypes.c_int)),
                ("neighbour", ctypes.POINTER(ctypes.c_int)),
                ("clampcb", CLAMPCB), 
                ("evn_istart", ctypes.POINTER(ctypes.c_int)),
                ("evn_jk",ctypes.POINTER(ctypes.c_int))
                ]
    """
    Topological mesh (no coordinates)
    """

    @staticmethod
    def _create_sub_simplices(elements, subdim, closures):
        n = elements.shape[1]
        nsub = subdim+1
        comb = np.array(list(itertools.combinations(range(n), nsub)))
        sube = elements[:,comb].reshape(-1,nsub)
        asort = sube.argsort(axis=1)
        sube,emap = np.unique(np.take_along_axis(sube,asort,1),axis=0,return_inverse=True)
        emap = emap.reshape(elements.shape[0],-1)
        nmap = np.take_along_axis(np.tile(comb,(elements.shape[0],1)),asort,1)
        def p2id(perm):
            return np.sum(perm[:,:subdim+1]*(n**np.arange(subdim+1))[None,:],axis=1)
        clrefid = p2id(closures)
        csort = np.argsort(clrefid)
        closureid = csort[np.searchsorted(clrefid,p2id(nmap),sorter=csort)]
        return sube, emap, closureid.reshape(-1,comb.shape[0])

    def _gen_neighbours(self):
        if self.dim == 2:
            t = self.el
            hedges = np.hstack([
                [t[:,0],t[:,1]],[t[:,1],t[:,0]],
                [t[:,1],t[:,2]],[t[:,2],t[:,1]],
                [t[:,2],t[:,0]],[t[:,0],t[:,2]]]).T
        else :
            e = self.el
            hedges = np.hstack([
                [e[:,0],e[:,1]],[e[:,1],e[:,0]]]).T
        hedges = np.unique(hedges,axis=0).astype(np.int64)
        self.neigh_start = np.cumsum(np.hstack([[0],np.bincount(hedges[:,0])]))
        self.neigh = hedges[:,1].copy()

    def _gen_edge_vneighbours(self):
        ''' from the array of triangle 2 vertices, computes the neighbors vertices of each edge
        '''
        t = self.el
        tp = np.hstack([
            [t[:,0], t[:,1], t[:,2]],
            [t[:,1], t[:,0], t[:,2]],
            [t[:,1], t[:,2], t[:,0]],
            [t[:,2], t[:,1], t[:,0]],
            [t[:,2], t[:,0], t[:,1]],
            [t[:,0], t[:,2], t[:,1]]
            ]).T

        i = np.lexsort((tp[:,2], tp[:,1], tp[:,0]))
        tp = tp[i,:]
        self.evn_istart = np.cumsum(np.hstack([[0],np.bincount(tp[:,0])])).astype(np.int32)
        self.evn_jk = tp[:,1:3].astype(np.int32)
        

    def get_neighbours(self,i):
        '''
            return the neighbours of vertex i
            (j is consdered a neighbour of vertex i if there is an edge in at least one triangle
            of the mesh that link vertex i and j)
        '''
        return self.neigh[self.neigh_start[i]:self.neigh_start[i+1 ]]

    def get_edge_vneighbours(self, i, j):
        '''
            return the neighbours of edge i, j
            (k is consdered a neighbour of edge i,j  if there is at least one triangle
            of the mesh that link vertices i, j and k)
        '''
        if not hasattr(self, 'evn_istart'):
            self._gen_edge_vneighbours()
        jk = self.evn_jk[self.evn_istart[i]:self.evn_istart[i+1]]
        # sj = np.searchsorted(jk[:,0], j)
        # ej = np.searchsorted(jk[sj:,0], j+1)
        # return jk[sj:sj+ej,1]
        # return jk[ np.searchsorted(jk[:,0], j): np.searchsorted(jk[:,0], j+1),1 ]
        # for some reason (probably jk small enough, the following version is faster)
        return jk[jk[:,0] == j ,1]

    @classmethod
    def import_from_gmsh(cls, clamp_boundaries=None, dim=2):
        self = Mesh()
        gmsh.model.mesh.renumber_nodes()
        model = gmsh.model.get_current()
        self.dim = dim
        n = dim+1
        if dim == 2:
            self.closures = np.array([(0,1),(1,2),(2,0),(1,0),(2,1),(0,2)])
        else :
            self.closures = np.array([[0],[1]])
        self.eltags, el = gmsh.model.mesh.get_elements_by_type(dim)
        self.el = el.reshape([-1,n]).astype(np.int32)-1
        tags, x, _ = gmsh.model.mesh.get_nodes()
        order = np.argsort(tags)
        self.nodetags = tags[order]
        x = x.reshape([-1,3])[order]
        self.num_nodes = x.shape[0]
        
        assert(np.all(self.nodetags == np.arange(1, tags.size+1)))
        self._gen_neighbours()
        self.boundary_nodes = {}
        for gdim, gtag in gmsh.model.getPhysicalGroups(dim-1):
            gname = gmsh.model.get_physical_name(gdim, gtag)
            if gname is not None:
                nodes = []
                for gent in gmsh.model.get_entities_for_physical_group(gdim, gtag):
                    nodes.append(gmsh.model.mesh.getNodes(gdim, gent, True,False)[0].astype(np.int32))
                nodes = np.unique(np.hstack(nodes))
                self.boundary_nodes[gname] = nodes-1
        self.edges, el_edges, el_edges_cl = Mesh._create_sub_simplices(self.el, dim-1, self.closures)
        self.el_edges = el_edges
        self.edges_el = np.full((self.edges.shape[0],2,2),-1, np.int32)
        self.edges_el[el_edges, np.where(el_edges_cl>2,0,1),0] = np.arange(self.el.shape[0])[:,None]
        self.edges_el[el_edges, np.where(el_edges_cl>2,0,1),1] = el_edges_cl

        self.meshC = self.MeshC()
        self.meshC.dimension = dim;
        self.meshC.n_elements = self.el.shape[0]
        self.meshC.elements = self.el.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.meshC.n_edges = self.edges_el.shape[0]
        self.meshC.edges = self.edges_el.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.meshC.n_nodes = x.shape[0]
        if clamp_boundaries is not None:
            self.meshC.clampcb = CLAMPCB(clampcb2c(lambda x : clamp_boundaries(self, x)))
        else:
            self.meshC.clampcb = CLAMPCB(0)
        nb = len(self.boundary_nodes)
        self.meshC.n_boundaries = nb
        self.meshC.boundary_tag = ((ctypes.c_char_p)*nb)(*(k.encode() for k in self.boundary_nodes.keys()))
        self.meshC.n_boundary_nodes = ((ctypes.c_int)*nb)(*(v.size for v in self.boundary_nodes.values()))
        self.meshC.boundary_nodes = ((ctypes.POINTER(ctypes.c_int))*nb)(*(
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) for v in self.boundary_nodes.values()
            ))
        self.meshC.neighbour_start = self.neigh_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.meshC.neighbour = self.neigh.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.periodic_boundary = False

        self._gen_edge_vneighbours()
        print(self.evn_istart[20], self.evn_istart[21])

        self.meshC.evn_istart = self.evn_istart.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.meshC.evn_jk = self.evn_jk.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        print(self.meshC.evn_istart[20], self.meshC.evn_istart[21])

        # gmsh.model.remove()
        return self, x

def initialize_front(mesh, x, sol_init, T0=0, move_mesh=None): 
    x0_coo = x.copy()
    sol = np.ones(x.shape[0])+T0
    front = np.full(x.shape[0], False, np.int32)
    while True:
        solold = sol.copy()
        sol = sol_init(x)
        xold = x.copy()
        if move_mesh is None:
            x, front = move_front(mesh, front, x, solold-T0, sol-T0)
        else:
            x, _, front, _, _ = fl.move_front_dist(move_mesh, x.copy(), solold.copy(), sol.copy(), front.copy())
        
        sol[front] = T0
        if (x-xold).max() < 1.e-12:
            sol = sol_init(x)
            sol[front] = T0
            break
    el = mesh.el
    dxdxi = np.stack([
        x[el[:,1],:2] - x[el[:,0],:2],
        x[el[:,2],:2] - x[el[:,0],:2]], axis=1)
    det = (dxdxi[:,0,0]*dxdxi[:,1,1]-dxdxi[:,1,0]*dxdxi[:,0,1])
    zero_el = np.where(det==0)[0]
    for i in range(len(zero_el)):
        x_el = x[el[zero_el[i],:],:2]
        h = np.ones(3)
        for j in range(3):
            x0 = x_el[j%3]
            x1 = x_el[(j+1)%3]
            x2 = x_el[(j+2)%3]
            xm = x0+x1
            h[j] = np.sqrt((x2[0]-xm[0])*(x2[0]-xm[0]) + (x2[1]-xm[1])*(x2[1]-xm[1]))
        
        ind = (np.argmin(h)+2)%3
        front[el[zero_el[i],ind]] = 0

    front = np.asarray(front, dtype=bool)
    x = relax(front, x, x0_coo, 1)        
    return x, front.astype(np.int32), sol


def move_front(mesh, front, x, f_old, f): #, edge_struct, front_edges
    T0 = 0
    x_next = x.copy()
    front_next = np.copy(front)
    lib.move_front_edge(ctypes.byref(mesh.meshC), np2c(x_next), np2c(f_old), np2c(f), ctypes.c_double(T0), np2c(front_next,np.int32)) #, np2c(edge_struct,np.int32), np2c(front_edges,np.int32)
    return x_next, front_next

def move_front_edge(mesh, front, x, f_old, f): #, edge_struct, front_edges
    T0 = 0
    x_next = x.copy()
    front_next = np.copy(front)
    lib.move_front_edge(ctypes.byref(mesh.meshC), np2c(x_next), np2c(f_old), np2c(f), ctypes.c_double(T0), np2c(front_next,np.int32)) #, np2c(edge_struct,np.int32), np2c(front_edges,np.int32)
    return x_next, front_next, f


def relax(front, x, x0, alpha):
    a = np.where(front, 0, alpha)
    return  x*(1-a[:,None])+x0*a[:,None]

def move_front_python(mesh, front, x, f_old, f):

    x_next = x.copy()
    front_next = front | ((f_old > 0) != (f > 0))

    for i in np.where(front_next)[0]:
        neigh = mesh.neigh[mesh.neigh_start[i]:mesh.neigh_start[i+1]]
        opposite_neigh = neigh[(~front[neigh]) & ((f[neigh]>0) != (f[i]>0))]
        if opposite_neigh.size == 0:
            front_next[i] = False
            continue
        dist = np.linalg.norm(x[opposite_neigh]- x[i], axis=1)
        dist = np.maximum(1e-12, dist)

        if f[i] < 0 : dist = -dist
        target = opposite_neigh[np.argmin(f[opposite_neigh]/dist)]

        den = f[i]-f[target]
        sign = np.sign(den)
        if sign==0:
            sign = 1
        den = sign*np.maximum(np.absolute(den),1e-12)

        alpha = np.clip((f[i])/(den), 0, 1)
        x_next[i] = alpha*x[target]+(1-alpha)*x[i]

    return x_next, front_next

def gen_line_mesh(X,Y,lc):
    p0 = gmsh.model.occ.add_point(0,0,0,lc)
    p1 = gmsh.model.occ.add_point(X,0,0,lc)
    line = gmsh.model.occ.add_line(p0, p1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.set_physical_name(0,gmsh.model.add_physical_group(0,[p0]),"left")
    gmsh.model.set_physical_name(0,gmsh.model.add_physical_group(0,[p1]),"right")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[line]),"domain")

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

def gen_circular_mesh(X,Y,R,lc):
    c = gmsh.model.occ.add_point(X,Y,0.,lc)
    p1 = gmsh.model.occ.add_point(X+R,Y,0.,lc)
    p2 = gmsh.model.occ.add_point(X,Y+R,0.,lc)
    p3 = gmsh.model.occ.add_point(X-R,Y,0.,lc)
    p4 = gmsh.model.occ.add_point(X,Y-R,0.,lc)

    arc1 = gmsh.model.occ.add_circle_arc(p1,c,p2)
    arc2 = gmsh.model.occ.add_circle_arc(p2,c,p3)
    arc3 = gmsh.model.occ.add_circle_arc(p3,c,p4)
    arc4 = gmsh.model.occ.add_circle_arc(p4,c,p1)

    l1 = gmsh.model.occ.add_curve_loop([arc1,arc2,arc3,arc4])
    s1 = gmsh.model.occ.add_plane_surface([l1])

    gmsh.model.mesh.set_size_callback(lambda *args: lc)
    gmsh.model.occ.synchronize()
    border = gmsh.model.get_entities(1)[0][1]
    gmsh.model.mesh.generate(2)
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[border]),"border")
    gmsh.model.set_physical_name(2,gmsh.model.add_physical_group(2,[s1]),"domain")

def gen_ring_mesh(X,Y,Rint,Rext,lc):
    c = gmsh.model.occ.add_point(X,Y,0.,lc)
    p1 = gmsh.model.occ.add_point(X+Rint,Y,0.,lc)
    p2 = gmsh.model.occ.add_point(X,Y+Rint,0.,lc)
    p3 = gmsh.model.occ.add_point(X-Rint,Y,0.,lc)
    p4 = gmsh.model.occ.add_point(X,Y-Rint,0.,lc)

    arc1 = gmsh.model.occ.add_circle_arc(p1,c,p2)
    arc2 = gmsh.model.occ.add_circle_arc(p2,c,p3)
    arc3 = gmsh.model.occ.add_circle_arc(p3,c,p4)
    arc4 = gmsh.model.occ.add_circle_arc(p4,c,p1)

    p5 = gmsh.model.occ.add_point(X+Rext,Y,0.,lc)
    p6 = gmsh.model.occ.add_point(X,Y+Rext,0.,lc)
    p7 = gmsh.model.occ.add_point(X-Rext,Y,0.,lc)
    p8 = gmsh.model.occ.add_point(X,Y-Rext,0.,lc)

    arc5 = gmsh.model.occ.add_circle_arc(p5,c,p6)
    arc6 = gmsh.model.occ.add_circle_arc(p6,c,p7)
    arc7 = gmsh.model.occ.add_circle_arc(p7,c,p8)
    arc8 = gmsh.model.occ.add_circle_arc(p8,c,p5)

    gmsh.model.occ.remove([(0,c)])

    l1 = gmsh.model.occ.add_curve_loop([arc1,arc2,arc3,arc4])
    l2 = gmsh.model.occ.add_curve_loop([arc5,arc6,arc7,arc8])

    s1 = gmsh.model.occ.add_plane_surface([l2,l1])

    gmsh.model.mesh.set_size_callback(lambda *args: lc)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[arc1,arc2,arc3,arc4]),"in")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[arc5,arc6,arc7,arc8]),"out")
    gmsh.model.set_physical_name(2,gmsh.model.add_physical_group(2,[s1]),"domain")

def gen_half_ring_mesh(X,Y,Rint,Rext,lc):
    c = gmsh.model.occ.add_point(X,Y,0.,lc)
    p1 = gmsh.model.occ.add_point(X,Y-Rint,0.,lc)
    p2 = gmsh.model.occ.add_point(X+Rint,Y,0.,lc)
    p3 = gmsh.model.occ.add_point(X,Y+Rint,0.,lc)

    arc1 = gmsh.model.occ.add_circle_arc(p1,c,p2)
    arc2 = gmsh.model.occ.add_circle_arc(p2,c,p3)

    p4 = gmsh.model.occ.add_point(X,Y-Rext,0.,lc)
    p5 = gmsh.model.occ.add_point(X+Rext,Y,0.,lc)
    p6 = gmsh.model.occ.add_point(X,Y+Rext,0.,lc)

    arc3 = gmsh.model.occ.add_circle_arc(p4,c,p5)
    arc4 = gmsh.model.occ.add_circle_arc(p5,c,p6)

    line1 = gmsh.model.occ.add_line(p1,p4)
    line2 = gmsh.model.occ.add_line(p6,p3)

    gmsh.model.occ.remove([(0,c)])

    l1 = gmsh.model.occ.add_curve_loop([line1,arc3,arc4,line2,-arc2,-arc1])

    s1 = gmsh.model.occ.add_plane_surface([l1])

    gmsh.model.mesh.set_size_callback(lambda *args: lc)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[arc1,arc2]),"in")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[arc3,arc4]),"out")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[line1]),"bottom")
    gmsh.model.set_physical_name(1,gmsh.model.add_physical_group(1,[line2]),"top")
    gmsh.model.set_physical_name(2,gmsh.model.add_physical_group(2,[s1]),"domain")

def load_gmsh_opt(fname):
    content = open(fname,"r").read()
    for key, value in re.findall("(.*?) = ((?:.|\n)*?);$", open(fname, "r").read(), re.MULTILINE):
        v = value.strip()
        if key.split(".")[-1] == "ColorTable":
            pass
        elif v[0] == '"' and v[-1] == '"' :
            gmsh.option.set_string(key, v[1:-1])
        elif v[0] == '{' and v[-1] == '}':
            k = key.split(".")
            v = v[1:-1].split(",")
            gmsh.option.set_color(k[0]+"."+k[2], int(v[0]), int(v[1]), int(v[2]))
        else :
            gmsh.option.set_number(key, float(v))
