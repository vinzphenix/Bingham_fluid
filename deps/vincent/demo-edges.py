from matplotlib import pyplot as plt
import gmsh
import numpy as np
import os

selection_mode = "nearest"  # nearest, front or back
displacement_mode = "edges"  # edges or triangles

gmsh.initialize()
os.makedirs("pdf", exist_ok=True)
os.makedirs("png", exist_ok=True)


class Mesh:
    def __init__(self):
        _, x, _ = gmsh.model.mesh.get_nodes(returnParametricCoord=False)
        _, tri = gmsh.model.mesh.get_elements_by_type(2)
        self.x = x.reshape(-1, 3)
        
        # elem_node_tags
        # self.el[i] lists all nodes belonging to element i
        self.el = (tri - 1).reshape(-1, 3).astype(int)
        t = self.el

        # Mapping from node to node
        hedges = np.hstack([
            [t[:, 0], t[:, 1]], [t[:, 1], t[:, 0]],
            [t[:, 1], t[:, 2]], [t[:, 2], t[:, 1]],
            [t[:, 2], t[:, 0]], [t[:, 0], t[:, 2]]]).T
        hedges = np.unique(hedges, axis=0).astype(np.int64)

        # Node i links to 'neigh[neigh_start[i]: neigh_start[i+1]]'
        self.neigh_start = np.cumsum(np.hstack([[0], np.bincount(hedges[:, 0])]))
        self.neigh = hedges[:, 1].copy()

        # print(hedges[:20] + 1)
        # print(self.neigh_start)
        # print(self.neigh)

        # List of all edges as (org, dst), not repeated twice in opposite direction
        edges = np.hstack([[t[:, 0], t[:, 1]], [t[:, 1], t[:, 2]], [t[:, 2], t[:, 0]]]).T
        print(t[:15])
        print(edges[:15])
        edges = np.sort(edges, axis=1)
        self.edges = np.unique(edges, axis=0)

        # Get nodes on the boundary
        bnd, _, _ = gmsh.model.mesh.get_nodes(1, includeBoundary=True, returnParametricCoord=False)
        self.boundary_nodes = np.unique(bnd).astype(int) - 1

        # Get nodes at the corners
        corner, _, _ = gmsh.model.mesh.get_nodes(0, returnParametricCoord=False)
        self.corner_nodes = np.unique(corner).astype(int) - 1

        # Mapping node to element (0 indexed both)
        hel = np.column_stack((self.el.flat, np.repeat(np.arange(self.el.shape[0]), 3)))
        # print(self.el.flat)
        # print(np.repeat(np.arange(self.el.shape[0]), 3))
        # print(hel.shape)
        hel = hel[np.argsort(hel[:, 0])]
        # print(hel[:30])

        # Node i belongs to elements 'node_el[node_el_start[i]: node_el_start[i+1]]'
        self.node_el_start = np.cumsum(np.hstack([[0], np.bincount(hel[:, 0])]))
        self.node_el = hel[:, 1]


def select_nodes(mesh, f_old, f, front_prev):
    edges = mesh.edges
    cut_edges = edges[(f[edges[:, 0]] > 0) != (f[edges[:, 1]] > 0)]
    front_next = front_prev | ((f_old > 0) != (f > 0))

    corner0 = (np.isin(cut_edges[:, 0], mesh.corner_nodes))
    corner1 = (np.isin(cut_edges[:, 1], mesh.corner_nodes))
    nearer1 = np.abs(f[cut_edges[:, 0]]) > np.abs(f[cut_edges[:, 1]])
    if selection_mode == "nearest":
        tomove = np.where(
            np.logical_or(
                np.logical_and(
                    nearer1,
                    np.logical_not(corner1)
                ),
                corner0
            ), 
            cut_edges[:, 1],
            cut_edges[:, 0]
        )
    elif selection_mode == "front":
        tomove = np.where(front_next[cut_edges[:, 0]], cut_edges[:, 0], cut_edges[:, 1])
    elif selection_mode == "back":
        tomove = np.where(front_next[cut_edges[:, 0]], cut_edges[:, 1], cut_edges[:, 0])
    else:
        print(f"Unknown mode '{mode}'.")
        exit(0)

    return np.unique(tomove)


def move_node_along_edges(mesh, x, i, f):
    neigh = mesh.neigh[mesh.neigh_start[i]:mesh.neigh_start[i + 1]]
    if i in mesh.boundary_nodes:
        neigh = np.intersect1d(neigh, mesh.boundary_nodes)
    opposite_neigh = neigh[(f[neigh] > 0) != (f[i] > 0)]
    if opposite_neigh.size == 0:
        print("cannot move node !")
        return x[i]
    dist = np.linalg.norm(x[opposite_neigh] - x[i], axis=1)
    closest = np.argmin((f[opposite_neigh] - f[i]) / dist * np.sign(f[i]))
    target = opposite_neigh[closest]
    alpha = f[i] / (f[i] - f[target])
    return alpha * x[target] + (1 - alpha) * x[i]

# def move_node_in_triangle(mesh, x, i, f):
#     if i in mesh.boundary_nodes:
#         return move_node_along_edges(mesh, x, i, f)
#     targets = []
#     tri = mesh.node_el[mesh.node_el_start[i]:mesh.node_el_start[i+1]]
#     for j in tri:
#         dxdxi = [x[tri[1]]-x[tri[0]],x[tri[2]]-x[tri[0]]]
#         j = dxdxi[0][0]*dxdxi[1][1]-dxdxi[0][1]*dxdxi[1][0]
#         dxidx = [[dxdxi[1][1]/j,-dxdxi[0][1]/j],[-dxdxi[1][0]/j, dxdxi[0][0]/j]]
#     dist = np.linalg.norm(x[opposite_neigh]- x[i], axis=1)
#     closest = np.argmin((f[opposite_neigh]-f[i])/dist*np.sign(f[i]))
#     target = opposite_neigh[closest]
#     alpha = f[i]/(f[i]-f[target])
#     return alpha*x[target]+(1-alpha)*x[i]


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
        moved.append((x[i, 0], x_next[i, 0], x[i, 1], x_next[i, 1], 0, 0))
    return x_next, f_next, front_next, tomove, np.array(moved)


def levelset_f(x):
    x = x - 0.5
    return np.cos(omega) * x[:, 0] + np.sin(omega) * x[:, 1]


def fig_print(name):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(name)


def fig_phases(x, tri, f, figname):
    plt.gca().clear()
    plt.tricontourf(x[:, 0], x[:, 1], tri, f, cmap="bwr", levels=1)
    plt.triplot(x[:, 0], x[:, 1], tri, color="k")
    fig_print(figname)


def fig_disp(x, tri, f, candidates, moved, figname):
    plt.gca().clear()
    plt.tricontourf(x[:, 0], x[:, 1], tri, f, cmap="bwr", levels=1)
    plt.triplot(x[:, 0], x[:, 1], tri, color="k")
    xp = x[candidates]
    plt.scatter(xp[:, 0], xp[:, 1], 50, marker="o", edgecolors="k", facecolors="k")
    plt.quiver(moved[:, 0], moved[:, 2], moved[:, 1] - moved[:, 0], moved[:, 3] - moved[:, 2], width=0.008,
               headlength=4, headaxislength=3.5, fc="w", ec="k", lw=0.5, scale=1, scale_units="xy", zorder=10)
    fig_print(figname)


gmsh.model.occ.add_rectangle(0, 0, 0, 1, 1)
gmsh.model.mesh.set_size_callback(lambda *a: 0.1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

mesh = Mesh()
x = mesh.x.copy()
omega = np.pi / 2
f = levelset_f(x)
f_old = np.ones_like(f)
front_old = np.full(f.shape[0], False)
x, f, front, _, _ = move_front(mesh, x, f_old, f, front_old)
fig = plt.figure(figsize=(6, 6))


omega = np.pi * 0.35
f_old = f
fig_phases(x, mesh.el, f_old, "pdf/fig1.pdf")
f = levelset_f(x)
fig_phases(x, mesh.el, f, "pdf/fig2.pdf")
x_next, f_next, front_next, candidates, moved = move_front(mesh, x, f_old, f, front)

fig_disp(x, mesh.el, f, candidates, moved, "pdf/fig3.pdf")
fig_phases(x_next, mesh.el, f_next, "pdf/fig4.pdf")
x_next[~front_next] = mesh.x[~front_next]
fig_phases(x_next, mesh.el, f_next, "pdf/fig5.pdf")

gmsh.fltk.run()

for i in range(200):
    print(i)
    f_old = f
    omega += np.pi*0.005
    f = levelset_f(x)
    x, f, front, candidates, moved = move_front(mesh, x, f_old, f, front)
    x[~front] = mesh.x[~front]*0.2+x[~front]*0.8
    fig_phases(x, mesh.el, f, f"png/{i:03}.png")
