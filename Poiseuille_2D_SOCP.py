import gmsh
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from cvxopt import matrix, solvers

ftSz1, ftSz2, ftSz3 = 15, 13, 11

class Simulation:
    def __init__(self, H, K, tau_zero, f, deg, meshFilename, save):
        self.H = H  # Half-channel width
        self.K = K  # Viscosity
        self.tau_zero = tau_zero  # yield stress
        self.f = f  # body force (pressure gradient)
        self.save = save  # Boolean

        # Reference velocity imposed by (1) pressure gradient, (2) channel width, (3) viscosity
        self.V = np.hypot(*self.f) * (self.H * self.H) / (2. * self.K)
        self.y0 = self.tau_zero / np.hypot(*self.f)
        self.Bn = self.tau_zero * self.H / (self.K * self.V)

        self.degree = deg
        self.meshFilename = "./mesh/" + meshFilename

        self.n_iterations = 0

        # self.nElem = nElem
        # if deg == 1:
        #     self.nVert = nElem + 1
        #     self.nG = 1
        #     self.xG = xG_P1
        #     self.wG = wG_P1
        #     self.PHI = PHI_P1
        #     self.DPHI = DPHI_P1
        # elif deg == 2:
        #     self.nVert = 2 * nElem + 1
        #     self.nG = 2  # quad shape fcts -> two gauss point needed
        #     self.xG = xG_P2
        #     self.wG = wG_P2
        #     self.PHI = PHI_P2
        #     self.DPHI = DPHI_P2
        # else:
        #     raise ValueError
        # self.nVar = self.nVert + 2 * self.nG * nElem
        # # velocities --- bounds on viscosity term --- bounds on yield-stress term

    def set_y(self, new_y):
        self.y = new_y
        self.dy = np.diff(self.y)
        self.ym = (self.y[:-1] + self.y[1:]) / 2.

    def set_reconstruction(self, dudy_reconstructed):
        self.dudy_reconstructed = dudy_reconstructed


def get_elements_info(sim):
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, -1)
    sim.element_type = elem_types[0]
    sim.nb_elem = len(elem_tags[0])
    sim.elem_node_tags = np.array(elem_node_tags[0]).astype(int) - 1  # nodes indices start at 1 in gmsh
    sim.elem_node_tags = (sim.elem_node_tags).reshape((sim.nb_elem, -1))  # matrix (nb_elem, nb_node_per_elem)

    # Name (?), dimension, order, nb vertices / element
    name, dim, order, nb_node_per_elem, _, _ = gmsh.model.mesh.getElementProperties(sim.element_type)
    sim.nb_node_local = nb_node_per_elem

    # location of gauss points in 3d space, and associated weights
    uvw, weights = gmsh.model.mesh.getIntegrationPoints(sim.element_type, "Gauss" + str(order))
    weights, nb_gauss_pts = np.array(weights), len(weights)
    sim.nb_gauss_pts = nb_gauss_pts
    sim.weights = weights

    # sf for shape function
    _, sf, _ = gmsh.model.mesh.getBasisFunctions(sim.element_type, uvw, 'Lagrange')
    sim.shape_functions = np.array(sf).reshape((sim.nb_gauss_pts, nb_node_per_elem))

    _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(sim.element_type, uvw, 'GradLagrange')
    sim.d_shape_functions = np.array(dsfdu).reshape((sim.nb_gauss_pts, nb_node_per_elem, 3))[:, :, :-1]

    jacobians, determinants, _ = gmsh.model.mesh.getJacobians(sim.element_type, [1. / 3., 1. / 3., 1. / 3.])
    jacobians = np.array(jacobians).reshape((sim.nb_elem, 3, 3))
    jacobians = np.swapaxes(jacobians[:, :-1, :-1], 1, 2)  
    # [[dX_xi, dX_eta],
    #  [dY_xi, dY_eta]]

    sim.determinants = np.array(determinants)

    inv_jac = np.empty_like(jacobians)  # trick to inverse 2x2 matrix
    inv_jac[:, 0, 0] = +jacobians[:, 1, 1]
    inv_jac[:, 0, 1] = -jacobians[:, 0, 1]
    inv_jac[:, 1, 0] = -jacobians[:, 1, 0]
    inv_jac[:, 1, 1] = +jacobians[:, 0, 0]
    sim.inverse_jacobians = inv_jac

    return sim.nb_elem, nb_node_per_elem, elem_node_tags[0], inv_jac, determinants, sim.element_type, order


def get_nodes_info(sim):
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    nodeTags = np.array(nodeTags) - 1
    coords = np.array(coords).reshape((-1, 3))[:, :-1]
    sim.nb_node = len(nodeTags)

    bd_nodes_00, coords_00 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=0)  # no_slip
    bd_nodes_01, coords_01 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=1)  # inflow
    bd_nodes_02, coords_02 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=2)  # outflow
    bd_nodes_10, coords_10 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=0)  # no-slip
    bd_nodes_11, coords_11 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=1)  # inflow
    bd_nodes_12, coords_12 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=2)  # outflow

    sim.nodes_no_slip = np.setdiff1d(bd_nodes_10, np.r_[bd_nodes_01, bd_nodes_02]).astype(int) - 1
    sim.nodes_inflow = np.setdiff1d(bd_nodes_11, np.r_[bd_nodes_00, bd_nodes_02]).astype(int) - 1
    sim.nodes_outflow = np.setdiff1d(bd_nodes_12, np.r_[bd_nodes_00, bd_nodes_01]).astype(int) - 1
    
    return len(nodeTags), nodeTags, coords


def solve_FE(sim, atol=1e-8, rtol=1e-6):
    gmsh.initialize()
    gmsh.open(sim.meshFilename)
    gmsh.model.mesh.setOrder(sim.degree)

    get_elements_info(sim)
    get_nodes_info(sim)
    
    ng_global = sim.nb_elem * sim.nb_gauss_pts

    sim.nb_var = (2 * sim.nb_node   # (u, v) at every node
                  + 3 * ng_global  # (du_dx, dv_dy, du_dy+dv_dx) at every gauss_pt
                  + 2 * ng_global  # bounds on |.|^2 and |.|^1 at every gauss_pt
                  )
    ID = 2 * sim.nb_node  # start of D variables
    IS = ID + 3 * ng_global  # start of S variables
    IT = ID + 4 * ng_global  # start of T variables

    # coefficients of linear minimization function
    cost = np.zeros(sim.nb_var)
    
    # set constraints (1) D = ..., (2) div(u) = 0 at every gauss_pt, (3) u,v = U,V on boundary
    I_div = 3 * ng_global
    I_bnd = 4 * ng_global
    nb_constraints_bd = 2 * len(sim.nodes_no_slip) + len(sim.nodes_inflow) + len(sim.nodes_outflow)
    A = np.zeros((4 * ng_global + nb_constraints_bd, sim.nb_var))  
    b = np.zeros(A.shape[0])

    # set SOCP constraints
    I_yield = 5 * ng_global
    G = np.zeros((9 * ng_global, sim.nb_var))
    h = np.zeros(G.shape[0])
    sqrt2 = np.sqrt(2.)
    
    for i in range(sim.nb_elem):
        
        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        
        for g, wg in enumerate(sim.weights):

            sf = sim.shape_functions[g]  # size (nb_node_local)
            dsf = sim.d_shape_functions[g]  # size (nb_node_local, 2)
            dphi = np.dot(dsf, inv_jac) / det # size (nb_node_local, 2)

            i_g_idx = i * sim.nb_gauss_pts + g
            d_idx = 3 * i_g_idx
            cost[IS + i_g_idx] += sim.K / 2. * wg * det
            cost[IT + i_g_idx] += sim.tau_zero * wg * det

            for j, idx_node in enumerate(idx_local_nodes):

                u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
                cost[u_idx] -= wg * sim.f[0] * sf[j] * det
                cost[v_idx] -= wg * sim.f[1] * sf[j] * det

                # set redudant variables d_{xx}, d_{yy}, 2d_{xy}
                A[d_idx + 0, ID + d_idx + 0] -= 1. # D_{i,g,0} = du_dx [part one]
                A[d_idx + 0, u_idx] += dphi[j, 0] # D_{i,g,0} = du_dx [part two]
                A[d_idx + 1, ID + d_idx + 1] -= 1. # D_{i,g,1} = dv_dy [part one]
                A[d_idx + 1, v_idx] += dphi[j, 1] # D_{i,g,1} = dv_dy [part two]
                A[d_idx + 2, ID + d_idx + 2] -= 1. # D_{i,g,2} = du_dy + dv_dx [part one]
                A[d_idx + 2, u_idx] += dphi[j, 1] # D_{i,g,2} = du_dy + ... [part two]
                A[d_idx + 2, v_idx] += dphi[j, 0] # D_{i,g,2} = ... + dv_dx [part three]

            # set divergence free
            A[I_div + i_g_idx, ID + d_idx + 0] = 1.
            A[I_div + i_g_idx, ID + d_idx + 1] = 1.

            # set |2D|^2 < Sig
            G[5 * i_g_idx + 0, IS + i_g_idx] = -1. / sqrt2  # (Sig + 0.5) / sqrt2 = s1
            h[5 * i_g_idx + 0] = +0.5 / sqrt2  # (Sig + 0.5) / sqrt2 = s1

            G[5 * i_g_idx + 1, IS + i_g_idx] = -1. / sqrt2  # (Sig - 0.5) / sqrt2 = s2
            h[5 * i_g_idx + 1] = -0.5 / sqrt2  # (Sig - 0.5) / sqrt2 = s2

            G[5 * i_g_idx + 2, ID + d_idx + 0] = -sqrt2  # sqrt(2) du_dx = s3
            G[5 * i_g_idx + 3, ID + d_idx + 1] = -sqrt2  # sqrt(2) dv_dy = s4
            G[5 * i_g_idx + 4, ID + d_idx + 2] = -1.  # du_dy + dv_dx = s5

            # set |2D|^1 < Tig
            G[I_yield + 4 * i_g_idx + 0, IT + i_g_idx] = -1.  # Tig = s1
            G[I_yield + 4 * i_g_idx + 1, ID + d_idx + 0] = -sqrt2  # sqrt(2) du_dx = s2
            G[I_yield + 4 * i_g_idx + 2, ID + d_idx + 1] = -sqrt2  # sqrt(2) dv_dy = s3
            G[I_yield + 4 * i_g_idx + 3, ID + d_idx + 2] = -1.  # du_dy + dv_dx = s4

    idx_bd_condition = I_bnd
    # set boundary conditions inflow
    for idx_node in sim.nodes_inflow:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        # A[idx_bd_condition, u_idx] = 1.
        # b[idx_bd_condition] = U_in
        # idx_bd_condition += 1
        A[idx_bd_condition, v_idx] = 1.
        b[idx_bd_condition] = 0.
        idx_bd_condition += 1
        # print(f"inflow {idx_node:3d}")

    # set boundary conditions outflow
    for idx_node in sim.nodes_outflow:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        # A[idx_bd_condition, u_idx] = 1.
        # b[idx_bd_condition] = U_out
        # idx_bd_condition += 1
        A[idx_bd_condition, v_idx] = 1.
        b[idx_bd_condition] = 0.
        idx_bd_condition += 1
        # print(f"outflow {idx_node:3d}")
    
    # set boundary conditions no-slip
    for idx_node in sim.nodes_no_slip:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        A[idx_bd_condition, u_idx] = 1.
        idx_bd_condition += 1
        A[idx_bd_condition, v_idx] = 1.
        idx_bd_condition += 1
        # print(f"no-slip {idx_node:3d}")

    # print(f"U : {0:3d} -> {ID-1:3d}")
    # print(f"D : {ID:3d} -> {IS-1:3d}")
    # print(f"S : {IS:3d} -> {IT-1:3d}")
    # print(f"T : {IT:3d} -> {IT+ng_global-1:3d}")
    # plt.spy(G)
    # plt.show()

    cost, G, h, A, b = matrix(cost), matrix(G), matrix(h), matrix(A), matrix(b)
    dims = {'l':0, 'q': [5 for i in range(ng_global)] + [4 for i in range(ng_global)], 's': []}
    # dims = {'l':0, 'q': [5 for i in range(ng_global)], 's': []}

    # solvers.options['abstol'] = 1.e-6
    # solvers.options['reltol'] = 1.e-4
    start_time = perf_counter()
    res = solvers.conelp(cost, G, h, dims, A, b)
    end_time = perf_counter()

    print(f"Time to solve conic optimization = {end_time-start_time:.2f}")
    print(f"Number variables of the problem  = {sim.nb_var:d}")


    u_num = np.array(res['x'])[:ID].reshape((sim.nb_node, 2))
    d_num = np.array(res['x'])[ID:IS].reshape((sim.nb_elem, sim.nb_gauss_pts, 3))
    s_num = np.array(res['x'])[IS:IT].reshape((sim.nb_elem, sim.nb_gauss_pts))
    # t_num = np.array(res['x'])[IT:].reshape((sim.nb_elem, sim.nb_gauss_pts))

    # print(u_num[:, 0])
    # print(u_num[:, 1])

    #####################
    gmsh.fltk.initialize()
    viewTag = gmsh.view.add("ux")
    modelName = gmsh.model.list()[0]
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    # print(node_tags)
    data = np.c_[u_num, np.zeros_like(u_num[:, 0])].reshape((sim.nb_node, -1))
    gmsh.view.addModelData(viewTag, 0, modelName, "NodeData", node_tags, data, numComponents=3)
    gmsh.fltk.run()

    gmsh.finalize()

    return u_num, d_num, s_num
    # return u_num, d_num, s_num, t_num


def solve_interface_tracking(sim, atol=1e-8, rtol=1e-6):
    return


def plot_solution(sim, u_nodes, pts_per_elem=50):
    return


if __name__ == "__main__":

    sim = Simulation(H=1., K=1., tau_zero=0.3, f=(1., 0.), deg=2, meshFilename="hole.msh", save=False)
    
    # Solve the problem ITERATE
    # u_nodes = solve_interface_tracking(sim, atol=1e-12, rtol=1e-10)
    
    # Solve problem ONE SHOT
    solve_FE(sim, atol=1e-12, rtol=1e-10)
    # u_nodes, s_num, t_num = solve_FE(sim, atol=1e-12, rtol=1e-10)
    
    # plot_solution(sim, u_nodes, pts_per_elem=150)
