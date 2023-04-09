import gmsh
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from cvxopt import matrix, solvers
from Poiseuille_1D_SOCP import Simulation_1D, plot_solution_1D
from matplotlib.tri.triangulation import Triangulation

ftSz1, ftSz2, ftSz3 = 15, 13, 11

PHI = lambda xi, eta: 27. * xi * eta * (1. - xi - eta)
DPHI_DXI = lambda xi, eta: -27. * xi * eta + 27. * eta * (1. - xi - eta)
DPHI_DETA = lambda xi, eta: -27. * xi * eta + 27. * xi * (1. - xi - eta)


class Simulation_2D:
    def __init__(self, K, tau_zero, f, element, meshFilename, save):
        self.K = K  # Viscosity
        self.tau_zero = tau_zero  # yield stress
        self.f = f  # body force (pressure gradient)
        self.save = save  # Boolean

        self.meshFilename = "./mesh/" + meshFilename
        gmsh.open(self.meshFilename)

        if element == "taylor-hood":
            gmsh.model.mesh.setOrder(2)
            self.degree = 2
        elif element == "mini":
            gmsh.model.mesh.setOrder(1)
            self.degree = 3
        else:
            raise ValueError(f"Element '{element:s}' not implemented. Choose 'mini' or 'taylor-hood'")

        self.n_iterations = 0

        res = self.get_elements_info()
        self.elem_type, self.elem_tags, self.elem_node_tags, self.local_node_coords = res[0:4]
        self.weights, self.weights_q = res[4:6]
        self.v_shape_functions, self.dv_shape_functions_at_v = res[6:8]
        self.q_shape_functions, self.dv_shape_functions_at_q = res[8:10]
        self.inverse_jacobians, self.determinants = res[10:12]
        self.n_elem, self.ng_loc, self.ng_loc_q = len(self.elem_tags), len(self.weights), len(self.weights_q)
        self.ng_all = self.n_elem * self.ng_loc

        res = self.get_nodes_info()
        self.node_tags, self.coords, self.nodes_no_slip, self.nodes_inflow, self.nodes_outflow = res
        self.n_node = len(self.node_tags)

        # variables :  (u, v) at every node --- bounds on |.|^2 and |.|^1 at every gauss_pt
        self.n_var = (2 * self.n_node + 2 * self.ng_all)
        if element == "mini":  # extra dof at center of every element (for u, and for v)
            self.n_var += 2 * self.n_elem
        
        return
    
    def get_elements_info(self):
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, -1)
        elem_type = elem_types[0]
        elem_tags = elem_tags[0]
        n_elem = len(elem_tags)
        
        elem_node_tags = np.array(elem_node_tags[0]).astype(int) - 1  # nodes indices start at 1 in gmsh
        elem_node_tags = (elem_node_tags).reshape((n_elem, -1))  # matrix (n_elem, nb_node_per_elem)

        element_props = gmsh.model.mesh.getElementProperties(elem_type)
        elem_name, dim, order, n_local_node_v, local_node_coords, n_primary_nodes = element_props
        local_node_coords = np.array(local_node_coords).reshape((n_local_node_v, 2))
        local_node_coords = np.c_[local_node_coords, np.zeros(n_local_node_v)]  # append zero for z component
        local_node_coords = local_node_coords.flatten()
        
        n_local_node_q = 3

        _3_nodes_tri = 2
        _6_nodes_tri = 9

        # location of gauss points in 3d space, and associated weights VELOCITY FIELD
        deg = (self.degree - 1) * 2  # Taylor-Hood: 2, MINI: 4 
        uvw_space_v, weights_space_v = gmsh.model.mesh.getIntegrationPoints(_3_nodes_tri, "Gauss" + str(deg))
        weights_space_v, ng_loc_space_v = np.array(weights_space_v), len(weights_space_v)

        # location of gauss points in 3d space, and associated weights PRESSURE FIELD
        deg = (self.degree - 1) + 1  # Taylor-Hood: 2, MINI: 3
        uvw_space_q, weights_space_q = gmsh.model.mesh.getIntegrationPoints(_3_nodes_tri, "Gauss" + str(deg))
        weights_space_q, ng_loc_space_q = np.array(weights_space_q), len(weights_space_q)

        # sf for shape function
        _, sf, _ = gmsh.model.mesh.getBasisFunctions(elem_type, uvw_space_v, 'Lagrange')
        v_shape_functions = np.array(sf).reshape((ng_loc_space_v, n_local_node_v))

        _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(elem_type, uvw_space_v, 'GradLagrange')
        dv_shape_functions_at_v = np.array(dsfdu).reshape((ng_loc_space_v, n_local_node_v, 3))[:, :, :-1]

        _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(elem_type, uvw_space_q, 'GradLagrange')
        dv_shape_functions_at_q = np.array(dsfdu).reshape((ng_loc_space_q, n_local_node_v, 3))[:, :, :-1]

        _, sf, _ = gmsh.model.mesh.getBasisFunctions(_3_nodes_tri, uvw_space_q, 'Lagrange')
        q_shape_functions = np.array(sf).reshape((ng_loc_space_q, n_local_node_q))
        
        if self.degree == 3:  # MINI
            xi_eta_eval = np.array(uvw_space_v).reshape(ng_loc_space_v, 3)[:, :-1].T
            bubble_sf = PHI(*xi_eta_eval)  # eval bubble at gauss pts
            bubble_dsf = np.c_[DPHI_DXI(*xi_eta_eval), DPHI_DETA(*xi_eta_eval)]  # eval bubble derivatives at gauss pts
            v_shape_functions = np.c_[v_shape_functions, bubble_sf]
            dv_shape_functions_at_v = \
                np.append(dv_shape_functions_at_v, bubble_dsf.reshape((ng_loc_space_v, 1, 2)), axis=1)
            xi_eta_eval = np.array(uvw_space_q).reshape(ng_loc_space_q, 3)[:, :-1].T
            bubble_dsf = np.c_[DPHI_DXI(*xi_eta_eval), DPHI_DETA(*xi_eta_eval)]  # eval bubble derivatives at gauss pts
            dv_shape_functions_at_q = \
                np.append(dv_shape_functions_at_q, bubble_dsf.reshape((ng_loc_space_q, 1, 2)), axis=1)
            elem_node_tags = np.c_[elem_node_tags, np.arange(n_elem)]

        # jacobian is constant over the triangle
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(elem_type, [1. / 3., 1. / 3., 1. / 3.])
        jacobians = np.array(jacobians).reshape((n_elem, 3, 3))
        jacobians = np.swapaxes(jacobians[:, :-1, :-1], 1, 2)
        # [[dX_xi, dX_eta],
        #  [dY_xi, dY_eta]]

        determinants = np.array(determinants)

        inv_jac = np.empty_like(jacobians)  # trick to inverse 2x2 matrix
        inv_jac[:, 0, 0] = +jacobians[:, 1, 1]
        inv_jac[:, 0, 1] = -jacobians[:, 0, 1]
        inv_jac[:, 1, 0] = -jacobians[:, 1, 0]
        inv_jac[:, 1, 1] = +jacobians[:, 0, 0]

        return (
            elem_type, elem_tags, elem_node_tags, local_node_coords,
            weights_space_v, weights_space_q,
            v_shape_functions, dv_shape_functions_at_v, q_shape_functions, dv_shape_functions_at_q,
            inv_jac, determinants
        )
    
    def get_nodes_info(self):
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        nodeTags = np.array(nodeTags) - 1
        coords = np.array(coords).reshape((-1, 3))[:, :-1]

        bd_nodes_00, coords_00 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=3)  # no_slip
        # bd_nodes_01, coords_01 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=1)  # inflow
        # bd_nodes_02, coords_02 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=2)  # outflow
        bd_nodes_10, coords_10 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=3)  # no-slip
        bd_nodes_11, coords_11 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=1)  # inflow
        bd_nodes_12, coords_12 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=2)  # outflow

        # setdiff1d to handle corners
        nodes_no_slip = np.setdiff1d(bd_nodes_10, []).astype(int) - 1
        nodes_inflow = np.setdiff1d(bd_nodes_11, bd_nodes_00).astype(int) - 1
        nodes_outflow = np.setdiff1d(bd_nodes_12, bd_nodes_00).astype(int) - 1
        
        # print("NO SLIP")
        # print(nodes_no_slip+1)
        # print("\nINFLOW")
        # print(nodes_inflow+1)
        # print("\nOUTFLOW")
        # print(nodes_outflow+1)
        # gmsh.fltk.initialize()
        # gmsh.fltk.run()
        # exit(0)
        
        return nodeTags, coords, nodes_no_slip, nodes_inflow, nodes_outflow, 


def solve_FE(sim: Simulation_2D, atol=1e-8, rtol=1e-6):
    
    ng_all = sim.ng_all
    IB = 2 * sim.n_node  # start of dofs related to bubble function
    IS = IB + sim.n_elem if sim.degree == 3 else IB  # start of S variables
    IT = IS + ng_all  # start of T variables
    
    if sim.degree == 3:
        sim.elem_node_tags[:, -1] += sim.n_node

    # coefficients of linear minimization function
    cost = np.zeros(sim.n_var)
    
    # set constraints (1) div(u) = 0 at every gauss_pt, (2) u,v = U,V on boundary
    nb_constraints_bd = 2 * len(sim.nodes_no_slip) + len(sim.nodes_inflow) + len(sim.nodes_outflow)
    A = np.zeros((sim.n_node, sim.n_var))
    node_is_vertex_list = np.zeros(sim.n_node)

    # Old method, where div(u) = 0 at every gauss point
    # I_div = 0
    # I_bnd = ng_all
    # A = np.zeros((ng_all + nb_constraints_bd, sim.n_var))
    # b = np.zeros(A.shape[0])

    # set SOCP constraints
    I_yield = 5 * ng_all
    G = np.zeros((9 * ng_all, sim.n_var))
    h = np.zeros(G.shape[0])
    sqrt2 = np.sqrt(2.)
    
    for i in range(sim.n_elem):
        
        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        
        for g, wg in enumerate(sim.weights_q):
            psi = sim.q_shape_functions[g]
            dphi = sim.dv_shape_functions_at_q[g]
            dphi = np.dot(dphi, inv_jac) / det

            # pressure field is P1 --> multiply div(velocity) by the linear shape functions of the 3 VERTICES
            for j, idx_node in enumerate(idx_local_nodes[:3]):  
                node_is_vertex_list[idx_node] = 1
                A[idx_node, 2*idx_local_nodes+0] += wg * psi[j] * dphi[:, 0] * det
                A[idx_node, 2*idx_local_nodes+1] += wg * psi[j] * dphi[:, 1] * det

        for g, wg in enumerate(sim.weights):

            sf = sim.v_shape_functions[g]  # size (n_sf)
            dphi = sim.dv_shape_functions_at_v[g]  # size (n_sf, 2)
            dphi = np.dot(dphi, inv_jac) / det # size (n_sf, 2)

            i_g_idx = i * sim.ng_loc + g
            cost[IS + i_g_idx] += sim.K / 2. * wg * det
            cost[IT + i_g_idx] += sim.tau_zero * wg * det

            for j, idx_node in enumerate(idx_local_nodes):

                u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
                cost[u_idx] -= wg * sim.f[0] * sf[j] * det
                cost[v_idx] -= wg * sim.f[1] * sf[j] * det

                # TOO MANY CONSTRAINTS
                # # set divergence free
                # A[I_div + i_g_idx, u_idx] += dphi[j, 0]
                # A[I_div + i_g_idx, v_idx] += dphi[j, 1]

                # set |2D|^2 < Sig
                G[5 * i_g_idx + 2, u_idx] = -sqrt2 * dphi[j, 0]  # sqrt(2) [du_dx] = s3
                G[5 * i_g_idx + 3, v_idx] = -sqrt2 * dphi[j, 1]  # sqrt(2) [dv_dy] = s4
                G[5 * i_g_idx + 4, u_idx] = -1. * dphi[j, 1]  # [du_dy] +
                G[5 * i_g_idx + 4, v_idx] = -1. * dphi[j, 0]  #         + [dv_dx] = s5

                # set |2D|^1 < Tig
                G[I_yield + 4 * i_g_idx + 1, u_idx] = -sqrt2 * dphi[j, 0]  # sqrt(2) [du_dx] = s2
                G[I_yield + 4 * i_g_idx + 2, v_idx] = -sqrt2 * dphi[j, 1]  # sqrt(2) [dv_dy] = s3
                G[I_yield + 4 * i_g_idx + 3, u_idx] = -1. * dphi[j, 1]  # [du_dy] +
                G[I_yield + 4 * i_g_idx + 3, v_idx] = -1. * dphi[j, 0]  #         + [dv_dx] = s4

            # set |2D|^2 < Sig
            G[5 * i_g_idx + 0, IS + i_g_idx] = -1. / sqrt2  # (Sig + 0.5) / sqrt2 = s1
            h[5 * i_g_idx + 0] = +0.5 / sqrt2  # (Sig + 0.5) / sqrt2 = s1
            G[5 * i_g_idx + 1, IS + i_g_idx] = -1. / sqrt2  # (Sig - 0.5) / sqrt2 = s2
            h[5 * i_g_idx + 1] = -0.5 / sqrt2  # (Sig - 0.5) / sqrt2 = s2

            # set |2D|^1 < Tig
            G[I_yield + 4 * i_g_idx + 0, IT + i_g_idx] = -1.  # Tig = s1
            # Could also copy paste the submatrix used for the viscous term, instead of filling it in the loop
            # G[I_yield + 4 * i_g_idx + 1: I_yield + 4 * (i_g_idx + 1), u_idx: v_idx + 1] = \
            #         G[5 * i_g_idx + 2: 5 * (i_g_idx + 1), u_idx: v_idx + 1]

    
    A = A[np.argwhere(node_is_vertex_list).flatten()]  # take only the non-zero lines of the matrix A
    a = np.zeros(A.shape[0])  # A x = a
    B = np.zeros((nb_constraints_bd, sim.n_var))
    b = np.zeros(B.shape[0])  # B x = b

    idx_bd_condition = 0
    # set boundary conditions inflow
    for idx_node in sim.nodes_inflow:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        # B[idx_bd_condition, u_idx] = 1.
        # b[idx_bd_condition] = U_in
        # idx_bd_condition += 1
        B[idx_bd_condition, v_idx] = 1.
        b[idx_bd_condition] = 0.
        idx_bd_condition += 1
        # print(f"inflow {idx_node + 1:3d}")

    # set boundary conditions outflow
    for idx_node in sim.nodes_outflow:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        # B[idx_bd_condition, u_idx] = 1.
        # b[idx_bd_condition] = U_out
        # idx_bd_condition += 1
        B[idx_bd_condition, v_idx] = 1.
        b[idx_bd_condition] = 0.
        idx_bd_condition += 1
        # print(f"outflow {idx_node + 1:3d}")
    
    # set boundary conditions no-slip
    for idx_node in sim.nodes_no_slip:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        B[idx_bd_condition, u_idx] = 1.
        idx_bd_condition += 1
        B[idx_bd_condition, v_idx] = 1.
        idx_bd_condition += 1
        # print(f"no-slip {idx_node + 1:3d}")

    # print(f"U : {0:3d} -> {ID-1:3d}")
    # print(f"D : {ID:3d} -> {IS-1:3d}")
    # print(f"S : {IS:3d} -> {IT-1:3d}")
    # print(f"T : {IT:3d} -> {IT+ng_all-1:3d}")
    # plt.spy(np.r_[A, B])
    # plt.spy(G, markersize=0.5, aspect='auto')
    # plt.show()
    # exit(0)

    # print(np.linalg.matrix_rank(np.r_[A, B]))
    # print(np.shape(np.r_[A, B]))
    # exit(0)

    cost, G, h, A, b = matrix(cost), matrix(G), matrix(h), matrix(np.r_[A, B]), matrix(np.r_[a, b])
    dims = {'l':0, 'q': [5 for i in range(ng_all)] + [4 for i in range(ng_all)], 's': []}

    solvers.options['abstol'] = atol
    solvers.options['reltol'] = rtol
    # solvers.options['maxiters'] = 1
    start_time = perf_counter()
    res = solvers.conelp(cost, G, h, dims, A, b)
    end_time = perf_counter()

    print(f"Time to solve conic optimization = {end_time-start_time:.2f}")
    print(f"Number variables of the problem  = {sim.n_var:d}")

    u_num = np.array(res['x'])[:IB].reshape((sim.n_node, 2))
    u_bbl = np.array(res['x'])[IB:IS].reshape((sim.n_elem * (sim.degree == 3), 2))
    s_num = np.array(res['x'])[IS:IT].reshape((sim.n_elem, sim.ng_loc))
    t_num = np.array(res['x'])[IT:].reshape((sim.n_elem, sim.ng_loc))

    return u_num, s_num, t_num


def solve_interface_tracking(sim: Simulation_2D, atol=1e-8, rtol=1e-6):
    return


def plot_solution_2D(u_num, sim: Simulation_2D):

    def eval_strain_norm(u_local, dphi_local):
        dudx = np.dot(u_local[:, 0], dphi_local[:, 0])
        dudy = np.dot(u_local[:, 0], dphi_local[:, 1])
        dvdx = np.dot(u_local[:, 1], dphi_local[:, 0])
        dvdy = np.dot(u_local[:, 1], dphi_local[:, 1]) 
        return np.sqrt(2 * dudx ** 2 + 2 * dvdy ** 2 + (dudy + dvdx) ** 2)

    # all_nodes = sim.n_node + sim.n_elem * (sim.degree == 3)
    velocity = np.c_[u_num, np.zeros_like(u_num[:, 0])].reshape((sim.n_node, -1))
    # strain_norm_node = np.zeros((*sim.elem_node_tags.shape, 1))
    # strain_norm_elem = np.zeros(sim.n_elem)

    # n_local_node = strain_norm_node.shape[1]
    # _, dsf_at_nodes, _ = gmsh.model.mesh.getBasisFunctions(sim.elem_type, sim.local_node_coords, 'GradLagrange')
    # dsf_at_nodes = np.array(dsf_at_nodes).reshape((n_local_node, n_local_node, 3))[:, :, :-1]

    # # if sim.degree == 3:
    # #     xi_eta_eval = np.array(sim.local_node_coords).reshape(n_local_node, 3)[:, :-1].T
    # #     bubble_dsf = np.c_[DPHI_DXI(*xi_eta_eval), DPHI_DETA(*xi_eta_eval)]  # eval bubble derivatives at node pts
    # #     dsf_at_nodes = np.append(dsf_at_nodes, bubble_dsf.reshape((n_local_node, 1, 2)), axis=1)

    # for i in range(sim.n_elem):
    #     idx_local_nodes = sim.elem_node_tags[i]
    #     det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        
    #     for j, idx_node in enumerate(idx_local_nodes):
            
    #         dphi = dsf_at_nodes[j, :]  # dphi in reference element
    #         dphi = np.dot(dphi, inv_jac) / det  # dphi in physical element
    #         strain_norm_node[i, j] = eval_strain_norm(u_num[idx_local_nodes], dphi)

    #     for g, wg in enumerate(sim.weights):

    #         dphi = sim.dv_shape_functions_at_v[g]  # size (n_sf, 2)
    #         dphi = np.dot(dphi, inv_jac) / det # size (n_sf, 2)
    #         strain_norm_elem[i] += wg * eval_strain_norm(u_num[idx_local_nodes], dphi)

    # strain_norm_elem = strain_norm_elem.flatten()
    # strain_norm_node = strain_norm_node.flatten()
    velocity = velocity.flatten()
    # # strain_norm_elem = strain_norm_elem.reshape((sim.n_elem, -1))
    # # strain_norm_node = strain_norm_node.reshape((strain_norm_node.size, -1))
    # # strain_data_elem = [list(vector) for vector in strain_norm_elem]
    # # strain_data_node = [list(vector) for vector in strain_norm_node]
    # # velocity_data = [list(vector) for vector in velocity]

    gmsh.fltk.initialize()
    tag_v = gmsh.view.add("Velocity")
    # tag_d1 = gmsh.view.add("|D| - by element")
    # tag_d2 = gmsh.view.add("|D| - by node")
    modelName = gmsh.model.list()[0]

    gmsh.view.addHomogeneousModelData(
        tag_v, 0, modelName, "NodeData", sim.node_tags + 1, velocity, numComponents=3)
    # gmsh.view.addHomogeneousModelData(
    #     tag_d1, 0, modelName, "ElementData", sim.elem_tags, strain_norm_elem, numComponents=1)
    # gmsh.view.addHomogeneousModelData(
    #     tag_d2, 0, modelName, "ElementNodeData", sim.elem_tags, strain_norm_node, numComponents=1)

    gmsh.option.setNumber("View.AdaptVisualizationGrid", 1)
    gmsh.option.setNumber("View.MaxRecursionLevel", 3)
    gmsh.option.setNumber("View.TargetError", 0.)
    gmsh.option.setNumber("View.VectorType", 6)
    gmsh.option.setNumber("View[0].NormalRaise", -2.)
    # gmsh.option.setNumber("View[1].Visible", 0)
    # gmsh.option.setNumber("View[2].Visible", 0)

    gmsh.fltk.run()
    return


def plot_solution_2D_matplotlib(u_num, sim: Simulation_2D):

    fig, ax = plt.subplots(1, 1, figsize=(10., 6.), constrained_layout=True)

    n_elem, n_node_per_elem = sim.elem_node_tags.shape
    coords = sim.coords  # size (n_nodes, 2)
    
    triang = Triangulation(sim.coords[:, 0], sim.coords[:, 1], sim.elem_node_tags[:, :3])
    tricontourset = ax.tricontourf(triang, u_num[:, 0])
    ax.triplot(triang, 'ko-', alpha=0.5)
    _ = fig.colorbar(tricontourset)

    ax.quiver(coords[:, 0], coords[:, 1], u_num[:, 0], u_num[:, 1])

    plt.show()
    
    return


def plot_1D_slice(u_num, sim: Simulation_2D):
    
    slice_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=4)
    slice_node_tags = np.array(slice_node_tags).astype(int) - 1

    # print(slice_node_tags + 1)
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    # exit(0)
    
    slice_xy = sim.coords[slice_node_tags]
    slice_y = slice_xy[:, 1]
    slice_u = u_num[slice_node_tags, 0]  # only u component of (u, v)
    H = np.amax(slice_y)

    arg_sorted_tags = np.argsort(slice_y)
    slice_u = slice_u[arg_sorted_tags]

    if sim.degree == 1:
        slice_y = slice_y[arg_sorted_tags]
        n_intervals = len(slice_node_tags) - 1
    else:
        slice_y = slice_y[arg_sorted_tags][::2]
        n_intervals = (len(slice_node_tags) - 1) // 2
    
    # print(slice_y)
    # print(slice_u)
    # print("n intervals = ", n_intervals)
    # exit(0)

    sim_1D = Simulation_1D(
        H=H, K=sim.K, tau_zero=sim.tau_zero, f=sim.f[0], deg=sim.degree,
        nElem=n_intervals, random_seed=-1, fix_interface=False, save=False
    )
    sim_1D.set_y(slice_y)
    plot_solution_1D(sim=sim_1D, u_nodes=slice_u, pts_per_elem=50)
    return


if __name__ == "__main__":

    gmsh.initialize()
    sim = Simulation_2D(
        K=1., tau_zero=0.3, f=(1., 0.), element="taylor-hood", meshFilename="rect_coarse.msh", save=False
        # K=1., tau_zero=0.3, f=(1., 0.), element="mini", meshFilename="rect_coarse.msh", save=False
    )

    # Solve the problem ITERATE
    # u_nodes = solve_interface_tracking(sim, atol=1e-12, rtol=1e-10)
    
    # Solve problem ONE SHOT
    u_nodes, s_num, t_num = solve_FE(sim, atol=1e-10, rtol=1e-8)

    # DUMB solution to debug
    # u_nodes = np.zeros((sim.n_node, 2))
    # u_nodes[:, 0] = 1. - sim.coords[:, 1]**2
    
    plot_solution_2D(u_nodes, sim)
    plot_1D_slice(u_nodes, sim)
    # plot_solution_2D_matplotlib(u_nodes, sim)

    gmsh.finalize()
