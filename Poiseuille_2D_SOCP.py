import gmsh
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from cvxopt import matrix, solvers
from Poiseuille_1D_SOCP import Simulation_1D, plot_solution_1D
from matplotlib.tri.triangulation import Triangulation
from tqdm import tqdm

ftSz1, ftSz2, ftSz3 = 15, 13, 11

PHI = lambda xi, eta: 27. * xi * eta * (1. - xi - eta)
DPHI_DXI = lambda xi, eta: -27. * xi * eta + 27. * eta * (1. - xi - eta)
DPHI_DETA = lambda xi, eta: -27. * xi * eta + 27. * xi * (1. - xi - eta)


class Simulation_2D:
    def __init__(self, K, tau_zero, f, element, mesh_filename):
        self.K = K  # Viscosity
        self.tau_zero = tau_zero  # yield stress
        self.f = f  # body force (pressure gradient)

        self.mesh_filename = mesh_filename
        gmsh.open("./mesh/" + mesh_filename + ".msh")

        self.element = element
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
        self.node_tags, self.coords = res[0:2]
        self.nodes_zero_u, self.nodes_zero_v, self.nodes_with_u = res[2:5]
        self.primary_nodes, = res[5:]
        self.n_node = len(self.node_tags)

        # variables :  (u, v) at every node --- bounds on |.|^2 and |.|^1 at every gauss_pt
        self.n_var = 2 * self.n_node + 2 * self.ng_all if self.tau_zero > 0. else 2 * self.n_node + 1 * self.ng_all
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
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.array(node_tags) - 1
        coords = np.array(coords).reshape((-1, 3))[:, :-1]

        # bd_nodes_01, coords_01 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=1)  # zero u
        # bd_nodes_02, coords_02 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=2)  # zero v
        # bd_nodes_03, coords_03 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=3)  # impose non-zero u
        bd_nodes_05, coords_05 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=5)  # singular pressure

        bd_nodes_11, coords_11 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=1)  # zero u
        bd_nodes_12, coords_12 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=2)  # zero v
        bd_nodes_13, coords_13 = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=3)  # impose non-zero u

        nodes_zero_u = bd_nodes_11.astype(int) - 1
        nodes_zero_v = bd_nodes_12.astype(int) - 1
        nodes_with_u = bd_nodes_13.astype(int) - 1
        nodes_dont_check = bd_nodes_05.astype(int) - 1
        # nodes_dont_check = np.array([], dtype=int)
        
        # print("Zero u")
        # print(nodes_zero_u+1)
        # print("\nZero v")
        # print(nodes_zero_v+1)
        # print("\nWith u")
        # print(nodes_with_u+1)
        # gmsh.fltk.initialize()
        # gmsh.fltk.run()
        # exit(0)

        node_is_vertex_list = np.zeros(len(node_tags))
        for i in range(self.n_elem):
            idx_local_nodes = self.elem_node_tags[i]
            node_is_vertex_list[idx_local_nodes[:3]] = 1
        
        node_is_vertex_list[nodes_dont_check] = 0
        primary_nodes = np.argwhere(node_is_vertex_list).flatten()
        
        return node_tags, coords, nodes_zero_u, nodes_zero_v, nodes_with_u, primary_nodes

    def save_solution(self, u_num):
        with open(f"./res/{self.mesh_filename:s}.txt", 'w') as file:
            file.write(f"{self.K:.6e}\n")
            file.write(f"{self.tau_zero:.6e}\n")
            file.write(f"{self.f[0]:.6e} {self.f[1]:.6e}\n")
            file.write(f"{self.element:s}\n")
            file.write(f"{self.mesh_filename:s}\n")
            np.savetxt(file, u_num, fmt="%.6e")
        return
    

def load_solution(res_file_name):
    with open(f"./res/{res_file_name:s}.txt", 'r') as file:
        K, tau_zero = float(next(file).strip('\n')), float(next(file).strip('\n')),
        f = [float(component) for component in next(file).strip('\n').split(' ')]
        element, mesh_filename = next(file).strip('\n'), next(file).strip('\n')
        u_num = np.loadtxt(file)
    
    return dict(K=K, tau_zero=tau_zero, f=f, element=element, mesh_filename=mesh_filename), u_num




def set_boundary_conditions(sim: Simulation_2D, B, b):
    idx_bd_condition = 0

    for idx_node in sim.nodes_zero_u:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        B[idx_bd_condition, u_idx] = 1.
        b[idx_bd_condition] = 0.
        idx_bd_condition += 1
    #     print(f"node {idx_node + 1:3d} : u = 0")
    # print("")

    for idx_node in sim.nodes_zero_v:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        B[idx_bd_condition, v_idx] = 1.
        b[idx_bd_condition] = 0.
        idx_bd_condition += 1
    #     print(f"node {idx_node + 1:3d} : v = 0")
    # print("")
    
    for idx_node in sim.nodes_with_u:
        u_idx, v_idx = 2*idx_node + 0, 2*idx_node + 1
        B[idx_bd_condition, u_idx] = 1.
        b[idx_bd_condition] = 1.
        # b[idx_bd_condition] = np.sin(np.pi * sim.coords[idx_node, 0] / 1.)**2
        idx_bd_condition += 1
    #     print(f"node {idx_node + 1:3d} : u = {b[idx_bd_condition-1]:.3f}")
    # print("")
    
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    # exit(0)
    return


def solve_FE(sim: Simulation_2D, atol=1e-8, rtol=1e-6):
    
    ng_all = sim.ng_all
    IB = 2 * sim.n_node  # start of dofs related to bubble function
    IS = IB + 2 * sim.n_elem if sim.degree == 3 else IB  # start of S variables
    IT = IS + ng_all  # start of T variables
    
    if sim.degree == 3:
        sim.elem_node_tags[:, -1] += sim.n_node

    # coefficients of linear minimization function
    cost = np.zeros(sim.n_var)
    
    # set constraints (1) div(u) = 0 at every gauss_pt, (2) u,v = U,V on boundary
    nb_constraints_bd = len(sim.nodes_zero_u) + len(sim.nodes_zero_v) + len(sim.nodes_with_u)
    A = np.zeros((sim.n_node, sim.n_var))

    # Old method, where div(u) = 0 at every gauss point
    # I_div = 0
    # I_bnd = ng_all
    # A = np.zeros((ng_all + nb_constraints_bd, sim.n_var))
    # b = np.zeros(A.shape[0])

    # set SOCP constraints
    I_yield = 5 * ng_all
    G = np.zeros((9 * ng_all, sim.n_var)) if sim.tau_zero > 0. else np.zeros((5 * ng_all, sim.n_var))
    h = np.zeros(G.shape[0])
    sqrt2 = np.sqrt(2.)
    
    for i in tqdm(range(sim.n_elem)):
        
        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        
        for g, wg in enumerate(sim.weights_q):
            psi = sim.q_shape_functions[g]
            dphi = sim.dv_shape_functions_at_q[g]
            dphi = np.dot(dphi, inv_jac) / det

            # pressure field is P1 --> multiply div(velocity) by the linear shape functions of the 3 VERTICES
            for j, idx_node in enumerate(idx_local_nodes[:3]):  
                A[idx_node, 2*idx_local_nodes+0] += wg * psi[j] * dphi[:, 0] * det
                A[idx_node, 2*idx_local_nodes+1] += wg * psi[j] * dphi[:, 1] * det

        for g, wg in enumerate(sim.weights):

            sf = sim.v_shape_functions[g]  # size (n_sf)
            dphi = sim.dv_shape_functions_at_v[g]  # size (n_sf, 2)
            dphi = np.dot(dphi, inv_jac) / det # size (n_sf, 2)

            i_g_idx = i * sim.ng_loc + g
            cost[IS + i_g_idx] += sim.K / 2. * wg * det
            if sim.tau_zero > 0.:
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
                if sim.tau_zero > 0.:
                    G[I_yield + 4 * i_g_idx + 1, u_idx] = -sqrt2 * dphi[j, 0]  # sqrt(2) [du_dx] = s2
                    G[I_yield + 4 * i_g_idx + 2, v_idx] = -sqrt2 * dphi[j, 1]  # sqrt(2) [dv_dy] = s3
                    G[I_yield + 4 * i_g_idx + 3, u_idx] = -1. * dphi[j, 1]  # [du_dy] +
                    G[I_yield + 4 * i_g_idx + 3, v_idx] = -1. * dphi[j, 0]  #         + [dv_dx] = s4

            # set |2D|^2 < Sig
            G[5 * i_g_idx + 0, IS + i_g_idx] = -1. / sqrt2  # (Sig + 0.5) / sqrt2 = s1
            h[5 * i_g_idx + 0] = +0.5 / sqrt2  # (Sig + 0.5) / sqrt2 = s1
            G[5 * i_g_idx + 1, IS + i_g_idx] = -1. / sqrt2  # (Sig - 0.5) / sqrt2 = s2
            h[5 * i_g_idx + 1] = -0.5 / sqrt2  # (Sig - 0.5) / sqrt2 = s2

            if sim.tau_zero > 0.:  # set |2D|^1 < Tig
                G[I_yield + 4 * i_g_idx + 0, IT + i_g_idx] = -1.  # Tig = s1
                # Could also copy paste the submatrix used for the viscous term, instead of filling it in the loop
                # G[I_yield + 4 * i_g_idx + 1: I_yield + 4 * (i_g_idx + 1), u_idx: v_idx + 1] = \
                #         G[5 * i_g_idx + 2: 5 * (i_g_idx + 1), u_idx: v_idx + 1]
        
    # vertices_idx = np.argwhere(node_is_vertex_list).flatten()
    A = A[sim.primary_nodes]  # take only the non-zero lines of the matrix A
    a = np.zeros(A.shape[0])  # A x = a
    B = np.zeros((nb_constraints_bd, sim.n_var))
    b = np.zeros(B.shape[0])  # B x = b

    set_boundary_conditions(sim, B, b)

    # print(f"U : {0:3d} -> {IB-1:3d}")
    # print(f"U' : {IB:3d} -> {IS-1:3d}")
    # print(f"S : {IS:3d} -> {IT-1:3d}")
    # print(f"T : {IT:3d} -> {IT+ng_all-1:3d}")
    # plt.spy(np.r_[A, B, G], markersize=0.5, aspect='auto')
    # plt.show()
    # exit(0)
    # print(np.linalg.matrix_rank(G))
    # print(np.linalg.matrix_rank(np.r_[A, B]))
    # print(np.linalg.matrix_rank(np.r_[A, B, G]))
    # print(np.shape(G))
    # print(np.shape(np.r_[A, B]))
    # print(np.shape(np.r_[A, B, G]))
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    # exit(0)

    cost, G, h, A, b = matrix(cost), matrix(G), matrix(h), matrix(np.r_[A, B]), matrix(np.r_[a, b])
    if sim.tau_zero > 0.:
        dims = {'l':0, 'q': [5 for i in range(ng_all)] + [4 for i in range(ng_all)], 's': []}
    else:
        dims = {'l':0, 'q': [5 for i in range(ng_all)], 's': []}

    solvers.options['abstol'] = atol
    solvers.options['reltol'] = rtol
    solvers.options['maxiters'] = 30
    start_time = perf_counter()
    res = solvers.conelp(cost, G, h, dims, A, b)
    end_time = perf_counter()

    print(f"Time to solve conic optimization = {end_time-start_time:.2f}")
    print(f"Number variables of the problem  = {sim.n_var:d}")

    u_num = np.array(res['x'])[:IB].reshape((sim.n_node, 2))
    # u_bbl = np.array(res['x'])[IB:IS].reshape((sim.n_elem * (sim.degree == 3), 2))
    # s_num = np.array(res['x'])[IS:IT].reshape((sim.n_elem, sim.ng_loc))
    # t_num = np.array(res['x'])[IT:].reshape((sim.n_elem, sim.ng_loc))

    return u_num


def solve_interface_tracking(sim: Simulation_2D, atol=1e-8, rtol=1e-6):
    return


def plot_solution_2D(u_num, sim: Simulation_2D):

    def eval_strain_norm(u_local, dphi_local):
        dudx = np.dot(u_local[:, 0], dphi_local[:, 0])
        dudy = np.dot(u_local[:, 0], dphi_local[:, 1])
        dvdx = np.dot(u_local[:, 1], dphi_local[:, 0])
        dvdy = np.dot(u_local[:, 1], dphi_local[:, 1]) 
        return dudx, dudy, dvdx, dvdy

    # strain_tensor = np.zeros((sim.n_elem,  3 * 3 + sim.elem_node_tags.shape[1] * 9))
    #     strain_tensor[i, :9] = np.r_[sim.coords[idx_local_nodes[:3], 0],
    #                                  sim.coords[idx_local_nodes[:3], 1],
    #                                  np.zeros(3)]
    #         strain_tensor[i, 9+9*j+np.array([0,1,3,4])] = local_strain_tensor.flatten() / np.sqrt(3)

    n_local_node = sim.elem_node_tags.shape[1]
    n_local_node -= 1 if sim.degree == 3 else 0
    velocity = np.c_[u_num, np.zeros_like(u_num[:, 0])]
    strain_tensor = np.zeros((sim.n_elem, n_local_node, 9))
    vorticity = np.zeros((sim.n_elem, n_local_node, 1))
    strain_norm_elem = np.zeros(sim.n_elem)

    _, dsf_at_nodes, _ = gmsh.model.mesh.getBasisFunctions(sim.elem_type, sim.local_node_coords, 'GradLagrange')
    dsf_at_nodes = np.array(dsf_at_nodes).reshape((n_local_node, n_local_node, 3))[:, :, :-1]
    # if sim.degree == 3:
    #     local_node_coords = sim.local_node_coords if sim.degree == 2 else np.r_[sim.local_node_coords, 1./3., 1./3., 0.]
    #     xi_eta_eval = np.array(sim.local_node_coords).reshape(n_local_node, 3)[:, :-1].T
    #     bubble_dsf = np.c_[DPHI_DXI(*xi_eta_eval), DPHI_DETA(*xi_eta_eval)]  # eval bubble derivatives at node pts
    #     dsf_at_nodes = np.append(dsf_at_nodes, bubble_dsf.reshape((n_local_node, 1, 2)), axis=1)
    
    for i in range(sim.n_elem):
        idx_local_nodes = sim.elem_node_tags[i, :n_local_node]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]

        for j, idx_node in enumerate(idx_local_nodes):
            
            dphi = dsf_at_nodes[j, :]  # dphi in reference element
            dphi = np.dot(dphi, inv_jac) / det  # dphi in physical element
            l11, l12, l21, l22 = eval_strain_norm(u_num[idx_local_nodes], dphi)
            vorticity[i, j] = l21 - l12
            local_strain_tensor = np.array([[l11, 0.5*(l12+l21)], [0.5*(l12+l21), l22]])
            strain_tensor[i, j, np.array([0,1,3,4])] = local_strain_tensor.flatten() / np.sqrt(3)
            # use sqrt(3) to rescale Von Mises

        for g, wg in enumerate(sim.weights):

            dphi = sim.dv_shape_functions_at_v[g][:n_local_node]  # size (n_sf, 2)
            dphi = np.dot(dphi, inv_jac) / det # size (n_sf, 2)
            l11, l12, l21, l22 = eval_strain_norm(u_num[idx_local_nodes], dphi)
            strain_norm_elem[i] += wg * np.sqrt(2 * l11 ** 2 + 2 * l22 ** 2 + (l12 + l21) ** 2)

    velocity = velocity.flatten()
    strain_tensor = strain_tensor.flatten()
    vorticity = vorticity.flatten()
    strain_norm_elem = strain_norm_elem.flatten()

    gmsh.fltk.initialize()
    tag_v = gmsh.view.add("Velocity")
    tag_strain = gmsh.view.add("Strain tensor")
    tag_vorticity = gmsh.view.add("Vorticity")
    tag_strain_norm_avg = gmsh.view.add("Strain norm averaged")
    modelName = gmsh.model.list()[0]

    gmsh.view.addHomogeneousModelData(
        tag_v, 0, modelName, "NodeData", sim.node_tags + 1, velocity, numComponents=3)
    # gmsh.view.addListData(tag_strain, "TT", sim.n_elem, strain_tensor)
    gmsh.view.addHomogeneousModelData(
        tag_strain, 0, modelName, "ElementNodeData", sim.elem_tags, strain_tensor, numComponents=9)
    gmsh.view.addHomogeneousModelData(
        tag_vorticity, 0, modelName, "ElementNodeData", sim.elem_tags, vorticity, numComponents=1)
    gmsh.view.addHomogeneousModelData(
        tag_strain_norm_avg, 0, modelName, "ElementData", sim.elem_tags, strain_norm_elem, numComponents=1)

    for tag in [tag_v, tag_strain, tag_vorticity]:
        gmsh.view.option.setNumber(tag, "AdaptVisualizationGrid", 1)
        gmsh.view.option.setNumber(tag, "TargetError", -0.0001)
        gmsh.view.option.setNumber(tag, "MaxRecursionLevel", 2)
    for tag in [tag_vorticity, tag_strain_norm_avg]:
        gmsh.view.option.setNumber(tag, "Visible", 0)
    gmsh.view.option.setNumber(tag_v, "VectorType", 6)
    gmsh.view.option.setNumber(tag_v, "NormalRaise", -2.)
    gmsh.view.option.setNumber(tag_v, "DrawLines", 0)

    gmsh.fltk.run()
    return


def plot_solution_2D_matplotlib(u_num, sim: Simulation_2D):

    fig, ax = plt.subplots(1, 1, figsize=(10., 6.), constrained_layout=True)

    n_elem, n_node_per_elem = sim.elem_node_tags.shape
    coords = sim.coords  # size (n_nodes, 2)
    
    triang = Triangulation(sim.coords[:, 0], sim.coords[:, 1], sim.elem_node_tags[:, :3])
    tricontourset = ax.tricontourf(triang, u_num[:sim.n_node, 0])
    ax.triplot(triang, 'ko-', alpha=0.5)
    _ = fig.colorbar(tricontourset)

    ax.quiver(coords[:, 0], coords[:, 1], u_num[:sim.n_node, 0], u_num[:sim.n_node, 1],
              angles='xy', scale_units='xy', scale=5)
    # ax.quiver(centers[:, 0], centers[:, 1], u_num[sim.n_node:, 0], u_num[sim.n_node:, 1],
    #  color='C2', angles='xy', scale_units='xy', scale=5)

    plt.show()
    
    return


def plot_1D_slice(u_num, sim: Simulation_2D):
    
    if sim.mesh_filename[:4] != "rect":
        return
    
    slice_node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=4)
    slice_node_tags = np.array(slice_node_tags).astype(int) - 1
    
    slice_xy = sim.coords[slice_node_tags]
    slice_y = slice_xy[:, 1]
    slice_u = u_num[slice_node_tags, 0]  # only u component of (u, v)
    H = np.amax(slice_y)

    arg_sorted_tags = np.argsort(slice_y)
    slice_u = slice_u[arg_sorted_tags]

    if sim.degree == 3:  # MINI
        slice_y = slice_y[arg_sorted_tags]
        n_intervals = len(slice_node_tags) - 1
        deg_along_edge = 1
    else:
        slice_y = slice_y[arg_sorted_tags][::2]
        n_intervals = (len(slice_node_tags) - 1) // 2
        deg_along_edge = 2

    sim_1D = Simulation_1D(
        H=H, K=sim.K, tau_zero=sim.tau_zero, f=sim.f[0], deg=deg_along_edge,
        nElem=n_intervals, random_seed=-1, fix_interface=False, save=False
    )
    sim_1D.set_y(slice_y)
    plot_solution_1D(sim=sim_1D, u_nodes=slice_u, pts_per_elem=50)
    return


if __name__ == "__main__":

    mode = 3  # 1: load previous, 2: solve problem iterative, 3: solve problem oneshot, 4: dummy solver debug

    gmsh.initialize()

    if mode == 1:
        parameters, u_nodes = load_solution("cavity_coarse")
    elif mode in [2, 3, 4]:
        # parameters = dict(K=1., tau_zero=0., f=[0., 0.], element="taylor-hood", mesh_filename="cavity_coarse")
        parameters = dict(K=1., tau_zero=0.3, f=[1., 0.], element="taylor-hood", mesh_filename="hole_coarse")
        # parameters = dict(K=1., tau_zero=0., f=[1., 0.], element="taylor-hood", mesh_filename="rect_coarse")
    else:
        raise ValueError

    sim = Simulation_2D(**parameters)

    if mode == 2:  # Solve the problem ITERATE
        u_nodes = solve_interface_tracking(sim, atol=1e-8, rtol=1e-6)
    elif mode == 3:  # Solve problem ONE SHOT
        u_nodes = solve_FE(sim, atol=1e-8, rtol=1e-6)
        sim.save_solution(u_nodes)
    elif mode == 4:  # DUMMY solution to debug
        u_nodes = np.zeros((sim.n_node, 2))
        u_nodes[:, 0] = 1. - sim.coords[:, 1]**2
        u_nodes[:, 1] = 0*sim.coords[:, 0] * (1.+sim.coords[:, 1])
    
    plot_solution_2D(u_nodes, sim)
    plot_1D_slice(u_nodes, sim)
    # plot_solution_2D_matplotlib(u_nodes, sim)

    gmsh.finalize()
