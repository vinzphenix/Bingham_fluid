import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri.triangulation as mpl_tri
import gmsh
import mosek


from tqdm import tqdm
from time import perf_counter
from bingham_1D_run import Simulation_1D, plot_solution_1D

ftSz1, ftSz2, ftSz3 = 15, 13, 11


def PHI(xi, eta): return 27. * xi * eta * (1. - xi - eta)
def DPHI_DXI(xi, eta): return -27. * xi * eta + 27. * eta * (1. - xi - eta)
def DPHI_DETA(xi, eta): return -27. * xi * eta + 27. * xi * (1. - xi - eta)


class Simulation_2D:
    np.set_printoptions(precision=4)

    def __init__(self, K, tau_zero, f, element, model_name):
        self.K = K  # Viscosity
        self.tau_zero = tau_zero  # yield stress
        self.f = np.array(f)  # body force (pressure gradient)

        self.model_name = model_name
        gmsh.open("../mesh/" + model_name + ".msh")

        self.element = element
        if element == "th":
            gmsh.model.mesh.setOrder(2)
            self.degree = 2
        elif element == "mini":
            gmsh.model.mesh.setOrder(1)
            self.degree = 3
        else:
            raise ValueError(f"Element '{element:s}' not implemented. Choose 'mini' or 'th'")

        self.iteration = 0

        res = self.get_elements_info()
        self.elem_type, self.elem_tags, self.elem_node_tags, self.local_node_coords = res
        self.n_local_node = len(self.local_node_coords) // 3  # neglects extra bubble
        self.n_elem, self.nsf = self.elem_node_tags.shape  # takes extra bubble into account

        res = self.get_nodes_info()
        self.node_tags, self.coords = res[0:2]
        self.nodes_zero_u, self.nodes_zero_v, self.nodes_with_u = res[2:5]
        self.nodes_singular_p, = res[5:]
        self.n_node = len(self.node_tags)

        if self.element == "mini":  # Add the index of bubble nodes
            idx_bubble_nodes = self.n_node + np.arange(self.n_elem)
            self.elem_node_tags = np.c_[self.elem_node_tags, idx_bubble_nodes]
            self.nsf += 1

        res = self.get_shape_fcts_info()
        self.weights, self.weights_q = res[:2]
        self.v_shape_functions, self.dv_shape_functions_at_v = res[2:4]
        self.q_shape_functions, self.dv_shape_functions_at_q = res[4:6]
        self.inverse_jacobians, self.determinants = res[6:8]

        self.ng_loc, self.ng_loc_q = len(self.weights), len(self.weights_q)
        self.ng_all = self.n_elem * self.ng_loc

        self.primary_nodes = self.get_primary_nodes()

        # variables :  (u, v) at every node --- bounds on |.|^2 and |.|^1 at every gauss_pt
        self.n_velocity_var = 2 * (self.n_node + self.n_elem * (element == "mini"))
        self.n_bound_var = self.ng_all + self.ng_all * (tau_zero > 0.)
        self.n_var = self.n_velocity_var + self.n_bound_var

        return

    def get_elements_info(self):
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, -1)
        elem_type = elem_types[0]
        elem_tags = elem_tags[0]
        n_elem = len(elem_tags)

        element_props = gmsh.model.mesh.getElementProperties(elem_type)
        elem_name, dim, order, n_local_node_v, local_node_coords, n_primary_nodes = element_props
        local_node_coords = np.array(local_node_coords).reshape((n_local_node_v, 2))

        # nodes indices start at 1 in gmsh -> make it start at 0
        elem_node_tags = np.array(elem_node_tags[0]).astype(int) - 1

        # store it as a matrix (n_elem, nb_node_per_elem)
        elem_node_tags = (elem_node_tags).reshape((n_elem, n_local_node_v))

        # append zero for z component
        local_node_coords = np.c_[local_node_coords, np.zeros(n_local_node_v)]
        local_node_coords = local_node_coords.flatten()

        return elem_type, elem_tags, elem_node_tags, local_node_coords,

    def get_nodes_info(self):
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.array(node_tags) - 1
        coords = np.array(coords).reshape((-1, 3))[:, :-1]

        # bd_nodes_01, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=1)  # zero u
        # bd_nodes_02, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=2)  # zero v
        # bd_nodes_03, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=3)  # with u
        bd_nodes_05, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=5)  # inf p

        bd_nodes_11, _  = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=1)  # zero u
        bd_nodes_12, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=2)  # zero v
        bd_nodes_13, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=3)  # with u

        nodes_zero_u = np.array(bd_nodes_11).astype(int) - 1
        nodes_zero_v = np.array(bd_nodes_12).astype(int) - 1
        nodes_with_u = np.array(bd_nodes_13).astype(int) - 1
        
        # Remove nodes both u = 0, u != 0
        nodes_zero_u = np.setdiff1d(nodes_zero_u, nodes_with_u)

        # Handle nodes where incompressibility is not imposed
        nodes_singular_p = np.array(bd_nodes_05).astype(int) - 1
        # nodes_singular_p = np.r_[nodes_zero_u, nodes_zero_v, nodes_with_u]
        # nodes_singular_p = np.array([], dtype=int)

        return node_tags, coords, nodes_zero_u, nodes_zero_v, nodes_with_u, nodes_singular_p

    def get_shape_fcts_info(self):
        n_local_node_v = self.n_local_node
        n_local_node_q = 3
        _3_nodes_tri = 2
        _6_nodes_tri = 9

        # location of gauss points in 3d space, and associated weights VELOCITY FIELD
        deg = (self.degree - 1) * 2  # Taylor-Hood: 2, MINI: 4
        integral_rule = "Gauss" + str(deg)
        uvw_space_v, weights_v = gmsh.model.mesh.getIntegrationPoints(_3_nodes_tri, integral_rule)
        weights_v, ng_loc_v = np.array(weights_v), len(weights_v)

        # location of gauss points in 3d space, and associated weights PRESSURE FIELD
        deg = (self.degree - 1) + 1  # Taylor-Hood: 2, MINI: 3
        integral_rule = "Gauss" + str(deg)
        uvw_space_q, weights_q = gmsh.model.mesh.getIntegrationPoints(_3_nodes_tri, integral_rule)
        weights_q, ng_loc_q = np.array(weights_q), len(weights_q)

        # sf for shape function
        _, sf, _ = gmsh.model.mesh.getBasisFunctions(self.elem_type, uvw_space_v, 'Lagrange')
        v_shape_functions = np.array(sf).reshape((ng_loc_v, n_local_node_v))

        _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(self.elem_type, uvw_space_v, 'GradLagrange')
        dv_shape_functions_at_v = np.array(dsfdu).reshape((ng_loc_v, n_local_node_v, 3))[:, :, :-1]

        _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(self.elem_type, uvw_space_q, 'GradLagrange')
        dv_shape_functions_at_q = np.array(dsfdu).reshape((ng_loc_q, n_local_node_v, 3))[:, :, :-1]

        _, sf, _ = gmsh.model.mesh.getBasisFunctions(_3_nodes_tri, uvw_space_q, 'Lagrange')
        q_shape_functions = np.array(sf).reshape((ng_loc_q, n_local_node_q))

        if self.element == "mini":  # MINI
            # Eval bubble and bubble derivatives at gauss pts of space V
            xi_eta_eval = np.array(uvw_space_v).reshape(ng_loc_v, 3)[:, :-1].T
            bubble_sf = PHI(*xi_eta_eval)
            bubble_dsf = np.c_[DPHI_DXI(*xi_eta_eval), DPHI_DETA(*xi_eta_eval)]

            v_shape_functions = np.c_[v_shape_functions, bubble_sf]
            dv_shape_functions_at_v = np.append(
                dv_shape_functions_at_v, bubble_dsf[:, None, :], axis=1
            )

            # Eval bubble and bubble derivatives at gauss pts of space Q
            xi_eta_eval = np.array(uvw_space_q).reshape(ng_loc_q, 3)[:, :-1].T
            bubble_dsf = np.c_[DPHI_DXI(*xi_eta_eval), DPHI_DETA(*xi_eta_eval)]
            dv_shape_functions_at_q = np.append(
                dv_shape_functions_at_q, bubble_dsf.reshape((ng_loc_q, 1, 2)), axis=1
            )

        # jacobian is constant over the triangle
        elem_center = np.array([1., 1., 1.]) / 3.
        jacobians, determinants, _ = gmsh.model.mesh.getJacobians(self.elem_type, elem_center)
        jacobians = np.array(jacobians).reshape((self.n_elem, 3, 3))
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
            weights_v, weights_q,
            v_shape_functions, dv_shape_functions_at_v,
            q_shape_functions, dv_shape_functions_at_q,
            inv_jac, determinants
        )

    def get_primary_nodes(self):
        node_is_vertex_list = np.zeros(self.n_node)
        for i in range(self.n_elem):
            idx_local_nodes = self.elem_node_tags[i]
            node_is_vertex_list[idx_local_nodes[:3]] = 1
        primary_nodes = np.argwhere(node_is_vertex_list).flatten()
        return primary_nodes

    def save_solution(self, u_num, p_num, model_variant):
        
        res_file_name = f"../res/{self.model_name:s}_{model_variant:s}"
        with open(res_file_name + "_params.txt", 'w') as file:
            file.write(f"{self.K:.6e}\n")
            file.write(f"{self.tau_zero:.6e}\n")
            file.write(f"{self.f[0]:.6e} {self.f[1]:.6e}\n")
            file.write(f"{self.element:s}\n")
        
        np.savetxt(res_file_name + "_velocity.txt", u_num, fmt="%.6e")
        np.savetxt(res_file_name + "_pressure.txt", p_num, fmt="%.6e")

        return
