import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri.triangulation as mpl_tri
import gmsh
import mosek

from tqdm import tqdm
from time import perf_counter
from cvxopt import solvers, matrix, spmatrix
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
        self.f = f  # body force (pressure gradient)

        self.model_name = model_name
        gmsh.open("./mesh/" + model_name + ".msh")

        self.element = element
        if element == "taylor-hood":
            gmsh.model.mesh.setOrder(2)
            self.degree = 2
        elif element == "mini":
            gmsh.model.mesh.setOrder(1)
            self.degree = 3
        else:
            raise ValueError(f"Element '{element:s}' not implemented. Choose 'mini' or 'taylor-hood'")

        self.iteration = 0

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
        self.primary_nodes, self.nodes_singular_p = res[5:]
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

        nodes_zero_u = np.setdiff1d(bd_nodes_11.astype(int), bd_nodes_13.astype(int)) - 1
        nodes_zero_v = bd_nodes_12.astype(int) - 1
        nodes_with_u = bd_nodes_13.astype(int) - 1

        nodes_singular_p = bd_nodes_05.astype(int) - 1
        # nodes_singular_p = np.r_[nodes_zero_u, nodes_zero_v, nodes_with_u]
        # nodes_singular_p = np.array([], dtype=int)

        node_is_vertex_list = np.zeros(len(node_tags))
        for i in range(self.n_elem):
            idx_local_nodes = self.elem_node_tags[i]
            node_is_vertex_list[idx_local_nodes[:3]] = 1

        primary_nodes = np.argwhere(node_is_vertex_list).flatten()

        return node_tags, coords, nodes_zero_u, nodes_zero_v, nodes_with_u, primary_nodes, nodes_singular_p

    def save_solution(self, u_num):
        with open(f"./res/{self.model_name:s}.txt", 'w') as file:
            file.write(f"{self.K:.6e}\n")
            file.write(f"{self.tau_zero:.6e}\n")
            file.write(f"{self.f[0]:.6e} {self.f[1]:.6e}\n")
            file.write(f"{self.element:s}\n")
            file.write(f"{self.model_name:s}\n")
            np.savetxt(file, u_num, fmt="%.6e")
        return
