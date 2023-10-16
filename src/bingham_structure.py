import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri.triangulation as mpl_tri
import gmsh
import mosek
import os

from tqdm import tqdm
from time import perf_counter
from bingham_1D_run import Simulation_1D, plot_solution_1D
from bingham_boundary_conditions import *


ftSz1, ftSz2, ftSz3 = 15, 13, 11


# Bubble function
def PHI(xi, eta): return 27. * xi * eta * (1. - xi - eta)
def DPHI_DXI(xi, eta): return -27. * xi * eta + 27. * eta * (1. - xi - eta)
def DPHI_DETA(xi, eta): return -27. * xi * eta + 27. * xi * (1. - xi - eta)


class Simulation_2D:
    np.set_printoptions(precision=4)

    def __init__(self, parameters: dict, new_coords=None, save_variant=""):
        self.K = parameters['K']  # Viscosity
        self.tau_zero = parameters['tau_zero']  # yield stress
        self.f = np.array(parameters.get('f', [0., 0.]))  # body force

        self.model_name = parameters['model']
        gmsh.open("../mesh/" + self.model_name + ".msh")

        self.element = parameters['elem']
        if self.element == "th":
            gmsh.model.mesh.setOrder(2)
            self.degree = 2
        elif self.element == "mini":
            gmsh.model.mesh.setOrder(1)
            self.degree = 3
        elif self.element == "p1p1":
            gmsh.model.mesh.setOrder(1)
            self.degree = 2
        else:
            raise ValueError(f"Element '{self.element:s}' not implemented. Choose 'mini' or 'th'")

        if new_coords is not None:
            self.set_all_nodes(new_coords)

        self.run_time = 0.
        self.iteration = 0
        self.tol_yield = 1.e-4
        self.tag = 0
        self.save_variant = save_variant

        res = self.get_elements_info()
        self.elem_type, self.elem_tags, self.elem_node_tags, self.local_node_coords = res
        self.n_local_node = len(self.local_node_coords) // 3  # neglects extra bubble
        self.n_elem, self.nsf = self.elem_node_tags.shape  # takes extra bubble into account

        res = self.get_nodes_info()
        self.node_tags, self.coords, self.nodes_singular_p = res[0:3]
        self.nodes_corner, self.nodes_boundary, self.nodes_cut = res[3:]
        self.n_node = len(self.node_tags)

        if self.element == "mini":  # Add the index of bubble nodes
            idx_bubble_nodes = self.n_node + np.arange(self.n_elem)
            self.elem_node_tags = np.c_[self.elem_node_tags, idx_bubble_nodes]
            self.nsf += 1

        res = self.get_shape_fcts_info()
        self.weights, self.weights_q, self.uvw, self.uvw_q = res[:4]
        self.v_shape_functions, self.dv_shape_functions_at_v = res[4:6]
        self.q_shape_functions, self.dv_shape_functions_at_q = res[6:8]
        self.inverse_jacobians, self.determinants = res[8:10]
        self.min_det = np.amin(self.determinants)

        self.ng_loc, self.ng_loc_q = len(self.weights), len(self.weights_q)
        self.ng_all = self.n_elem * self.ng_loc

        self.primary_nodes = self.get_primary_nodes()
        # self.nodes_singular_p = np.intersect1d(self.nodes_singular_p, self.primary_nodes)

        self.line_tag, self.weights_edge, self.sf_edge = self.get_edge_info()
        self.ng_edge, self.nsf_edge = self.sf_edge.shape

        self.n2e_map, self.n2e_st = self.get_node_elem_map()
        self.n2n_map, self.n2n_st = self.get_node_node_map()

        # variables :  (u, v) at every node --- bounds on |.|^2 and |.|^1 at every gauss_pt
        self.n_velocity_var = 2 * (self.n_node + self.n_elem * (self.element == "mini"))
        self.n_bound_var = self.ng_all + self.ng_all * (self.tau_zero > 0.)
        self.n_var = self.n_velocity_var + self.n_bound_var

        res = self.bind_bc_functions()
        self.eval_vn, self.eval_vt, self.eval_gn, self.eval_gt, self.get_idx_corner_to_rm = res

        return

    def get_elements_info(self):
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, -1)  # type: ignore
        elem_type = elem_types[0]  # type: ignore
        elem_tags = np.array(elem_tags[0])
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

        bd_nodes, _, _ = gmsh.model.mesh.getNodes(dim=1)
        corner_nodes, _, _ = gmsh.model.mesh.getNodes(dim=0)
        cut_nodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=1, tag=5)
        nodes_singular_p, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim=0, tag=5)

        bd_nodes = np.array(bd_nodes).astype(int) - 1
        corner_nodes = np.array(corner_nodes).astype(int) - 1
        cut_nodes = np.array(cut_nodes).astype(int) - 1
        nodes_singular_p = np.array(nodes_singular_p).astype(int) - 1
        # nodes_singular_p = np.array([], dtype=int)

        return node_tags, coords, nodes_singular_p, corner_nodes, bd_nodes, cut_nodes

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
        # deg = (self.degree - 1) + 1  # Taylor-Hood: 2, MINI: 3
        deg = 2  # --> 3 gauss points, which corresponds to P1-discontinuous pressure
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

        uvw = np.array(uvw_space_v).reshape(ng_loc_v, 3)
        uvw_q = np.array(uvw_space_q).reshape(ng_loc_q, 3)

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
        elem_center = np.array([1., 1., 0.]) / 3.
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
            weights_v, weights_q, uvw, uvw_q,
            v_shape_functions, dv_shape_functions_at_v,
            q_shape_functions, dv_shape_functions_at_q,
            inv_jac, determinants
        )

    def get_primary_nodes(self):
        # node_is_vertex_list = np.zeros(self.n_node)
        # for i in range(self.n_elem):
        #     idx_local_nodes = self.elem_node_tags[i]
        #     node_is_vertex_list[idx_local_nodes[:3]] = 1
        # primary_nodes = np.argwhere(node_is_vertex_list).flatten()

        primary_nodes = np.unique(self.elem_node_tags[:, :3])
        # primary_nodes = np.setdiff1d(primary_nodes, self.nodes_boundary)
        # primary_nodes = np.setdiff1d(primary_nodes, self.nodes_corner)

        return primary_nodes

    def update_transformation(self, elements):
        for elem in elements:
            elem_center = np.array([1., 1., 0.]) / 3.
            jacobian, determinant, _ = gmsh.model.mesh.getJacobian(
                self.elem_tags[elem], elem_center)
            jacobian = np.array(jacobian).reshape((3, 3))

            self.inverse_jacobians[elem, 0, 0] = +jacobian[1, 1]
            self.inverse_jacobians[elem, 0, 1] = -jacobian[1, 0]
            self.inverse_jacobians[elem, 1, 0] = -jacobian[0, 1]
            self.inverse_jacobians[elem, 1, 1] = +jacobian[0, 0]

            self.determinants[elem] = determinant[0]
        return

    def get_node_elem_map(self):

        node_elem_pairs = np.c_[
            self.elem_node_tags[:, :3].flatten(),
            np.repeat(np.arange(self.n_elem), 3)
        ]

        pairs = node_elem_pairs[np.argsort(node_elem_pairs[:, 0])]

        # node_elem_st = np.r_[0, 1 + np.where(pairs[:-1, 0] != pairs[1:, 0])[0]]
        # not working due to higher order nodes with smaller index than primary nodes
        node_elem_st = np.cumsum(np.r_[0, np.bincount(pairs[:, 0])])
        node_elem_map = pairs[:, 1]

        return node_elem_map, node_elem_st

    def get_node_node_map(self):

        # Double direction (org-dst and dst-org) is needed only because of
        # boundary nodes, connected through one element only (two in the bulk)
        node_node_pairs = np.c_[
            [self.elem_node_tags[:, 0], self.elem_node_tags[:, 1]],
            [self.elem_node_tags[:, 1], self.elem_node_tags[:, 0]],
            [self.elem_node_tags[:, 1], self.elem_node_tags[:, 2]],
            [self.elem_node_tags[:, 2], self.elem_node_tags[:, 1]],
            [self.elem_node_tags[:, 2], self.elem_node_tags[:, 0]],
            [self.elem_node_tags[:, 0], self.elem_node_tags[:, 2]],
        ].T
        pairs = np.unique(node_node_pairs, axis=0)
        node_node_st = np.cumsum(np.r_[0, np.bincount(pairs[:, 0])])
        node_node_map = pairs[:, 1]

        return node_node_map, node_node_st

    def get_edges(self, elems):
        edges = np.c_[
            [self.elem_node_tags[elems, 0], self.elem_node_tags[elems, 1]],
            [self.elem_node_tags[elems, 1], self.elem_node_tags[elems, 2]],
            [self.elem_node_tags[elems, 2], self.elem_node_tags[elems, 0]],
        ].T
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    def is_yielded(self, strains):
        # strains is a matrix with each gauss point value
        # 0 or 1 ->   yielded
        # 2 or 3 -> unyielded
        return np.sum(strains < self.tol_yield, axis=1) < 3

    def get_support_approx(self, node):
        neighs = self.n2n_map[self.n2n_st[node]: self.n2n_st[node + 1]]
        elems = [self.n2e_map[self.n2e_st[neigh]: self.n2e_st[neigh + 1]] for neigh in neighs]
        elems = np.unique(np.concatenate(elems))
        # elems = self.n2e_map[self.n2e_st[node]: self.n2e_st[node + 1]]
        return elems

    def set_all_nodes(self, new_coords):
        n_node = new_coords.shape[0]
        new_coords = np.c_[new_coords, np.zeros(n_node)]
        for node in range(n_node):
            gmsh.model.mesh.setNode(node + 1, new_coords[node], [0., 0.])
        return

    def get_edge_info(self):
        line_tag = 8 if self.element == "th" else 1
        integral_rule = "Gauss" + str(1 * (self.element != "th") + 2 * (self.element == "th"))
        uvw, weights_edge = gmsh.model.mesh.getIntegrationPoints(line_tag, integral_rule)
        ng_edge = len(weights_edge)
        _, sf, _ = gmsh.model.mesh.getBasisFunctions(line_tag, uvw, 'Lagrange')
        sf_edge = np.array(sf).reshape((ng_edge, -1))
        n_sf_edge = sf_edge.shape[1]
        return line_tag, weights_edge, sf_edge

    def get_edge_node_tags(self, physical_name, exclude=[]):
        gmsh.model.mesh.createEdges()
        edge_node_tags = np.zeros((0, self.nsf_edge), dtype=int)

        if physical_name == "":
            tags = [-1]
            dim_tags_physical = gmsh.model.getPhysicalGroups(dim=1)
            physical_tags = [tag for (_, tag) in dim_tags_physical if tag not in exclude]
            func = gmsh.model.getEntitiesForPhysicalGroup
            tags = [func(1, physical_tag) for physical_tag in physical_tags]
            tags = np.unique(np.concatenate(tags))
        else:
            dim_tags_physical = gmsh.model.getPhysicalGroups(dim=1)
            physical_tag = -1
            for (dim, tag) in dim_tags_physical:
                name = gmsh.model.getPhysicalName(dim, tag)
                if name == physical_name:
                    physical_tag = tag
            if physical_tag == -1:
                return np.zeros((0, 0)), *[None] * 3
            else:
                tags = gmsh.model.getEntitiesForPhysicalGroup(1, physical_tag)

        for tag in tags:
            tmp = gmsh.model.mesh.getElementEdgeNodes(elementType=self.line_tag, tag=tag)
            tmp = np.array(tmp).astype(int) - 1
            edge_node_tags = np.r_[edge_node_tags, tmp.reshape((-1, self.nsf_edge))]

        coords_org = self.coords[edge_node_tags[:, 0]]
        coords_dst = self.coords[edge_node_tags[:, 1]]
        length = np.linalg.norm(coords_dst - coords_org, axis=1)
        tangent = (coords_dst - coords_org) / length[:, None]
        tsfm = np.array([[0, 1], [-1, 0]])
        normal = np.dot(tsfm[:, :], tangent[:, :].T).T

        return edge_node_tags, length, tangent, normal

    def bind_bc_functions(self):
        if self.model_name in ["rectangle", "rectanglerot"]:
            return vn_poiseuille, vt_poiseuille, gn_poiseuille, gt_poiseuille, corner_poiseuille
        elif self.model_name in ["cavity", "cavity_cheat", "cavity_test", "cavity_SV"]:
            return vn_cavity, vt_cavity, gn_cavity, gt_cavity, corner_cavity
        elif self.model_name in ["cylinder"]:
            return vn_cylinder, vt_cylinder, gn_cylinder, gt_cylinder, corner_cylinder
        elif self.model_name in ["opencavity"]:
            return vn_opencavity, vt_opencavity, gn_opencavity, gt_opencavity, corner_opencavity
        elif self.model_name in ["bfs", "necksmooth", "necksharp"]:
            return vn_bfs, vt_bfs, gn_bfs, gt_bfs, corner_bfs
        elif self.model_name in ["pipe", "finepipe"]:
            return vn_pipe, vt_pipe, gn_pipe, gt_pipe, corner_pipe
        else:
            warning_msg = f"Boundary conditions not yet implemented for model '{self.model_name}'"
            raise ValueError(warning_msg)

    def save_solution(self, u_num, p_num, t_num, model_variant):

        res_file_name = f"../res/{self.model_name:s}_{model_variant:s}"
        with open(res_file_name + "_params.txt", 'w') as file:
            file.write(f"{self.K:.6e}\n")
            file.write(f"{self.tau_zero:.6e}\n")
            file.write(f"{self.f[0]:.6e} {self.f[1]:.6e}\n")
            file.write(f"{self.element:s}\n")
            file.write(f"{self.run_time:.6e}\n")

        np.savetxt(res_file_name + "_velocity.txt", u_num, fmt="%.14e")
        np.savetxt(res_file_name + "_pressure.txt", p_num, fmt="%.6e")
        np.savetxt(res_file_name + "_strain.txt", t_num, fmt="%.6e")
        np.savetxt(res_file_name + "_coords.txt", self.coords, fmt="%.14e")

        return
