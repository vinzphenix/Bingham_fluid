from bingham_structure import *


#################################################################################
############################  -  Poiseuille Flow  -  ############################

def vn_poiseuille(coords, res):
    n_edge, n_pts, _ = coords.shape
    beta = np.pi / 6.
    rot_matrix = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    rot_coords = np.einsum("mn,ijn->ijm", rot_matrix, coords)
    mask_inflow = np.abs(rot_coords[:, :, 0] - np.amin(rot_coords[:, :, 0])) <= 1.e-5
    res[mask_inflow] = -(0.125 - 0.5 * rot_coords[mask_inflow, 1] ** 2)
    return


def vt_poiseuille(coords, res):
    # vt always zero
    return


def gn_poiseuille(coords, res):
    beta = np.pi / 6.
    rot_matrix = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    rot_coords = np.einsum("mn,ijn->ijm", rot_matrix, coords)
    x_min, x_max = np.amin(rot_coords[:, :, 0]), np.amax(rot_coords[:, :, 0])
    mask_inflow = np.abs(rot_coords[:, :, 0] - x_min) <= 1.e-5
    mask_outflow = np.abs(rot_coords[:, :, 0] - x_max) <= 1.e-5
    p_inflow, p_outflow = x_max - x_min, 0.
    res[mask_inflow] = -p_inflow + 0. 
    res[mask_outflow] = -p_outflow + 0.
    return


def gt_poiseuille(coords, res):
    # gt always zero
    return


#################################################################################
##############################  -  Cavity Flow  -  ##############################

def vn_cavity(coords, res):
    return


def vt_cavity(coords, res):
    return


def gn_cavity(coords, res):
    return


def gt_cavity(coords, res):
    return


#################################################################################
#############################  -  Cylinder Flow  -  #############################

def vn_cylinder(coords, res):
    return


def vt_cylinder(coords, res):
    return


def gn_cylinder(coords, res):
    return


def gt_cylinder(coords, res):
    return


    # # bound_value = 1. + 0. * sim.coords[sim.nodes_with_u, 0]
    # # bound_value = np.sin(np.pi * sim.coords[sim.nodes_with_u, 0] / 1.)**2
    # bound_value = (0.25 - sim.coords[sim.nodes_with_u, 1] ** 2) / 2.
