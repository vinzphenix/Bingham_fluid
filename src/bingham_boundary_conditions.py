from bingham_structure import *

tol = 1.e-5
beta = 0. * np.pi / 6.


#################################################################################
############################  -  Poiseuille Flow  -  ############################

def vn_poiseuille(coords, res):
    # n_edge, n_pts, _ = coords.shape
    # rot_matrix = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    # rot_coords = np.einsum("mn,ijn->ijm", rot_matrix, coords)
    # mask_inflow = np.abs(rot_coords[:, :, 0] - np.amin(rot_coords[:, :, 0])) <= tol
    # res[mask_inflow] = -(0.125 - 0.5 * rot_coords[mask_inflow, 1] ** 2)
    return


def vt_poiseuille(coords, res):
    # vt always zero
    return


def gn_poiseuille(coords, res):
    rot_matrix = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    rot_coords = np.einsum("mn,ijn->ijm", rot_matrix, coords)
    x_min, x_max = np.amin(rot_coords[:, :, 0]), np.amax(rot_coords[:, :, 0])
    mask_inflow = np.abs(rot_coords[:, :, 0] - x_min) <= tol
    mask_outflow = np.abs(rot_coords[:, :, 0] - x_max) <= tol
    p_inflow, p_outflow = (x_max - x_min), 0.
    res[mask_inflow] = -p_inflow + 0.
    res[mask_outflow] = -p_outflow + 0.
    return


def gt_poiseuille(coords, res):
    # gt always zero
    return


def corner_poiseuille(coords, normals):
    rot_matrix = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])
    rot_coords = np.einsum("mn,ijn->ijm", rot_matrix, coords)
    y_min, y_max = np.amin(rot_coords[:, :, 1]), np.amax(rot_coords[:, :, 1])

    ref_normal = np.dot(rot_matrix.T, np.array([1., 0.]))
    mask_normal = np.abs(np.einsum("ijd,d->ij", normals, ref_normal)) > 1. - tol
    mask_top = rot_coords[:, :, 1] > y_max - tol
    mask_bot = rot_coords[:, :, 1] < y_min + tol

    return np.logical_and(np.logical_or(mask_top, mask_bot), mask_normal)


#################################################################################
#############################  -  Cylinder Flow  -  #############################

def vn_cylinder(coords, res):
    # vn = 0. on lateral walls, also zero on cylinder
    # vn = 1. at inflow
    mask_left = np.abs(coords[:, :, 0] - 0.) <= tol
    res[mask_left] = -1.
    return


def vt_cylinder(coords, res):
    # vt = 0. at inflow and outflow, also zero on cylinder
    return


def gn_cylinder(coords, res):
    x_min, x_max = np.amin(coords[:, :, 0]), np.amax(coords[:, :, 0])
    mask_inflow = np.abs(coords[:, :, 0] - x_min) <= tol
    mask_outflow = np.abs(coords[:, :, 0] - x_max) <= tol
    # dp = 6. * np.pi / 10.  # should produce a inflow velocity of 1 (Stokes law)
    # delta_x = x_max - x_min
    dp = 1.
    p_inflow, p_outflow = dp / 2., -dp / 2.
    res[mask_inflow] = -p_inflow + 0.
    res[mask_outflow] = -p_outflow + 0.
    return


def gt_cylinder(coords, res):
    # gt = 0. on lateral walls
    return


def corner_cylinder(coords, normals):
    x_min, x_max = np.amin(coords[:, :, 0]), np.amax(coords[:, :, 0])
    ref_normal = np.array([0., 1.])
    mask_normal = np.abs(np.einsum("ijd,d->ij", normals, ref_normal)) > 1. - tol
    mask_lf = coords[:, :, 0] > x_max - tol
    mask_rg = coords[:, :, 0] < x_min + tol
    return np.logical_and(np.logical_or(mask_lf, mask_rg), mask_normal)


#################################################################################
##############################  -  Cavity Flow  -  ##############################

def vn_cavity(coords, res):
    # vn always 0
    return


def vt_cavity(coords, res):
    mask_top = np.abs(coords[:, :, 1] - 0.) < tol
    mask_lf = np.abs(coords[:, :, 0] - 0.) < tol
    mask_rg = np.abs(coords[:, :, 0] - 1.) < tol
    mask = mask_top & ~mask_lf & ~mask_rg
    res[mask] = -1.
    # x_coord_edge = coords[mask, 0]
    # res[mask] = -np.sin(np.pi * x_coord_edge)**2
    return


def gn_cavity(coords, res):
    # zero because never called
    return


def gt_cavity(coords, res):
    # zero because never called
    return


def corner_cavity(coords, normals):
    ref_normal = np.array([0., 1.])
    mask_normal = np.abs(np.einsum("ijd,d->ij", normals, ref_normal)) > 1. - tol
    mask_lf = np.abs(coords[:, :, 0] - 0.) < tol
    mask_rg = np.abs(coords[:, :, 0] - 1.) < tol
    return np.logical_and(np.logical_or(mask_lf, mask_rg), mask_normal)


#################################################################################
############################  -  OpenCavity Flow  -  ############################

def vn_opencavity(coords, res):
    # vn always 0 (top slip wall, and no-slip walls)
    return


def vt_opencavity(coords, res):
    # vt = 1 on top wall
    # vt = 0. on inflow and outflow
    mask_top = np.abs(coords[:, :, 1] - 0.) < tol
    res[mask_top] = -1.
    return


def gn_opencavity(coords, res):
    # always zero (no pressure gradient at inflow/outflow ?)
    return


def gt_opencavity(coords, res):
    # zero because never called
    return


def corner_opencavity(coords, normals):
    ref_normal = np.array([1., 0.])
    mask_normal = np.abs(np.einsum("ijd,d->ij", normals, ref_normal)) > 1. - tol
    mask_top = np.abs(coords[:, :, 1] - 0.) < tol
    mask_bot = np.abs(coords[:, :, 1] - (-1.)) < tol
    return np.logical_and(np.logical_or(mask_top, mask_bot), mask_normal)


#################################################################################
#######################  -  Backward facing step Flow  -  #######################

def vn_bfs(coords, res):
    # vn always zero
    return


def vt_bfs(coords, res):
    # vt always zero
    return


def gn_bfs(coords, res):
    x_min, x_max = np.amin(coords[:, :, 0]), np.amax(coords[:, :, 0])
    mask_inflow = np.abs(coords[:, :, 0] - x_min) <= tol
    mask_outflow = np.abs(coords[:, :, 0] - x_max) <= tol
    p_inflow, p_outflow = (x_max - x_min), 0.
    res[mask_inflow] = -p_inflow + 0.
    res[mask_outflow] = -p_outflow + 0.
    return


def gt_bfs(coords, res):
    # gt never called
    return


def corner_bfs(coords, normals):
    x_min, x_max = np.amin(coords[:, :, 0]), np.amax(coords[:, :, 0])
    y_min, y_max = np.amin(coords[:, :, 1]), np.amax(coords[:, :, 1])

    ref_normal = np.array([1., 0.])
    mask_normal = np.abs(np.einsum("ijd,d->ij", normals, ref_normal)) > 1. - tol
    mask_top = coords[:, :, 1] > y_max - tol
    mask_lf = coords[:, :, 0] < x_min + tol
    mask_rg = coords[:, :, 0] > x_max - tol
    mask_bot_lf = np.logical_and(mask_lf, coords[:, :, 1] < np.amin(coords[mask_lf, 1]) + tol)
    mask_bot_rg = np.logical_and(mask_rg, coords[:, :, 1] < np.amin(coords[mask_rg, 1]) + tol)
    mask_bot = np.logical_or(mask_bot_lf, mask_bot_rg)

    return np.logical_and(np.logical_or(mask_top, mask_bot), mask_normal)


#################################################################################
###############################  -  Pipe Flow  -  ###############################
# l1, l2, width, radius, theta = 2., 1., 1., 0.5, 90.  # copy-paste depending on pipe
l1, l2, width, radius, theta = 0.1, 0.1, 1., 1.5, 180.  # copy-paste depending on pipe

rot_matrix = np.array([
    [+np.cos(np.radians(theta)), +np.sin(np.radians(theta))],
    [-np.sin(np.radians(theta)), +np.cos(np.radians(theta))]
])
corners = np.array([
    [l1+l2, l1+l2],
    [0., width]
])
center = np.array([l1, -radius])
corners = center[:, None] + np.dot(rot_matrix, (corners - center[:, None]))
corners = np.c_[[0., 0.], [0., width], corners].T
ref_normal_in = np.array([1., 0.])
ref_normal_out = np.dot(rot_matrix, ref_normal_in)

def vn_pipe(coords, res):
    # vn always zero
    return


def vt_pipe(coords, res):
    # vt always zero
    return


def gn_pipe(coords, res):
    mask_inflow = np.logical_and(coords[:, :, 0] < 0. + tol, coords[:, :, 1] > 0.)

    inv_rot_coords = np.einsum("mn,ijn->ijm", rot_matrix.T, coords - center[None, None, :])
    inv_rot_coords = center[None, None, :] + inv_rot_coords
    mask_outflow = np.logical_and(
        inv_rot_coords[:, :, 0] > l1 + l2 - tol,
        inv_rot_coords[:, :, 1] > 0. - tol,
    )

    dpdx = 1.
    p_inflow, p_outflow = dpdx * (l1 + l2 + 0.5 * (radius + width) * np.radians(theta)), 0.
    res[mask_inflow] = -p_inflow + 0.
    res[mask_outflow] = -p_outflow + 0.
    return


def gt_pipe(coords, res):
    # gt always zero
    return


def corner_pipe(coords, normals):

    mask_c1 = np.linalg.norm(coords - corners[None, None, 0, :], axis=2) < tol
    mask_c2 = np.linalg.norm(coords - corners[None, None, 1, :], axis=2) < tol
    mask_c3 = np.linalg.norm(coords - corners[None, None, 2, :], axis=2) < tol
    mask_c4 = np.linalg.norm(coords - corners[None, None, 3, :], axis=2) < tol
    
    mask_normal_in = np.abs(np.einsum("ijd,d->ij", normals, ref_normal_in)) > 1. - tol
    mask_in = np.logical_and(np.logical_or(mask_c1, mask_c2), mask_normal_in)
    
    mask_normal_out = np.abs(np.einsum("ijd,d->ij", normals, ref_normal_out)) > 1. - tol
    mask_out = np.logical_and(np.logical_or(mask_c3, mask_c4), mask_normal_out)

    return np.logical_or(mask_in, mask_out)
