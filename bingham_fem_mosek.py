from bingham_structure import *
from mosek import *
from scipy.sparse import coo_matrix

# Define a stream printer to grab output from MOSEK


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def set_boundary_conditions(sim: Simulation_2D, task: mosek.Task):

    # impose u = 0, where needed
    idxs_zero_u = 2 * sim.nodes_zero_u + 0
    bound_key_var = np.full(idxs_zero_u.size, mosek.boundkey.fx)
    bound_value = np.zeros(idxs_zero_u.size)
    task.putvarboundlist(idxs_zero_u, bound_key_var, bound_value, bound_value)

    # impose v = 0, where needed
    idxs_zero_v = 2 * sim.nodes_zero_v + 1
    bound_key_var = np.full(idxs_zero_v.size, mosek.boundkey.fx)
    bound_value = np.zeros(idxs_zero_v.size)
    task.putvarboundlist(idxs_zero_v, bound_key_var, bound_value, bound_value)

    # impose u = ..., where needed
    idxs_with_u = 2 * sim.nodes_with_u + 1
    bound_key_var = np.full(idxs_with_u.size, mosek.boundkey.fx)
    bound_value = 1. + 0. * sim.coords[sim.nodes_with_u, 0]
    # bound_value = np.sin(np.pi * sim.coords[sim.nodes_with_u, 0] / 1.)**2
    # bound_value = (1. - sim.coords[sim.nodes_with_u, 1] ** 2) / 2.
    task.putvarboundlist(idxs_with_u, bound_key_var, bound_value, bound_value)
    return


def get_dphi(sim: Simulation_2D, at_v):
    # dphi/dxi, dphi/deta in reference element
    if at_v:
        dphi_loc = sim.dv_shape_functions_at_v  # size (ng_v, nsf, 2)
    else:
        dphi_loc = sim.dv_shape_functions_at_q  # size (ng_q, nsf, 2)

    # inverse transformation matrix (xi,eta) wrt (x, y)
    inv_jacobians = sim.inverse_jacobians  # size (ne, 2, 2)

    # dphi/dx, dphi/dy in each element
    dphi = np.einsum('gjm,imn->igjn', dphi_loc, inv_jacobians)  # size (ne, ng, nsf, 2)
    # dphi[:] /= sim.determinants[:, np.newaxis, np.newaxis, np.newaxis]
    
    return dphi


def set_weak_divergence_free(sim: Simulation_2D):
    """
    set integral [ psi * (du/dx + dv/dy) ] = 0 for every sf psi_k
    """
    # def impose_weak_divergence_mosek(sim: Simulation_2D):

    num_con = sim.primary_nodes.size - sim.nodes_singular_p.size

    # number of shape functions per element
    nsf = sim.elem_node_tags.shape[1]

    # row of the sparse matrix element, related to the P1 pressure sf 'psi'
    # duplicate for every P2 velocity sf phi, 
    # duplicate for (du/dx) and (dv/dy) components
    psi_indices = sim.elem_node_tags[:, :3, np.newaxis].repeat(nsf, axis=2)
    psi_indices = psi_indices[:, :, :, np.newaxis].repeat(2, axis=3)

    # column of the sparse matrix element, related to the variables Ui or Vi
    # duplicate for every P1 pressure sf phi, 
    # duplicate for (du/dx) and (dv/dy) components
    phi_indices = sim.elem_node_tags[:, np.newaxis, :].repeat(3, axis=1)
    phi_indices = phi_indices[:, :, :, np.newaxis].repeat(2, axis=3)
    phi_indices[:, :, :, 0] = 2 * phi_indices[:, :, :, 0] + 0  # u_idx
    phi_indices[:, :, :, 1] = 2 * phi_indices[:, :, :, 1] + 1  # v_idx

    dphi = get_dphi(sim, at_v=False)  # (ne, ng, nsf, 2)
    coefficients = np.einsum("g,kg,igjn->ikjn", sim.weights_q, sim.q_shape_functions, dphi)

    # construct sparse matrix, and sum duplicates
    rows, cols, vals = psi_indices.flatten(), phi_indices.flatten(), coefficients.flatten()
    vals[np.abs(vals) < 1e-14] = 0.  # remove (almost) zero entries
    sparse_matrix = coo_matrix((vals, (rows, cols)), shape=(sim.n_node, sim.n_var))
    sparse_matrix.sum_duplicates()
    sparse_matrix.eliminate_zeros()
    rows, cols, vals = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data

    # renumber the rows of divergence constraints from node idx to vertex idx
    # i.e. remove zero rows of the sparse matrix
    _, rows = np.unique(rows, return_inverse=True)

    # print(spmatrix(vals, rows, cols))
    
    return rows, cols, vals, num_con


def set_strong_divergence_free(sim: Simulation_2D):
    """
    set (du/dx + dv/dy) = 0 at every gauss point of every element
    """
    
    num_con = sim.n_elem * sim.ng_loc

    # number of shape functions per element
    nsf = sim.elem_node_tags.shape[1]

    # row of the sparse matrix element, related to each gauss point
    # duplicate for every P2 velocity sf phi,
    # duplicate for (du/dx) and (dv/dy) components 
    gauss_indices = np.arange(sim.n_elem * sim.ng_loc).reshape((sim.n_elem, sim.ng_loc))
    gauss_indices = gauss_indices[:, :, np.newaxis].repeat(nsf, axis=2)
    gauss_indices = gauss_indices[:, :, :, np.newaxis].repeat(2, axis=3)

    # column of the sparse matrix element, related to the variables Ui or Vi
    # duplicate for every gauss point
    # duplicate for (du/dx) and (dv/dy) components
    phi_indices = sim.elem_node_tags[:, np.newaxis, :].repeat(sim.ng_loc, axis=1)
    phi_indices = phi_indices[:, :, np.newaxis, :].repeat(2, axis=2)
    phi_indices[:, :, 0, :] = 2 * phi_indices[:, :, 0, :] + 0  # u_idx
    phi_indices[:, :, 1, :] = 2 * phi_indices[:, :, 1, :] + 1  # v_idx

    dphi = get_dphi(sim, at_v=True)  # (ne, ng, nsf, 2)
    dphi /= sim.determinants[:, np.newaxis, np.newaxis, np.newaxis]
    dphi = np.swapaxes(dphi, 2, 3)

    # construct sparse matrix, and sum duplicates
    rows, cols, vals = gauss_indices.flatten(), phi_indices.flatten(), dphi.flatten()
    vals[np.abs(vals) < 1e-14] = 0.  # remove (almost) zero entries
    sparse_matrix = coo_matrix((vals, (rows, cols)), shape=(num_con, sim.n_var))
    sparse_matrix.sum_duplicates()
    sparse_matrix.eliminate_zeros()
    rows, cols, vals = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data

    # print("\n\n")
    # for i in range(12):
    #     for j in range(2*13):
    #         print(f"{sparse_matrix.todense()[i, j]: 8.3g}", end=', ')
    #     print("")
    # print("\n\n")
    # print(spmatrix(vals, rows, cols))

    return rows, cols, vals, num_con


def impose_divergence_free(sim: Simulation_2D, task: mosek.Task, strong: bool):
        
    res = set_strong_divergence_free(sim) if strong else set_weak_divergence_free(sim)
    rows, cols, vals, num_con = res

    # Append 'numcon' empty constraints.
    # The constraints will initially have no bounds.
    task.appendcons(num_con)
    
    bound_key_con = np.full(num_con.size, mosek.boundkey.fx)
    bound_val_con = np.zeros(num_con.size)

    # set the matrix-vector equality constraint
    task.putaijlist(rows, cols, vals)
    task.putconboundslice(0, num_con, bound_key_con, bound_val_con, bound_val_con)

    return


def impose_conic_constraints(sim: Simulation_2D, task: mosek.task):
    return


def solve_FE_mosek(sim: Simulation_2D, strong=False):
    with mosek.Task() as task:

        # Attach a log stream printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(sim.n_var)

        impose_divergence_free(sim, task, strong)


    return
