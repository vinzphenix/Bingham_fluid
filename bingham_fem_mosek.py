from bingham_structure import *
from mosek import *
from scipy.sparse import coo_matrix

# Define a stream printer to grab output from MOSEK


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def set_objective(sim: Simulation_2D, task: mosek.Task):

    cost = np.zeros(sim.n_var)

    # Handle objective coefficients of the bounds Sig, Tig
    wg_det = np.outer(sim.determinants, sim.weights).flatten()
    cost[sim.n_velocity_var: sim.n_velocity_var + sim.ng_all] = 0.5 * sim.K * wg_det
    if sim.tau_zero > 0.:
        cost[-sim.ng_all:] = sim.tau_zero * wg_det

    # Handle objective coefficients of the body force terms
    # -integral (fx, fy) * (u, v)
    force = sim.f  # size = (2,)
    phi = sim.v_shape_functions

    rows = np.zeros((sim.n_elem, sim.nsf, 2), dtype=int).flatten()

    cols = sim.elem_node_tags[:, :, np.newaxis].repeat(2, axis=2)
    cols[:, :, 0] = 2 * cols[:, :, 0] + 0  # u_idxs
    cols[:, :, 1] = 2 * cols[:, :, 1] + 1  # v_idxs
    cols = cols.flatten()

    vals = np.einsum("g,d,gj,i->ijd", sim.weights, force, phi, sim.determinants)
    # vals[np.abs(vals) < 1e-14] = 0.
    vals = vals.flatten()

    sparse_cost = coo_matrix((vals, (rows, cols)), shape=(1, sim.n_velocity_var))
    sparse_cost.sum_duplicates()
    cols, vals = sparse_cost.col, sparse_cost.data
    cost[cols] = -vals

    # Input the objective sense (minimize/maximize)
    task.putobjsense(mosek.objsense.minimize)
    task.putclist(np.arange(sim.n_var), cost)

    return


def set_boundary_conditions(sim: Simulation_2D, task: mosek.Task):

    # Impose u = 0, where needed
    idxs_zero_u = 2 * sim.nodes_zero_u + 0
    task.putvarboundlistconst(idxs_zero_u, mosek.boundkey.fx, 0., 0.)

    # Impose v = 0, where needed
    idxs_zero_v = 2 * sim.nodes_zero_v + 1
    task.putvarboundlistconst(idxs_zero_v, mosek.boundkey.fx, 0., 0.)

    # Impose u = ..., where needed
    idxs_with_u = 2 * sim.nodes_with_u + 0
    bound_key_var = np.full(idxs_with_u.size, mosek.boundkey.fx)
    bound_value = 1. + 0. * sim.coords[sim.nodes_with_u, 0]
    # bound_value = np.sin(np.pi * sim.coords[sim.nodes_with_u, 0] / 1.)**2
    # bound_value = (1. - sim.coords[sim.nodes_with_u, 1] ** 2) / 2.
    task.putvarboundlist(idxs_with_u, bound_key_var, bound_value, bound_value)

    return


def get_dphi(sim: Simulation_2D, at_v):
    """
    compute the variation of phi: [dphi_j/dx, dphi_j/dy] * det[element]
    for every shape function j, at every gauss point of every element
    """
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

    # Row of the sparse matrix element, related to the P1 pressure sf 'psi'
    # Duplicate for every P2 velocity sf phi,
    # Duplicate for (du/dx) and (dv/dy) components
    psi_indices = sim.elem_node_tags[:, :3, np.newaxis].repeat(sim.nsf, axis=2)
    psi_indices = psi_indices[:, :, :, np.newaxis].repeat(2, axis=3)

    # Column of the sparse matrix element, related to the variables Ui or Vi
    # Duplicate for every P1 pressure sf phi,
    # Duplicate for (du/dx) and (dv/dy) components
    phi_indices = sim.elem_node_tags[:, np.newaxis, :].repeat(3, axis=1)
    phi_indices = phi_indices[:, :, :, np.newaxis].repeat(2, axis=3)
    phi_indices[:, :, :, 0] = 2 * phi_indices[:, :, :, 0] + 0  # u_idx
    phi_indices[:, :, :, 1] = 2 * phi_indices[:, :, :, 1] + 1  # v_idx

    dphi = get_dphi(sim, at_v=False)  # (ne, ng, nsf, 2)
    coefficients = np.einsum("g,gk,igjn->ikjn", sim.weights_q, sim.q_shape_functions, dphi)

    # Construct sparse matrix, and sum duplicates
    rows, cols, vals = psi_indices.flatten(), phi_indices.flatten(), coefficients.flatten()
    sparse_matrix = coo_matrix((vals, (rows, cols)), shape=(sim.n_node, sim.n_var))
    sparse_matrix.sum_duplicates()
    rows, cols, vals = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data

    # filter the elements belonging to rows of singular pressures
    filter_sgl = np.isin(rows, sim.nodes_singular_p, invert=True)  # looks fast
    rows, cols, vals = rows[filter_sgl], cols[filter_sgl], vals[filter_sgl]

    # Renumber the rows of divergence constraints from node idx to vertex idx
    # i.e. remove zero rows of the sparse matrix
    _, rows = np.unique(rows, return_inverse=True)

    return rows, cols, vals, num_con


def set_strong_divergence_free(sim: Simulation_2D):
    """
    set (du/dx + dv/dy) = 0 at every gauss point of every element
    """

    num_con = sim.n_elem * sim.ng_loc

    # Row of the sparse matrix element, related to each gauss point
    # Duplicate for every P2 velocity sf phi,
    # Duplicate for (du/dx) and (dv/dy) components
    gauss_indices = np.arange(sim.n_elem * sim.ng_loc).reshape((sim.n_elem, sim.ng_loc))
    gauss_indices = gauss_indices[:, :, np.newaxis].repeat(sim.nsf, axis=2)
    gauss_indices = gauss_indices[:, :, :, np.newaxis].repeat(2, axis=3)

    # Column of the sparse matrix element, related to the variables Ui or Vi
    # Duplicate for every gauss point
    # Duplicate for (du/dx) and (dv/dy) components
    phi_indices = sim.elem_node_tags[:, np.newaxis, :].repeat(sim.ng_loc, axis=1)
    phi_indices = phi_indices[:, :, :, np.newaxis].repeat(2, axis=3)
    phi_indices[:, :, :, 0] = 2 * phi_indices[:, :, :, 0] + 0  # u_idx
    phi_indices[:, :, :, 1] = 2 * phi_indices[:, :, :, 1] + 1  # v_idx

    dphi = get_dphi(sim, at_v=True)  # (ne, ng, nsf, 2)
    dphi /= sim.determinants[:, np.newaxis, np.newaxis, np.newaxis]

    # Construct sparse matrix, and sum duplicates
    rows, cols, vals = gauss_indices.flatten(), phi_indices.flatten(), dphi.flatten()
    sparse_matrix = coo_matrix((vals, (rows, cols)), shape=(num_con, sim.n_var))
    sparse_matrix.sum_duplicates()
    rows, cols, vals = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data

    return rows, cols, vals, num_con


def impose_divergence_free(sim: Simulation_2D, task: mosek.Task, strong: bool):

    res = set_strong_divergence_free(sim) if strong else set_weak_divergence_free(sim)
    rows, cols, vals, num_con = res

    # Append 'numcon' empty constraints.
    task.appendcons(num_con)
    task.putaijlist(rows, cols, vals)
    task.putconboundsliceconst(0, num_con, mosek.boundkey.fx, 0., 0.)
    # task.putatruncatetol(1e-14)

    return


def get_affine_expressions(sim: Simulation_2D, n_local_afe: int):
    """
    (0.5, Sig, sqrt2 dudx, sqrt2 dvdy, dudy + dvdx) in rot quad cone
    (Tig, sqrt2 dudx, sqrt2 dvdy, dudy + dvdx) in quad cone
    affine expressions :
        0*x      + 0.5
        1*x[Sig] + 0.
        sqrt2 * dphi_dx * x[U]
        sqrt2 * dphi_dy * x[V]
        dphi_dy * x[U] + dphi_dx * x[V]
        1*x[Tig] + 0.
    """
    sqrt2 = np.sqrt(2.)

    dphi = get_dphi(sim, at_v=True)  # (ne, ng, nsf, 2)
    dphi /= sim.determinants[:, np.newaxis, np.newaxis, np.newaxis]

    n_global_afe = sim.n_elem * sim.ng_loc * n_local_afe
    F_rows = np.arange(n_global_afe)
    F_rows.resize((sim.n_elem, sim.ng_loc, n_local_afe))  # in-place
    repeats = [0, 1, sim.nsf, sim.nsf, 2 * sim.nsf] + [1] * (sim.tau_zero > 0.)
    nnz_local = sum(repeats)
    F_rows = np.repeat(F_rows, repeats=repeats, axis=2)

    slice_du_dx = np.s_[:, :, 1 + 0 * sim.nsf: 1 + 1 * sim.nsf]
    slice_dv_dy = np.s_[:, :, 1 + 1 * sim.nsf: 1 + 2 * sim.nsf]
    slice_du_dy = np.s_[:, :, 1 + 2 * sim.nsf: 1 + 3 * sim.nsf]
    slice_dv_dx = np.s_[:, :, 1 + 3 * sim.nsf: 1 + 4 * sim.nsf]

    F_cols = np.empty((sim.n_elem, sim.ng_loc, nnz_local), dtype=int)
    elem_node_tags = np.tile(sim.elem_node_tags[:, np.newaxis, :], reps=(1, sim.ng_loc, 1))
    F_cols[slice_du_dx] = 2 * elem_node_tags + 0
    F_cols[slice_dv_dy] = 2 * elem_node_tags + 1
    F_cols[slice_du_dy] = 2 * elem_node_tags + 0
    F_cols[slice_dv_dx] = 2 * elem_node_tags + 1

    arange_elem_gauss = np.arange(sim.ng_all).reshape((sim.n_elem, sim.ng_loc))
    F_cols[:, :, 0] = sim.n_velocity_var + arange_elem_gauss  # Sig
    if sim.tau_zero > 0.:
        F_cols[:, :, -1] = sim.n_velocity_var + sim.ng_all + arange_elem_gauss  # Tig

    F_vals = np.empty((sim.n_elem, sim.ng_loc, nnz_local))
    F_vals[slice_du_dx] = sqrt2 * dphi[:, :, :, 0]
    F_vals[slice_dv_dy] = sqrt2 * dphi[:, :, :, 1]
    F_vals[slice_du_dy] = dphi[:, :, :, 1]
    F_vals[slice_dv_dx] = dphi[:, :, :, 0]
    # F_vals[np.abs(F_vals) < 1e-14] = 0.  # remove (almost) zero entries
    F_vals[:, :, 0] = 1.
    if sim.tau_zero > 0.:
        F_vals[:, :, -1] = 1.

    g_idxs = np.arange(0, n_global_afe, n_local_afe)
    g_vals = 0.5 * np.ones(sim.ng_all)

    return (F_rows.flatten(), F_cols.flatten(), F_vals.flatten()), (g_idxs, g_vals)


def impose_conic_constraints(sim: Simulation_2D, task: mosek.Task):

    n_local_afe = 6 if sim.tau_zero > 0. else 5
    n_global_afe = sim.n_elem * sim.ng_loc * n_local_afe

    # append empty AFE rows for affine expression storage
    # store affine expression F x + g
    # locally equal to: [sqrt2 du/dx, sqrt2 dv/dy, du/dx+dv/dy, Sig, Tig]
    task.appendafes(n_global_afe)

    sparse_afe, g_afe = get_affine_expressions(sim, n_local_afe)

    task.putafefentrylist(*sparse_afe)
    task.putafeglist(*g_afe)

    # Define the conic quadratic domains
    rot_lor_cone = task.appendrquadraticconedomain(5)  # viscous bound

    # Create the ACC (affine conic constraints)
    dom_idxs = np.full(sim.ng_all, rot_lor_cone)
    elem_gauss_idx = np.arange(n_local_afe * sim.ng_all)
    rot_cones_idxs = elem_gauss_idx.reshape((sim.n_elem, sim.ng_loc, n_local_afe))
    rot_cones_idxs = rot_cones_idxs[:, :, [0, 1, 2, 3, 4]].flatten()
    task.appendaccs(dom_idxs, rot_cones_idxs, b=None)

    if sim.tau_zero > 0.:
        lorentz_cone = task.appendquadraticconedomain(4)  # yield bound
        dom_idxs = np.full(sim.ng_all, lorentz_cone)
        ltz_cones_idxs = elem_gauss_idx.reshape((sim.n_elem, sim.ng_loc, n_local_afe))
        ltz_cones_idxs = ltz_cones_idxs[:, :, [5, 2, 3, 4]].flatten()
        task.appendaccs(dom_idxs, ltz_cones_idxs, b=None)

    return


def solve_FE_mosek(sim: Simulation_2D, strong=False):
    with mosek.Task() as task:

        # Attach a log stream printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        n_velocity_var = sim.n_velocity_var

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        inf = 0
        task.appendvars(sim.n_var)
        task.putvarboundsliceconst(0, n_velocity_var, mosek.boundkey.fr, -inf, +inf)
        task.putvarboundsliceconst(n_velocity_var, sim.n_var, mosek.boundkey.lo, 0., +inf)

        # Build the problem
        start_build_time = perf_counter()
        set_objective(sim, task)
        set_boundary_conditions(sim, task)
        impose_divergence_free(sim, task, strong)
        impose_conic_constraints(sim, task)
        end_build_time = perf_counter()

        # set solver options
        task.putintparam(mosek.iparam.num_threads, 4)
        task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1.e-12)

        # Solve the minimization problem
        start_time = perf_counter()
        task.optimize()
        end_time = perf_counter()
        print(f"\nTime to BUILD conic optimization = {end_build_time-start_build_time:.2f} s")
        print(f"Time to SOLVE conic optimization = {end_time-start_time:.2f} s\n")

        # Retrieve solution (only velocity field variables)
        task.onesolutionsummary(mosek.streamtype.log, mosek.soltype.itr)

        p_num = np.array(task.gety(soltype.itr))
        u_num = np.array(task.getxxslice(soltype.itr, 0, sim.n_velocity_var))
        u_num = u_num.reshape((sim.n_node + sim.n_elem * (sim.element == 'mini'), 2))

        # if sim.tau_zero > 0.:
        #     idx_st, idx_fn = n_velocity_var + sim.ng_all, n_velocity_var + 2 * sim.ng_all
        #     yield_bounds = np.array(task.getxxslice(soltype.itr, idx_st, idx_fn))
        #     print(yield_bounds[:6 * sim.ng_loc].reshape((6, sim.ng_loc)))
        #     print(sim.elem_node_tags[:6] + 1)
        #     print(yield_bounds[-6 * sim.ng_loc:].reshape((6, sim.ng_loc)))
        #     print(sim.elem_node_tags[-6:] + 1)

    return u_num, p_num


#########################################

    # set_weak_divergence_free
    # print("check")
    # vals[np.abs(vals) < 1e-14] = 0.  # remove (almost) zero entries
    # np.savetxt("./mosek_array.txt", coo_matrix((vals, (rows, cols))).todense().T, fmt='%8.3g')


    # impose_conic_constraints    
    # # print("\n\n")
    # mat = spmatrix(F_vals.flatten(), F_rows.flatten(), F_cols.flatten())
    # for i1 in range(4*3):
    #     for i2 in range(3):
    #         coef = sqrt2 if i2 == 2 else 1.
    #         for j in range(2*13):
    #             print(f"{0.+(-coef)*mat[i1*3+i2, j]: 8.3g}", end=', ')
    #         print("")
    #     print("")
    # print("\n\n")

    # get_affine_expressions()
    # print(g_idxs)
    # print(F_rows)
    # print(F_cols)
    # print(F_vals)
    # print("\n\n")
    # print("NOWW")
    # mat = spmatrix(F_vals.flatten(), F_rows.flatten(), F_cols.flatten())
    # for i in range(n_global_afe):
    #     if i > 2 * sim.ng_loc * n_local_afe:
    #         break
    #     for j in range(sim.n_var//10):
    #         coef = sqrt2 if i % n_local_afe == 4 else 1.
    #         tmp = coef * mat[i, j]
    #         if abs(tmp) > 0.:
    #             print(f"{tmp: 6.2g}", end=', ')
    #         else:
    #             print(f"{'':6s}", end=', ')
    #     print("")
    #     if (i+1) % n_local_afe == 0:
    #         print("")
    # print("\n\n")