from bingham_structure import *
from scipy.sparse import coo_matrix
import mosek


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def compute_bulk_source(sim: Simulation_2D):
    force = sim.f  # size = (2,)
    phi = sim.v_shape_functions

    rows = np.zeros((sim.n_elem, sim.nsf, 2), dtype=int).flatten()

    cols = sim.elem_node_tags[:, :, np.newaxis].repeat(2, axis=2)
    cols[:, :, 0] = 2 * cols[:, :, 0] + 0  # u_idxs
    cols[:, :, 1] = 2 * cols[:, :, 1] + 1  # v_idxs
    cols = cols.flatten()

    vals = np.einsum("g,d,gj,i->ijd", sim.weights, force, phi, sim.determinants)
    vals = vals.flatten()

    cost_bulk = coo_matrix((vals, (rows, cols)), shape=(1, sim.n_velocity_var))  # type: ignore
    cost_bulk.sum_duplicates()

    return cost_bulk.col, cost_bulk.data


def compute_boundary_source(sim: Simulation_2D):

    all_rows = np.zeros(0, dtype=int)
    all_cols = np.zeros(0, dtype=int)
    all_vals = np.zeros(0, dtype=float)

    for physical_name in ["setNormalForce", "setTangentForce"]:

        info = sim.get_edge_node_tags(physical_name)
        edge_node_tags, length, tangent, normal = info
        n_edge, n_pts = edge_node_tags.shape
        if n_edge == 0:
            continue

        # values = sigma * n = [-p n + n * tau]
        if physical_name == "setNormalForce":
            gn = np.zeros((n_edge, n_pts))
            sim.eval_gn(sim.coords[edge_node_tags], gn)
            g_vector = np.einsum("ij,id->ijd", gn, normal)
        else:  # setTangentForce
            gt = np.zeros((n_edge, n_pts))
            sim.eval_gt(sim.coords[edge_node_tags], gt)
            g_vector = np.einsum("ij,id->ijd", gt, tangent)

        coefs = np.einsum(
            'g,ijd,gj,i->ijd',
            sim.weights_edge, g_vector, sim.sf_edge, length / 2.
        )  # size (nedge, nsf, 2)
        all_vals = np.r_[all_vals, coefs.flatten()]

        rows = np.zeros(2 * edge_node_tags.size, dtype=int)
        all_rows = np.r_[all_rows, rows]

        cols = edge_node_tags[:, :, np.newaxis].repeat(2, axis=2)
        cols[:, :, 0] = 2 * cols[:, :, 0] + 0  # u_idxs
        cols[:, :, 1] = 2 * cols[:, :, 1] + 1  # v_idxs
        all_cols = np.r_[all_cols, cols.flatten()]

    cost_boundary = coo_matrix((all_vals, (all_rows, all_cols)), shape=(1, 2 * sim.n_node))
    cost_boundary.sum_duplicates()

    return cost_boundary.col, cost_boundary.data


def set_objective(sim: Simulation_2D, task: mosek.Task):

    cost = np.zeros(sim.n_var)

    # Handle objective coefficients of the bounds Sig, Tig
    wg_det = np.outer(sim.determinants, sim.weights).flatten()
    cost[sim.n_velocity_var: sim.n_velocity_var + sim.ng_all] = 0.5 * sim.K * wg_det
    if sim.tau_zero > 0.:
        cost[-sim.ng_all:] = sim.tau_zero * wg_det

    # Handle objective coefficients of the body force terms
    # -integral_{Omega} (fx, fy) * (u, v)
    cols, vals = compute_bulk_source(sim)
    cost[cols] -= vals

    # Handle objective coefficients of the Neumann boundary condition
    # -integral_{Gamma} (gx, gy) * (u, v) ds
    cols, vals = compute_boundary_source(sim)
    cost[cols] -= vals

    # Input the objective sense (minimize/maximize)
    task.putobjsense(mosek.objsense.minimize)
    task.putclist(np.arange(sim.n_var), cost)

    return


def set_boundary_conditions(sim: Simulation_2D, task: mosek.Task):

    for physical_name in ["setTangentFlow", "setNormalFlow"]:

        info = sim.get_edge_node_tags(physical_name)
        edge_node_tags, length, tangent, normal = info
        n_edge, n_pts = edge_node_tags.shape
        if n_edge == 0:
            continue

        # renumber rows from '0' to 'nb of unique nodes' in 'edge_node_tags'
        rows = edge_node_tags[:, :, None].repeat(2, axis=2)

        cols = edge_node_tags[:, :, None].repeat(2, axis=2)
        cols[:, :, 0] = 2 * cols[:, :, 0] + 0  # idxs u
        cols[:, :, 1] = 2 * cols[:, :, 1] + 1  # idxs v

        if physical_name == "setNormalFlow":
            speed = np.zeros((n_edge, n_pts))
            sim.eval_vn(sim.coords[edge_node_tags], speed)
            coefs = normal[:, None, :].repeat(n_pts, axis=1)
        else:
            speed = np.zeros((n_edge, n_pts))
            sim.eval_vt(sim.coords[edge_node_tags], speed)
            coefs = tangent[:, None, :].repeat(n_pts, axis=1)

        # Handle corners where the normal/tangent velocity is not well defined
        # Take into account only one of the two possible edges
        mask_rmv = sim.get_idx_corner_to_rm(
            sim.coords[edge_node_tags],
            normal[:, None, :].repeat(n_pts, axis=1)
        )

        rows = rows[~mask_rmv].flatten()
        cols = cols[~mask_rmv].flatten()
        coefs = coefs[~mask_rmv].flatten()
        speed = speed[~mask_rmv].flatten()
        _, rows = np.unique(rows, return_inverse=True)

        # number of constraints (unique nodes, corners possibly removed)
        n_const = np.amax(rows) + 1

        rhs = coo_matrix((speed.flatten(), (rows[::2], 0 * cols[::2])), shape=(n_const, 1))
        rhs.sum_duplicates()

        coefs = coo_matrix((coefs.flatten(), (rows, cols)), shape=(n_const, sim.n_node))
        coefs.sum_duplicates()

        row_start = task.getnumcon()
        bkc = np.full(n_const, mosek.boundkey.fx)

        mask = np.logical_or(coefs.col == 0, coefs.col == 1)

        task.appendcons(n_const)
        task.putaijlist(coefs.row + row_start, coefs.col, coefs.data)
        task.putconboundlist(rhs.row + row_start, bkc, rhs.data, rhs.data)

    return task.getnumcon()


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
    # dphi /= sim.determinants[:, np.newaxis, np.newaxis, np.newaxis]
    dphi /= 2 * sim.ng_loc  # so that dual variables correspond to pressure

    # Construct sparse matrix, and sum duplicates
    rows, cols, vals = gauss_indices.flatten(), phi_indices.flatten(), dphi.flatten()
    sparse_matrix = coo_matrix((vals, (rows, cols)), shape=(num_con, sim.n_var))
    sparse_matrix.sum_duplicates()
    rows, cols, vals = sparse_matrix.row, sparse_matrix.col, sparse_matrix.data

    return rows, cols, vals, num_con


def impose_divergence_free(sim: Simulation_2D, task: mosek.Task, strong: bool):

    res = set_strong_divergence_free(sim) if strong else set_weak_divergence_free(sim)
    rows, cols, vals, num_con = res

    row_start = task.getnumcon()

    # Append 'numcon' empty constraints.
    task.appendcons(num_con)
    task.putaijlist(rows + row_start, cols, vals)
    task.putconboundsliceconst(row_start, num_con + row_start, mosek.boundkey.fx, 0., 0.)
    # task.putatruncatetol(1e-14)

    # sim.tmp = [rows, cols, vals]
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

    idxs_zero = np.where(sim.determinants < 1.e-8)[0]
    if idxs_zero.size > 0:
        print("\nElements with negative jacobian !!!")
        print(sim.coords[sim.elem_node_tags[idxs_zero, 0]])
        print("")

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

    print("\n====================   MOSEK OPTIMIZATION LOG - start   ====================")
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
        n_bc = set_boundary_conditions(sim, task)
        impose_divergence_free(sim, task, strong)
        impose_conic_constraints(sim, task)
        end_build_time = perf_counter()

        # set solver options
        # task.putatruncatetol(1e-12)
        task.putintparam(mosek.iparam.num_threads, 4)
        # task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, 1.e-12)
        # task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, 1.e-12)
        task.putdouparam(mosek.dparam.mio_max_time, -1.0)

        # Solve the minimization problem
        start_time = perf_counter()
        task.optimize()
        end_time = perf_counter()
        print(f"\nTime to BUILD conic optimization = {end_build_time-start_build_time:.2f} s")
        print(f"Time to SOLVE conic optimization = {end_time-start_time:.2f} s\n")
        sim.run_time = end_time - start_time

        # Retrieve solution (only velocity field variables)
        task.onesolutionsummary(mosek.streamtype.log, mosek.soltype.itr)
        max_d_viol_var = task.getsolutioninfonew(mosek.soltype.itr)[10]
        max_p_viol_acc = task.getsolutioninfonew(mosek.soltype.itr)[5]
        p_cost = task.getsolutioninfonew(mosek.soltype.itr)[0]
        # print(f"{p_cost - (-1./12.):.3e}")
        # print(max_d_viol_var, max_p_viol_acc)

        i_start, i_end = n_bc, task.getnumcon()
        p_num = np.array(task.getyslice(mosek.soltype.itr, i_start, i_end))

        i_start, i_end = 0, sim.n_velocity_var
        u_num = np.array(task.getxxslice(mosek.soltype.itr, i_start, i_end))
        u_num = u_num.reshape((sim.n_node + sim.n_elem * (sim.element == 'mini'), 2))

        i_start, i_end = sim.n_var - sim.ng_all, sim.n_var
        t_num = np.array(task.getxxslice(mosek.soltype.itr, i_start, i_end))
        t_num = t_num.reshape((sim.n_elem, sim.ng_loc))

    if sim.save_variant != "":
        sim.save_solution(u_num, p_num, t_num, model_variant=sim.save_variant)

    print("\n====================   MOSEK OPTIMIZATION LOG - end  ====================\n")

    return u_num, p_num, t_num


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
