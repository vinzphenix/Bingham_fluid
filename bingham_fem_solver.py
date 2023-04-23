from bingham_structure import *


def set_coo_entries(new_data, new_rows, new_cols, data, rows, cols, idx):
    # new_data, new_rows, new_cols should have the same size
    # they are added in the corresponding global arrays 'data', 'rows', 'cols'
    # idx indicates the position where the new values are inserted
    idxs = idx + np.arange(new_data.size)
    data[idxs] = new_data
    rows[idxs] = new_rows
    cols[idxs] = new_cols
    return idx + new_data.size


def impose_weak_incompressibility(sim: Simulation_2D):
    # set linear constraints integral[psi div(u)] = 0 for all psi (1 / node)

    n_constraints = 2 * (sim.primary_nodes.size - sim.nodes_singular_p.size)
    n_node_per_elem = sim.elem_node_tags.shape[1]

    n_div_constraints_sp_entries = sim.n_elem * sim.ng_loc_q * 3 * 2 * n_node_per_elem
    n_div_constraints_sp_entries *= 2  # to impose (... = 0), one must impose (... >= 0) AND (... <= 0)

    Gl_idx = 0
    Gl_data = np.zeros(n_div_constraints_sp_entries, dtype=float)
    Gl_rows = np.zeros(n_div_constraints_sp_entries, dtype=int)
    Gl_cols = np.zeros(n_div_constraints_sp_entries, dtype=int)
    Gl_sparse = (Gl_data, Gl_rows, Gl_cols)

    # loop over each element (integrate over the hole domain)
    for i in range(sim.n_elem):

        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        u_idxs, v_idxs = 2 * idx_local_nodes, 2 * idx_local_nodes + 1

        # loop over each gauss point (discrete local integration)
        for g, wg in enumerate(sim.weights_q):

            psi = sim.q_shape_functions[g]
            dphi = sim.dv_shape_functions_at_q[g]
            dphi = np.dot(dphi, inv_jac) / det

            # pressure field is P1 --> multiply div(velocity)
            # by the linear shape functions of the 3 VERTICES only
            for k, idx_node in enumerate(idx_local_nodes[:3]):

                # do not apply the incompressibility constraint when there are singular pressures
                if idx_node not in sim.nodes_singular_p:

                    tmp_data_x = wg * psi[k] * dphi[:, 0] * det
                    tmp_data_y = wg * psi[k] * dphi[:, 1] * det

                    # set psi (dudx + dvdy) >= 0
                    Gl_idx = set_coo_entries(-tmp_data_x, 2 * idx_node + 0, u_idxs, *Gl_sparse, Gl_idx)
                    Gl_idx = set_coo_entries(-tmp_data_y, 2 * idx_node + 0, v_idxs, *Gl_sparse, Gl_idx)

                    # set psi (dudx + dvdy) <= 0
                    Gl_idx = set_coo_entries(+tmp_data_x, 2 * idx_node + 1, u_idxs, *Gl_sparse, Gl_idx)
                    Gl_idx = set_coo_entries(+tmp_data_y, 2 * idx_node + 1, v_idxs, *Gl_sparse, Gl_idx)

    # remove non initialized entries when there are singular pressures
    # renumber the rows of divergence constraints from node idx to vertex idx
    Gl_data, Gl_rows, Gl_cols = Gl_data[:Gl_idx], Gl_rows[:Gl_idx], Gl_cols[:Gl_idx]
    _, Gl_rows = np.unique(Gl_rows, return_inverse=True)
    print(f"Should be equal: {np.amax(Gl_rows)+1:d} and {n_constraints:d}")

    # print("check")
    # from scipy.sparse import coo_matrix
    # tmp_matrix = coo_matrix((Gl_data, (Gl_rows, Gl_cols)))
    # tmp_matrix.sum_duplicates()
    # tmp_matrix.data[np.abs(tmp_matrix.data) < 1e-14] = 0.
    # np.savetxt("./cvxopt_array.txt", tmp_matrix.todense()[1::2].T, fmt='%8.3g')

    return Gl_data, Gl_rows, Gl_cols, n_constraints


def impose_strong_incompressibility(sim: Simulation_2D):
    # set linear constraints div(u) = 0 for all gauss points in the hole domain

    n_constraints = 2 * sim.n_elem * sim.ng_loc
    n_node_per_elem = sim.elem_node_tags.shape[1]

    n_div_constraints_sp_entries = sim.n_elem * sim.ng_loc * 2 * n_node_per_elem
    n_div_constraints_sp_entries *= 2  # to impose (... = 0), one must impose (... >= 0) AND (... <= 0)

    Gl_idx = 0
    Gl_data = np.zeros(n_div_constraints_sp_entries, dtype=float)
    Gl_rows = np.zeros(n_div_constraints_sp_entries, dtype=int)
    Gl_cols = np.zeros(n_div_constraints_sp_entries, dtype=int)
    Gl_sparse = (Gl_data, Gl_rows, Gl_cols)

    # loop over each element (integrate over the hole domain)
    for i in range(sim.n_elem):

        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        u_idxs, v_idxs = 2 * idx_local_nodes, 2 * idx_local_nodes + 1

        # loop over each gauss point (discrete local integration)
        for g, wg in enumerate(sim.weights):

            i_g_idx = i * sim.ng_loc + g
            dphi = sim.dv_shape_functions_at_v[g]
            dphi = np.dot(dphi, inv_jac) / det
            
            # set (dudx + dvdy) >= 0
            Gl_idx = set_coo_entries(dphi[:, 0], 2 * i_g_idx + 0, u_idxs, *Gl_sparse, Gl_idx)
            Gl_idx = set_coo_entries(dphi[:, 1], 2 * i_g_idx + 0, v_idxs, *Gl_sparse, Gl_idx)

            # set (dudx + dvdy) >= 0
            Gl_idx = set_coo_entries(-dphi[:, 0], 2 * i_g_idx + 1, u_idxs, *Gl_sparse, Gl_idx)
            Gl_idx = set_coo_entries(-dphi[:, 1], 2 * i_g_idx + 1, v_idxs, *Gl_sparse, Gl_idx)
    
    return Gl_data, Gl_rows, Gl_cols, n_constraints


def build_objective_and_socp(sim: Simulation_2D, IS, IT):

    sqrt2 = np.sqrt(2.)
    n_node_per_elem = sim.elem_node_tags.shape[1]

    # initialize list of conic constraints
    n_cone_per_gauss_pt = 2 if sim.tau_zero > 0. else 1
    Gq = np.empty((sim.n_elem * sim.ng_loc, n_cone_per_gauss_pt), dtype=spmatrix)
    hq = np.empty((sim.n_elem * sim.ng_loc, n_cone_per_gauss_pt), dtype=matrix)

    # entries indicate the column and rows of an entry in the socp constraint
    # (n_elem, n_gauss, n_entry_per_elem)

    idx_sparse_s1 = 2 + 0*n_node_per_elem + np.arange(n_node_per_elem)
    idx_sparse_s2 = 2 + 1*n_node_per_elem + np.arange(n_node_per_elem)
    idx_sparse_s3 = 2 + 2*n_node_per_elem + np.arange(n_node_per_elem)
    idx_sparse_s4 = 2 + 3*n_node_per_elem + np.arange(n_node_per_elem)
    elem_node_tags_duplicated = sim.elem_node_tags[:, np.newaxis, :]
    u_idxs = 2 * elem_node_tags_duplicated + 0
    v_idxs = 2 * elem_node_tags_duplicated + 1

    Gq_rows_local = np.r_[[0, 4], [1] * n_node_per_elem, [2] * n_node_per_elem, [3] * n_node_per_elem * 2, [0]]
    Gq_rows = 5*np.arange(sim.n_elem * sim.ng_loc).reshape((sim.n_elem, sim.ng_loc))[:, :, np.newaxis] + Gq_rows_local[np.newaxis, np.newaxis, :]
    
    Gq_cols = np.empty((sim.n_elem, sim.ng_loc, 2 + 4 * n_node_per_elem + 1), dtype=int)
    Gq_cols[:, :, [0, 1]] = IS + np.arange(sim.n_elem * sim.ng_loc).reshape((sim.n_elem, sim.ng_loc))[:, :, np.newaxis]
    Gq_cols[:, :, -1] = Gq_cols[:, :, 0] + IT - IS
    Gq_cols[:, :, idx_sparse_s1] = u_idxs
    Gq_cols[:, :, idx_sparse_s2] = v_idxs
    Gq_cols[:, :, idx_sparse_s3] = u_idxs
    Gq_cols[:, :, idx_sparse_s4] = v_idxs

    dphi_loc = sim.dv_shape_functions_at_v  # size (ng, nsf, 2)
    inv_jacobians = sim.inverse_jacobians  # size (ne, 2, 2)
    determinants = sim.determinants
    dphi = np.einsum('gjk,ikl->igjl', dphi_loc, inv_jacobians)
    dphi[:] /= determinants[:, np.newaxis, np.newaxis, np.newaxis]
    
    Gq_data = np.empty((sim.n_elem, sim.ng_loc, 2 + 4 * n_node_per_elem + 1))
    Gq_data[:, :, [0, 1]] = -1. / sqrt2
    Gq_data[:, :, idx_sparse_s1] = -dphi[:, :, :, 0] * sqrt2
    Gq_data[:, :, idx_sparse_s2] = -dphi[:, :, :, 1] * sqrt2
    Gq_data[:, :, idx_sparse_s3] = -dphi[:, :, :, 1]
    Gq_data[:, :, idx_sparse_s4] = -dphi[:, :, :, 0]
    Gq_data[:, :, [-1]] = -1.

    Gq_global = spmatrix(Gq_data[:, :, :-1].flatten(), Gq_rows[:, :, :-1].flatten(), Gq_cols[:, :, :-1].flatten(),
                         size=(5 * sim.n_elem * sim.ng_loc, sim.n_var))

    # cannot find other method
    start = perf_counter()
    hq_local_viscous = matrix(np.array([0.5 / sqrt2, 0., 0., 0., -0.5 / sqrt2]))
    hq_local_yield = matrix(np.zeros(4))
    # for i in range(sim.n_elem):
    #     for g in range(sim.ng_loc):
    #     cone_idx = i * sim.ng_loc + g

    for cone_idx in range(sim.n_elem * sim.ng_loc):
        i, g = cone_idx // sim.ng_loc, cone_idx % sim.ng_loc
        Gq[cone_idx, 0] = spmatrix(Gq_data[i, g, :-1], Gq_rows_local[:-1], Gq_cols[i, g, :-1], size=(5, sim.n_var))
        hq[cone_idx, 0] = hq_local_viscous
    # if sim.tau_zero > 0.:
    #     for cone_idx in range(sim.n_elem * sim.ng_loc):
    #         i, g = cone_idx // sim.ng_loc, cone_idx % sim.ng_loc
    #         Gq[cone_idx, 1] = spmatrix(Gq_data[i, g, 2:], Gq_rows[2:], Gq_cols[i, g, 2:], size=(4, sim.n_var))
    #         hq[cone_idx, 1] = hq_local_yield

    # for cone_idx in range(sim.n_elem * sim.ng_loc):
    #     i, g = cone_idx // sim.ng_loc, cone_idx % sim.ng_loc
    #     Gq[cone_idx, 0] = Gq_global[5*cone_idx:5*(cone_idx+1), :]
    #     hq[cone_idx, 0] = hq_local_viscous


    Gq_fast = Gq.flatten().tolist()
    hq_fast = hq.flatten().tolist()
    end = perf_counter()
    print(f"Fast ? {end-start:.3f} s")

    # coefficients of linear minimization function
    cost = np.zeros(sim.n_var)

    # initialize list of conic constraints
    Gq, hq = [], []

    start = perf_counter()
    # loop over each element
    for i in range(sim.n_elem):

        idx_local_nodes = sim.elem_node_tags[i]
        det, inv_jac = sim.determinants[i], sim.inverse_jacobians[i]
        u_idxs, v_idxs = 2 * idx_local_nodes, 2 * idx_local_nodes + 1

        # initialize rows and cols of SOCP matrix constraints
        # Gq*x + s = hq, with s >= 0 in the conic sense
        Gq_rows = np.r_[[0], [1] * n_node_per_elem, [2] * n_node_per_elem, [3] * n_node_per_elem * 2, [4]]
        Gq_cols = np.r_[-1, u_idxs, v_idxs, u_idxs, v_idxs, -1]

        # loop over each gauss point
        for g, wg in enumerate(sim.weights):

            i_g_idx = i * sim.ng_loc + g

            sf = sim.v_shape_functions[g]  # size (n_sf)
            dphi = sim.dv_shape_functions_at_v[g]  # size (n_sf, 2)
            dphi = np.dot(dphi, inv_jac) / det  # size (n_sf, 2)

            Gq_data = np.r_[[0.], -sqrt2 * dphi[:, 0], -sqrt2 * dphi[:, 1], -dphi[:, 1], -dphi[:, 0], [0.]]

            # set cost coefficients
            cost[u_idxs] -= wg * sim.f[0] * sf[:] * det
            cost[v_idxs] -= wg * sim.f[1] * sf[:] * det
            cost[IS + i_g_idx] += sim.K / 2. * wg * det
            if sim.tau_zero > 0.:
                cost[IT + i_g_idx] += sim.tau_zero * wg * det

            # set |2D|^2 < Sig
            Gq_cols[[0, -1]] = IS + i_g_idx
            Gq_data[[0, -1]] = -1. / sqrt2

            Gq += [spmatrix(Gq_data, Gq_rows, Gq_cols, size=(5, sim.n_var))]
            hq += [matrix(np.array([0.5 / sqrt2, 0., 0., 0., -0.5 / sqrt2]))]

            # set |2D|^1 < Tig if bingham fluid
            if sim.tau_zero > 0.:
                Gq_cols[0] = IT + i_g_idx
                Gq_data[0] = -1.
                Gq += [spmatrix(Gq_data[:-1], Gq_rows[:-1],
                                Gq_cols[:-1], size=(4, sim.n_var))]
                hq += [matrix(np.array([0., 0., 0., 0.]))]

    end = perf_counter()
    print(f"Slow ? {end-start:.3f} s")

    return cost, Gq, hq


def set_boundary_conditions_sparse(sim: Simulation_2D, start_idx):
    idx_bd_condition = 0
    nb_constraints_bd = len(sim.nodes_zero_u) + len(sim.nodes_zero_v) + len(sim.nodes_with_u)
    nb_constraints_bd *= 2  # >= and <=

    rows = np.arange(nb_constraints_bd)
    cols = np.zeros(nb_constraints_bd, dtype=int)
    data = np.empty(nb_constraints_bd, dtype=float)
    data[0::2] = -1.
    data[1::2] = +1.
    vect = np.zeros(nb_constraints_bd, dtype=float)

    for idx_node in sim.nodes_zero_u:
        u_idx, v_idx = 2 * idx_node + 0, 2 * idx_node + 1
        # print(f"node {idx_node + 1:3d} : u = 0")

        # set U_i >= 0
        cols[idx_bd_condition] = u_idx
        vect[idx_bd_condition] = -0.
        idx_bd_condition += 1

        # set U_i <= 0
        cols[idx_bd_condition] = u_idx
        vect[idx_bd_condition] = +0.
        idx_bd_condition += 1

    for idx_node in sim.nodes_zero_v:
        u_idx, v_idx = 2 * idx_node + 0, 2 * idx_node + 1
        # print(f"node {idx_node + 1:3d} : v = 0")

        # set V_i >= 0
        cols[idx_bd_condition] = v_idx
        vect[idx_bd_condition] = -0.
        idx_bd_condition += 1

        # set V_i <= 0
        cols[idx_bd_condition] = v_idx
        vect[idx_bd_condition] = +0.
        idx_bd_condition += 1

    for idx_node in sim.nodes_with_u:
        u_idx, v_idx = 2 * idx_node + 0, 2 * idx_node + 1
        # local_speed = 1.
        # local_speed = np.sin(np.pi * sim.coords[idx_node, 0] / 1.)**2
        local_speed = (1. - sim.coords[idx_node, 1] ** 2) / 2.
        # print(f"node {idx_node + 1:3d} : u = llocal_speedoc_speed:.3f}")

        # set U_i >= 0
        cols[idx_bd_condition] = u_idx
        vect[idx_bd_condition] = -local_speed
        idx_bd_condition += 1

        # set U_i <= 0
        cols[idx_bd_condition] = u_idx
        vect[idx_bd_condition] = +local_speed
        idx_bd_condition += 1

    # set the constraint number after the last incompressibility constraint
    # (singular nodes already removed from primary nodes in 'Simulation' object initialization)
    rows += start_idx

    return data, rows, cols, vect, nb_constraints_bd


def solve_FE_sparse(sim: Simulation_2D, solver_name='mosek', strong=False):
    if solver_name not in ['mosek', 'conelp']:
        raise ValueError(
            f"solver_name should be either 'mosek' or 'conelp', not {solver_name:s}")

    IB = 2 * sim.n_node  # start of dofs related to bubble function
    IS = IB + 2 * sim.n_elem if sim.degree == 3 else IB  # start of S variables
    IT = IS + sim.ng_all  # start of T variables

    start_build_time = perf_counter()

    tmp = impose_strong_incompressibility(sim) if strong else impose_weak_incompressibility(sim)
    Gl_data, Gl_rows, Gl_cols, n_div_constraints = tmp
    bd_data, bd_rows, bd_cols, bd_hl, n_bd_constraints = set_boundary_conditions_sparse(sim, n_div_constraints)
    cost, Gq, hq = build_objective_and_socp(sim, IS, IT)

    # stack both kinds to create the full matrix of linear constraints
    size_lin_eq = (n_div_constraints + n_bd_constraints, sim.n_var)
    hl = matrix(np.r_[np.zeros(n_div_constraints), bd_hl])
    # Gl_data[np.abs(Gl_data) < 1e-14] = 0.
    # print(spmatrix(Gl_data, Gl_rows, Gl_cols)[::2, :].T)
    # for i in range(12):
    #     for j in range(2*13):
    #         print(f"{spmatrix(Gl_data, Gl_rows, Gl_cols)[2*i, j]: 8.3g}", end=', ')
    #     print("")

    Gl = spmatrix(np.r_[Gl_data, bd_data], np.r_[Gl_rows, bd_rows], np.r_[Gl_cols, bd_cols], size=size_lin_eq)

    end_build_time = perf_counter()

    # set solver options
    solvers.options['abstol'] = 1.e-10
    solvers.options['reltol'] = 1.e-8
    solvers.options['maxiters'] = 30
    solvers.options['show_progress'] = True
    solvers.options['mosek'] = {
        mosek.iparam.log: 1,
        mosek.iparam.num_threads: 4,
        mosek.dparam.intpnt_co_tol_dfeas: 1.e-12,
    }

    # solve the conic optimization problem with 'mosek', or 'conelp'
    start_time = perf_counter()
    print(sim.n_var)
    print(2*(sim.n_node+sim.n_elem)+2*sim.ng_all)
    res = solvers.socp(matrix(cost), Gl=Gl, hl=hl, Gq=Gq, hq=hq, solver=solver_name)
    end_time = perf_counter()

    print(f"\nTime to BUILD conic optimization = {end_build_time-start_build_time:.2f} s")
    print(f"Time to SOLVE conic optimization = {end_time-start_time:.2f} s\n")

    u_num = np.array(res['x'])[:IS].reshape((sim.n_node + sim.n_elem * (sim.element=='mini'), 2))
    # u_bbl = np.array(res['x'])[IB:IS].reshape((sim.n_elem * (sim.degree == 3), 2))
    # s_num = np.array(res['x'])[IS:IT].reshape((sim.n_elem, sim.ng_loc))
    # t_num = np.array(res['x'])[IT:].reshape((sim.n_elem, sim.ng_loc))

    return u_num
