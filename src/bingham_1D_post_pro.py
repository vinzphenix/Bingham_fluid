from bingham_1D_structure import *
plt.rcParams['font.family'] = 'serif'
plt.rcParams["text.usetex"] = True

lw, alp = 5., 0.5  # setting to display analytical solution
al = 0.3  # setting for the error plot


def make_step(array_x, array_y, step):
    pos = np.arange(step, len(array_x), step)
    return np.insert(array_x, pos, np.nan), np.insert(array_y, pos, np.nan)


def eval_analytical_sol(sim: Simulation_1D, y):
    e0, eta = 2. * sim.y_zero / sim.H, 2. * y / sim.H

    m_bot = eta <= -e0
    m_mid = (-e0 < eta) & (eta < e0)
    m_top = e0 <= eta

    u_ana, du_ana = np.zeros(y.size), np.zeros(y.size)

    u_ana[m_top] = -sim.Bn / 2. * (1. - eta[m_top]) + (1. - np.square(eta[m_top]))
    u_ana[m_bot] = -sim.Bn / 2. * (1. + eta[m_bot]) + (1. - np.square(eta[m_bot]))
    u_ana[m_mid] = (1. - sim.Bn / 4.) ** 2

    du_ana[m_top] = (+sim.tau_zero - y[m_top] * sim.f) / sim.K
    du_ana[m_bot] = (-sim.tau_zero - y[m_bot] * sim.f) / sim.K
    du_ana[m_mid] = 0.

    tau_ana = sim.f * np.abs(y)
    tau_ref = sim.tau_zero if sim.tau_zero > 0. else sim.H * sim.f

    u_ana *= sim.V if sim.dimensions else 1.
    du_ana *= 1. if sim.dimensions else sim.H / sim.V
    tau_ana /= 1. if sim.dimensions else tau_ref

    return u_ana, du_ana, tau_ana


def eval_stress(sim: Simulation_1D, y, du):
    tau = np.abs(sim.K * du.reshape(sim.n_elem, -1)) + sim.tau_zero
    rigid_mask, = np.where(np.all(np.abs(np.reshape(du, (sim.n_elem, -1))) < 1e-6, axis=1))
    tau[rigid_mask] = np.nan
    return tau.flatten()


def eval_velocity_strain(sim: Simulation_1D, y, u_num):

    dy, ym = np.diff(y), 0.5 * (y[1:] + y[:-1])
    pts_per_elem = sim.plot_density + 1
    xi_dense = np.linspace(-1., 1., pts_per_elem)
    y_dense = np.zeros((sim.n_elem, pts_per_elem))
    u_dense = np.zeros((sim.n_elem, pts_per_elem))
    du_dense = np.zeros((sim.n_elem, pts_per_elem))
    du_discrete = np.zeros((sim.n_elem, 2))
    du_gauss = np.zeros((sim.n_elem, sim.nG))

    for i in range(sim.n_elem):
        y_dense[i] = np.linspace(y[i], y[i + 1], pts_per_elem)

        for idx, phi, dphi in zip(sim.idx_nodes_elem[i], sim.PHI, sim.DPHI):
            u_dense[i, :] += u_num[idx] * phi(xi_dense)
            du_dense[i, :] += 2. / dy[i] * u_num[idx] * dphi(xi_dense)
            du_discrete[i, :] += 2. / dy[i] * u_num[idx] * dphi(np.array([-1., 1.]))
            du_gauss[i, :] += 2. / dy[i] * u_num[idx] * dphi(sim.xG)

    y_discrete = y_dense[:, [0, -1]].flatten()
    y_gauss = np.r_[[ym[i] + sim.xG * dy[i] / 2. for i in range(sim.n_elem)]]

    ys = [y_dense.flatten(), y_discrete.flatten(), y_gauss.flatten()]
    strains = [du_dense.flatten(), du_discrete.flatten(), du_gauss.flatten()]
    stresses = [eval_stress(sim, y, du) for (du, y) in zip(strains, ys)]

    if not sim.dimensions:
        u_dense /= sim.V
        tau_ref = sim.tau_zero if sim.tau_zero > 0. else sim.H * sim.f
        for i in range(3):
            strains[i][:] /= sim.V / (sim.H)
            stresses[i][:] /= tau_ref

    return [u_dense.flatten(), *ys, *strains, *stresses]


def plot_reconstruction(sim: Simulation_1D, y_bfr, u_bfr, y_aft, u_aft, idxs_switch, coefs):

    fs = (8., 3.5)
    fig, axs = plt.subplots(1, 2, figsize=fs, constrained_layout=True, sharex='all', sharey='all')

    y_ana = np.array([-sim.H / 2., -sim.y_zero, 0., sim.y_zero, sim.H / 2.])
    _, du_ana_fit, _ = eval_analytical_sol(sim, y_ana)  # already scaled

    sy, sdu = (1., 1.) if sim.dimensions else (2. / sim.H, sim.H / sim.V)

    # 2: y_discrete, 3: y_gauss, 5: du_discrete, 6: du_gauss
    res1 = eval_velocity_strain(sim, y_bfr, u_bfr)
    axs[0].plot(du_ana_fit, y_ana * sy, label="Analytical", color="C0", alpha=alp, lw=lw)
    axs[0].plot(*make_step(res1[5], res1[2] * sy, 2), '-o', color='C1', ms=5, label='Numerical')
    axs[0].plot(res1[6], sy * res1[3], 'x', color='C1', ms=8)

    for (i, dir), coef in zip(idxs_switch, coefs):
        y_guess = -coef[0] / coef[1]

        y_min = y_guess if dir == 1 else y_bfr[max(i - 1, 0)]
        y_max = y_bfr[min(i + 2, sim.n_elem)] if dir == 1 else y_guess
        delta = (y_max - y_min) / 6.
        y_array = np.linspace(y_min - delta, y_max + delta, 2)  # straight line
        dudy_approx = np.dot(np.c_[np.ones(2), y_array], coef)
        dudy_approx *= -1. if dir == 1 else 1.

        idxs_gauss = np.arange(sim.nG * i, sim.nG * (i + 1))
        if 0 <= i + dir < sim.n_elem:
            idxs_gauss = np.r_[idxs_gauss, np.arange(sim.nG * (i + dir), sim.nG * (i + dir + 1))]

        labels = ("Reconstructed", "Root") if dir == 1 else ("", "")
        axs[0].plot(dudy_approx * sdu, y_array * sy, '--k', ms=8, label=labels[0])
        axs[0].plot(res1[6][idxs_gauss], sy * res1[3][idxs_gauss], 'xk')
        axs[0].plot([0.], [y_guess * sy], 'D', color='C3', ms=7, label=labels[1])

        for ax in axs:
            ax.axhline(y_guess * sy, color='C2', alpha=alp, ls=':', lw=1.5)

    res2 = eval_velocity_strain(sim, y_aft, u_aft)
    axs[1].plot(du_ana_fit, y_ana * sy, label="Analytical", color="C0", alpha=alp, lw=lw)
    axs[1].plot(*make_step(res2[5], res2[2] * sy, 2), '-o', color='C1', ms=5, label='Numerical')
    axs[1].plot(res2[6], sy * res2[3], 'x', color='C1', ms=8)

    mask = res1[2] != res2[2]
    axs[0].plot(res1[5][mask], sy * res1[2][mask], 'o', color='C2', ms=5, label="Interface")
    axs[1].plot(res2[5][mask], sy * res2[2][mask], 'o', color='C2', ms=5, label="Interface")

    extra_label = r"" if sim.dimensions else r" / (h/2)"
    axs[0].set_ylabel(r"$y {:s}$".format(extra_label), fontsize=ftSz2)
    for ax, title_name in zip(axs, ['Before update', 'After update']):
        if sim.iteration == 2:
            extra_label = r"" if sim.dimensions else r"h / U_{\infty}"
            ax.set_xlabel(r"${:s} \partial_y u $".format(extra_label), fontsize=ftSz2)
        if sim.iteration == 1:
            ax.set_title(title_name, fontsize=ftSz1)
        ax.legend(fontsize=ftSz3, loc="lower left")
        ax.grid(ls=':')

    if sim.save:
        path = "../figures/"
        filename = f"res_P{sim.degree:d}_iteration_{sim.iteration:02d}"
        fig.savefig(f"{path:s}{filename:s}.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()
    return


def plot_solution_1D(
    sim: Simulation_1D, u_nodes, mini_display=False,
    extra_name="", window="Overview Poiseuille"
):

    u_num = u_nodes.copy()

    res = eval_velocity_strain(sim, sim.y, u_num)
    u_dense = res[0]
    y_dense, y_discrete, y_gauss = res[1:4]
    du_dense, du_discrete, du_gauss = res[4:7]
    tau_dense, tau_discrete, tau_gauss = res[7:10]

    y_all = np.insert(sim.y, np.arange(1, sim.n_elem + 1), sim.ym)
    y_ana = np.array([-sim.H / 2., -sim.y_zero, 0., sim.y_zero, sim.H / 2.])
    u_ana_dense, du_ana_dense, tau_ana_dense = eval_analytical_sol(sim, y_dense)
    u_ana_nodes, _, _ = eval_analytical_sol(sim, y_all)
    _, du_ana_fit, tau_ana_fit = eval_analytical_sol(sim, y_ana)
    _, du_ana_gauss, tau_ana_gauss = eval_analytical_sol(sim, y_gauss)

    if not sim.dimensions:
        u_num /= sim.V
        for this_y in [y_all, y_ana, y_gauss, y_dense, y_discrete]:
            this_y[:] /= sim.H / 2.

    # FIGURE
    if mini_display:
        figsize, dims = (8., 3.75), (1, 2)
    elif sim.save:
        figsize, dims = (9.5, 5.75), (2, 3)
    else:
        figsize, dims = (12., 8.), (2, 3)

    quadrature = False
    fig, axs = plt.subplots(*dims, figsize=figsize, sharey="all", num=window)
    axs = axs.reshape(dims)

    ax = axs[0, 0]
    extra_label = r"" if sim.dimensions else r"/ U_{\infty}"
    ax.set_xlabel(r"$u(y) {:s}$".format(extra_label), fontsize=ftSz2)
    ax.set_title("Velocity profile", fontsize=ftSz1)
    ax.plot(u_ana_dense, y_dense, ls='-', color='C0', alpha=alp, lw=lw, label="Analytical")
    ax.plot(u_dense, y_dense, '-', color='C1')
    ax.plot([], [], color='C1', ls='-', marker='o', label="Numerical")  # FEM solution
    ax.plot(u_num[::sim.degree], y_all[::2], marker="o", ls="", color='C1')
    if sim.degree == 2:
        ax.plot(u_num[1::2], y_all[1::2], marker=".", ls="", color='C1')

    ax = axs[0, 1]
    extra_label = r"" if sim.dimensions else r"h / U_{\infty}"
    ax.set_xlabel(r"${:s} \partial_y u $".format(extra_label), fontsize=ftSz2)
    if not quadrature:
        ax.set_title("Strain rate profile", fontsize=ftSz1)
        ax.plot(du_ana_fit, y_ana, label="Analytical", color="C0", alpha=alp, lw=lw)
        ax.plot(du_gauss, y_gauss, color="C3", ls="", marker='x', ms=6, label='Gauss pt')
        ax.plot(*make_step(du_discrete, y_discrete, 2), '-o', ms=5, color='C1', label='Numerical')    
    else:
        ax.set_title(r"Strain rate norm profile -- $\dot\gamma(u)$", fontsize=ftSz1)
        ax.plot(np.abs(du_ana_fit), y_ana, label="Analytical", color="C0", alpha=alp, lw=lw)
        ax.plot(np.abs(du_gauss), y_gauss, color="C3", ls="", marker='x', ms=10, label='Gauss pt')
        tmp_x, tmp_y = make_step(du_dense, y_dense, sim.plot_density+1)
        ax.plot(np.abs(tmp_x), tmp_y, '-', color='C1', lw=3, label='FEM solution')
        ax.plot(
            *make_step(du_discrete * np.sign(-y_discrete), y_discrete, 2), 
            '--', ms=5, color='C2', label='Seen by quadrature'
        )

    if not mini_display:
        ax = axs[0, 2]
        extra_label = r"\tau_0" if sim.tau_zero > 0. else r"h \partial_x p"
        extra_label = r"" if sim.dimensions else r"\:/\: {:s}".format(extra_label)
        ax.set_xlabel(r"$|\tau_{{xy}}| {:s}$".format(extra_label), fontsize=ftSz2)
        ax.set_title("Shear stress profile", fontsize=ftSz1)
        ax.plot(tau_ana_fit, y_ana, label="Analytical", color="C0", alpha=alp, lw=lw)
        ax.plot(tau_gauss, y_gauss, color="C3", ls="", marker='x', ms=6, label='Gauss pt')
        ax.plot([], [], color='C1', ls='-', marker='o', label="Numerical")
        ax.plot(*make_step(tau_dense, y_dense, sim.plot_density+1), '-', ms=5, color='C1')
        ax.plot(*make_step(tau_discrete, y_discrete, 2), 'o', ls="", ms=5, color='C1')

        zipped = [axs[1, :]]
        zipped += [[y_all[::1 + (sim.degree == 1)], y_gauss, y_gauss]]
        zipped += [[u_ana_nodes[::1 + (sim.degree == 1)], du_ana_gauss, tau_ana_gauss]]
        zipped += [[u_num, du_gauss, tau_gauss]]
        zipped += [[u_ana_dense, du_ana_dense, tau_ana_dense]]
        zipped += [[u_dense, du_dense, tau_dense]]

        for ax, y_pts, phi_ana_pts, phi_pts, phi_ana_dense, phi_dense in zip(*zipped):
            rel_error_nodes = (phi_pts - phi_ana_pts) / np.amax(np.abs(phi_ana_pts))
            rel_error_dense = (phi_dense - phi_ana_dense) / np.amax(np.abs(phi_ana_dense))
            ax.plot(rel_error_nodes, y_pts, marker="x", ls="", label="Error", color='red')
            if (sim.degree == 2) and (ax==axs[1, 0]) and (1 == 0):
                x_vals, y_vals = rel_error_dense, y_dense
            else:
                x_vals, y_vals = rel_error_nodes, y_pts
            ax.plot(x_vals, y_vals, ls="-", color='red', alpha=al)

    for ax in axs.flatten():
        xmin, xmax = ax.get_xbound()
        y_pos = sim.y_zero if sim.dimensions else 2. * sim.y_zero / sim.H
        ax.fill_between([xmin, xmax], [-y_pos, -y_pos], [y_pos, y_pos], color='grey', alpha=0.25)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=ftSz3)
        ax.grid(ls=':')
    for ax in axs[:, 0]:
        extra_label = r"" if sim.dimensions else r" / (h/2)"
        ax.set_ylabel(r"$y {:s}$".format(extra_label), fontsize=ftSz2)
    if not mini_display:
        for ax, phi in zip(axs[1, :], [r'u', r'\partial_y u', r'\tau']):
            y_pos = sim.H / 2. if sim.dimensions else 1.
            # ax.vlines(0., -y_pos, y_pos, color='black', alpha=0.5)
            ax.plot(np.zeros_like(y_discrete), y_discrete, '-_k', lw=1., alpha=0.5)
            xlabel = r"$({:s}^h - {:s}) / {:s}_{{max}}$".format(phi, phi, phi)
            ax.set_xlabel(xlabel, fontsize=ftSz2)

    fig.tight_layout()
    if sim.save:
        path = "../figures/"
        # filename = f"res_P{sim.degree:d}_{extra_name:s}"
        # filename = f"sensibility_1D_quadrature"
        filename = f"result_1D_P{sim.degree:d}"
        fig.savefig(f"{path:s}{filename:s}.svg", bbox_inches="tight", transparent=True)
        # plt.show()
    else:
        plt.show()
    return
