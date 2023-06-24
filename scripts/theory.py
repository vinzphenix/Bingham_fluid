import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

ftSz1, ftSz2, ftSz3 = 16, 14, 12


def plot_interior_point(save=False):
    nx, ny = 500, 500
    x1, y1 = np.linspace(0.8, 2.3, nx), np.linspace(0.4, 1.4, ny)  # 1.3, 0.9
    xs = np.linspace(1.3, 2.3)
    x, y = np.meshgrid(x1, y1)

    nlevels = 15
    levels_list = [np.linspace(0.1, 3, nlevels), np.linspace(
        1.3, 4, nlevels), np.linspace(11.5, 20., nlevels)]

    nsteps = 101
    mu_range = np.geomspace(0.1, 10., nsteps)
    x_path, y_path = np.empty(nsteps), np.empty(nsteps)
    for i, mu in enumerate(mu_range):
        fmu = x / mu - np.log(2. * x + y - 3.) - np.log(1. - 2. *
                                                        x + 4 * y) - np.log(5. - x - 3 * y)
        fmu = np.nan_to_num(fmu, nan=100.)
        idx = np.unravel_index(fmu.argmin(), fmu.shape)
        x_path[i], y_path[i] = x[idx], y[idx]

    fig, axs = plt.subplots(1, 3, figsize=(10., 4.), constrained_layout=True, sharey='all')
    for mu, levels, ax in zip([10., 1., 0.1], levels_list, axs):
        label_1 = "Constraints" if ax == axs[0] else ""
        label_2 = r"Level sets of $f_{\mu}$" if ax == axs[1] else ""
        label_3 = "Central path" if ax == axs[2] else ""

        ax.plot(x1, (5. - x1) / 3., color='black')  # 1st side
        ax.plot((3 - y1) / 2., y1, color='black')  # 2nd side
        ax.plot(xs, (2 * xs - 1) / 4., color='black', label=label_1)  # 3rd side

        fmu = x / mu - np.log(2. * x + y - 3.) \
            - np.log(1. - 2. * x + 4 * y) \
            - np.log(5. - x - 3 * y)
        fmu = np.nan_to_num(fmu, nan=100.)

        idx = np.unravel_index(fmu.argmin(), fmu.shape)
        ax.plot(
            x_path[mu_range >= mu], y_path[mu_range >= mu],
            ls='-', color='C1', label=label_3
        )
        ax.plot(x[idx], y[idx], ls='', marker='o', ms=10, color='C1')

        ax.set_title(r"$\mu = {{{:.1f}}}$".format(mu), fontsize=ftSz1)
        ax.contour(x, y, fmu, colors='C2', levels=levels)
        ax.plot([], [], ls='-', color='C2', label=label_2)
        ax.set_aspect('equal')
        ax.grid(ls=':')
        ax.set_ylim(0.1, 1.5)
        ax.set_xlabel(r"$x$", fontsize=ftSz2)
        ax.legend(fontsize=ftSz2, loc="lower right")

    axs[0].set_ylabel(r"$y$", fontsize=ftSz2)

    if save:
        plt.savefig(
            path + "interior_point_example.svg",
            format="svg", bbox_inches="tight",
            transparent=True
        )
    else:
        plt.show()

    return


def plot_interior_point_steps(save=False):
    nx, ny = 300, 300
    x1, y1 = np.linspace(0.8, 2.3, nx), np.linspace(1.4, 0.9, ny)  # 1.3, 0.9
    x2, y2 = np.linspace(0.8, 1.2333, nx), np.linspace(1.4, 0.1, ny)  # 1.3, 0.9
    x3, y3 = np.linspace(1.2333, 2.3, nx), np.linspace(0.1, 0.9, ny)  # 1.3, 0.9
    x, y = np.meshgrid(x1, y2)

    fig, ax = plt.subplots(1, 1, figsize=(5., 5.))

    ax.plot(x1, (5. - x1) / 3., color='black')  # 1st side
    ax.plot(0.8 - (y2 - 1.4) / 3.00, y2, color='black')  # 2nd side
    ax.plot(2.3 + (y3 - 0.9) / 0.75, y3, color='black')  # 3rd side
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(ls=':')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_ylim(0.09, 1.41)

    fps, t_anim = 15, 5.0
    nt = int(t_anim * fps)

    central_path, = ax.plot([], [], ls='-', lw=3, color='red', marker="o", markevery=[-1])
    mu_text = ax.text(0.7, 0.9, "", fontsize=1.2 * ftSz1, transform=ax.transAxes)
    x_path, y_path = np.empty(nt + 1), np.empty(nt + 1)
    mu_range = 10. * np.exp(-5.75 * np.arange(0, nt + 1) / nt)
    nlevels = 40
    contourl = ax.contour(x, y, x, alpha=0.5)
    contourf = ax.contourf(x, y, x, cmap=plt.get_cmap("viridis"), levels=nlevels)

    def update(i):
        nonlocal contourf, contourl
        mu = mu_range[i]
        fmu = x / mu - (
            + np.log((+5.0 - 1. * x - 3. * y) / np.sqrt(10)) +
            np.log((-3.8 + 3. * x + 1. * y) / np.sqrt(10)) +
            np.log((+3.3 - 3. * x + 4. * y) / np.sqrt(25))
        )
        fmu = np.nan_to_num(fmu, nan=100.)
        # fmu[~mask] = 100.
        fmu -= np.amin(fmu)
        # print(np.amax(fmu[fmu <= 100000]))

        idx = np.unravel_index(fmu.argmin(), fmu.shape)
        x_path[i], y_path[i] = x[idx], y[idx]

        central_path.set_data(x_path[mu_range >= mu], y_path[mu_range >= mu])
        mu_text.set_text(r"$\mu = {{{:.2f}}}$".format(mu))
        # if contourf is not None:
        for c in contourf.collections:
            c.remove()
        for c in contourl.collections:
            c.remove()
        levels = np.arange(0., 51., 0.5)
        contourf = ax.contourf(x, y, fmu, cmap=plt.get_cmap("Spectral_r"), levels=levels)
        contourl = ax.contour(x, y, fmu, levels=levels, colors='k', alpha=0.5, linewidths=.5)
        return contourf, contourl

    fig.tight_layout()
    path_anim = f"./anim/interior_point"

    save = "mp4"
    if save == "none":
        anim = FuncAnimation(fig, update, nt + 1, interval=20, repeat=False)
        plt.show()

    elif save == "gif":
        anim = FuncAnimation(fig, update, nt + 1, interval=20, repeat=False)
        writerGIF = PillowWriter(fps=fps)
        anim.save(f"{path_anim}.gif", writer=writerGIF, savefig_kwargs={"transparent": True})

    elif save == "mp4":
        anim = FuncAnimation(fig, update, nt + 1, interval=20, repeat=False)
        writerMP4 = FFMpegWriter(fps=fps)
        anim.save(f"{path_anim}.mp4", writer=writerMP4, savefig_kwargs={"transparent": True})

    return


def plot_poiseuille_profiles(save=False):
    def eval_velocity(bn, eta):
        u_ana, du_ana = np.zeros(eta.size), np.zeros(eta.size)
        e0 = bn / 4.
        m_bot = eta <= -e0
        m_mid = (-e0 < eta) & (eta < e0)
        m_top = e0 <= eta
        u_ana[m_top] = -bn / 2. * (1. - eta[m_top]) + (1. - np.square(eta[m_top]))
        u_ana[m_bot] = -bn / 2. * (1. + eta[m_bot]) + (1. - np.square(eta[m_bot]))
        u_ana[m_mid] = (1. - bn / 4.) ** 2
        return u_ana

    fig, ax = plt.subplots(1, 1, figsize=(6. * 0.9, 3.5 * 0.9))
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=5)
    ax.axis([0., 1., -1., 1.])
    ax.grid(ls=':')
    ax.set_xlabel(r"$u/U_{\infty}$", fontsize=ftSz2)
    ax.set_ylabel(r"$2y/h$", fontsize=ftSz2)
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)

    eta = np.linspace(-1., 1., 150)
    for i, bn in enumerate([0., 1., 2., 3., 4.]):
        c, lw = (f'C{i:d}', 1.) if i < 4 else ('k', 3.)
        u = eval_velocity(bn, eta)
        umax = (1. - bn / 4.) ** 2
        eta_zero = np.full(2, bn / 4.)
        ax.plot(u, eta, color=c, label=f"$Bn = {bn:.0f}$", lw=lw)
        ax.fill_between([0., umax], +eta_zero, -eta_zero, color=c, ls='', alpha=0.50)

    lgd = ax.legend(fontsize=ftSz3, ncols=1, labelspacing=1., bbox_to_anchor=(1., .86),)
    for legobj in lgd.legend_handles:
        legobj.set_linewidth(3.0)
    fig.tight_layout()

    if save:
        plt.savefig(path + "poiseuille_bn.svg", bbox_inches="tight", transparent=True)
    else:
        plt.show()

    return


def plot_shape_fct_1D(save=False):
    xi = np.linspace(-1., 1., 200)
    xi_1 = np.linspace(-1., 1., 2)
    xi_2 = np.linspace(-1., 1., 3)

    fig, axs = plt.subplots(1, 2, figsize=(8., 3.), constrained_layout=True,
                            sharey=True, sharex=True)

    axs[0].set_title("Linear", fontsize=ftSz1)
    axs[0].plot(xi, (1. - xi) / 2., color='C0', label=r"$\phi_1$")
    axs[0].plot(xi, (xi + 1.) / 2., color='C1', label=r"$\phi_2$")
    axs[0].plot(xi_1, (1. - xi_1) / 2., color='C0', ls='', marker='o')
    axs[0].plot(xi_1, (xi_1 + 1.) / 2., color='C1', ls='', marker='o')

    axs[1].set_title("Quadratic", fontsize=ftSz1)
    axs[1].plot(xi, xi * (xi - 1.) / 2., color='C0', label=r"$\phi_1$")
    axs[1].plot(xi, xi * (xi + 1.) / 2., color='C1', label=r"$\phi_2$")
    axs[1].plot(xi, (1. - xi) * (1. + xi), color='C2', label=r"$\phi_3$")
    axs[1].plot(xi_2, xi_2 * (xi_2 - 1.) / 2., color='C0', ls='', marker='o')
    axs[1].plot(xi_2, (1. - xi_2) * (1. + xi_2), color='C2', ls='', marker='o')
    axs[1].plot(xi_2, xi_2 * (xi_2 + 1.) / 2., color='C1', ls='', marker='o')

    for ax, this_xi in zip(axs, [xi_1, xi_2]):
        ax.plot(xi_1, np.zeros_like(xi_1), marker='|', ls='',
                markersize=20, color='black')  # bounds
        ax.plot(this_xi, np.zeros_like(this_xi), marker='o',
                lw=4, alpha=0.5, color='black')  # nodes
        ax.legend(fontsize=ftSz2, loc="center right")
        ax.grid(ls=':')

    if save:
        fig.savefig(path + "shape_fct_1D.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()
    return


def plot_fluid_models(save=False):
    fig, axs = plt.subplots(
        1, 2, figsize=(7., 4. * 0.875), constrained_layout=True, sharey='all', sharex='all'
    )

    powers = [0.4, 2., 1.]
    tau_y = 1.
    strain = 1.5 * np.linspace(0., 1., 200) ** 2
    titles = [r"Power-law $\quad \tau_0 = 0$", r"Herschel-Bulkey $ \quad \tau_0 > 0$"]
    labels = [
        [r"Shear thinning", r"Shear thickening", r"Newtonian"],
        [r"", r"", r"Bingham"],
    ]

    axs[0].set_ylabel(r"Shear stress $\tau$ \; [Pa]", fontsize=ftSz2)
    axs[0].set_xlim([-0.03, strain[-1]])
    axs[0].set_ylim([-0.05, tau_y + strain[-1]])

    for ax, tau, title, lbs in zip(axs, [0., tau_y], titles, labels):

        for i, (power, label) in enumerate(zip(powers, lbs)):
            # alpha, lw = (1., 2.5) if power == 1. else (0.75, 1.5)
            alpha, lw = (1., 2.5) if (power == 1. and tau > 0.) else (0.25, 1.5)
            stress = np.r_[0., tau + strain ** power]
            ls = "--" if (tau == 0. and power == 1.) else "-"
            ax.plot(np.r_[0., strain], stress, f'C{i:d}{ls:s}', lw=lw, label=label, alpha=alpha)

        # loc = "upper left" if tau == 0. else "lower right"
        # ax.legend(fontsize=ftSz3, loc=loc)
        ax.set_title(title, fontsize=ftSz1)
        ax.set_xlabel(r"Strain rate $\dot\gamma$ \; [1/s]", fontsize=ftSz2)

    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(line_or_label, []) for line_or_label in zip(*lines_labels)]
    # _, idxs = np.unique(labels, return_index=True)
    for ax in axs:
        ax.tick_params(axis='both', direction='in', bottom=True, left=True, top=True, right=True)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    _ = axs[0].legend(
        lines, labels,  # bbox_to_anchor=(1.0, 0.25, 0.18, 0.5),
        facecolor='wheat', framealpha=0.25, fancybox=True,
        # labelspacing=1., mode='expand',
        fontsize=ftSz2, ncol=1,
    )

    arrow_x = 0.1
    axs[1].arrow(
        arrow_x, 0., 0., tau_y, width=0.02, head_width=0.08,
        length_includes_head=True, edgecolor="none", facecolor='black'
    )
    axs[1].text(
        arrow_x + 0.05, tau_y / 2., r"$\tau_0$",
        fontsize=1.2 * ftSz1, ha='left', va='center'
    )
    # plt.arrow

    if save:
        fig.savefig(path + "fluid_classification_slide.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()
    return


if __name__ == "__main__":
    save_global = False
    path = "./figures/"
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.usetex"] = True

    plot_interior_point_steps(save_global)
    # plot_interior_point(save_global)
    # plot_poiseuille_profiles(save_global)
    # plot_shape_fct_1D(save_global)
    # plot_fluid_models(save_global)
