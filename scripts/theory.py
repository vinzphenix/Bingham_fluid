import numpy as np
import matplotlib.pyplot as plt

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

    fig, axs = plt.subplots(1, 3, figsize=(12., 4.), constrained_layout=True, sharey='all')
    for mu, levels, ax in zip([10., 1., 0.1], levels_list, axs):
        ax.plot(x1, (5. - x1) / 3., color='black')  # 1st side
        ax.plot((3 - y1) / 2., y1, color='black')  # 2nd side
        ax.plot(xs, (2 * xs - 1) / 4., color='black', label='constraints')  # 3rd side

        fmu = x / mu - np.log(2. * x + y - 3.) - np.log(1. - 2. *
                                                        x + 4 * y) - np.log(5. - x - 3 * y)
        fmu = np.nan_to_num(fmu, nan=100.)

        idx = np.unravel_index(fmu.argmin(), fmu.shape)
        ax.plot(x_path[mu_range >= mu], y_path[mu_range >= mu],
                ls='-', color='C1', label='central path')
        ax.plot(x[idx], y[idx], ls='', marker='o', ms=10, color='C1')

        ax.set_title(r"$\mu = {{{:.1f}}}$".format(mu), fontsize=14)
        ax.contour(x, y, fmu, colors='C2', levels=levels)
        ax.plot([], [], ls='-', color='C2', label=r'level set of $f_{\mu}$')
        ax.set_aspect('equal')
        ax.grid(ls=':')
        ax.set_ylim(0.1, 1.5)
        ax.set_xlabel(r"$x$", fontsize=13)

    axs[0].set_ylabel(r"$y$", fontsize=13)
    axs[-1].legend(fontsize=13)
    # plt.savefig(path + "interior_point_example.svg", format="svg", bbox_inches="tight")
    plt.show()


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
        1, 2, figsize=(7., 4.*0.875), constrained_layout=True, sharey='all', sharex='all'
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
            alpha, lw = (1., 2.5) if power == 1. else (0.75, 1.5)
            stress = np.r_[0., tau + strain ** power]
            ls = "--" if (tau == 1. and power == 1.) else "-"
            ax.plot(np.r_[0., strain], stress, f'C{i:d}{ls:s}', lw=lw, label=label, alpha=alpha)
        
        # loc = "upper left" if tau == 0. else "lower right"
        # ax.legend(fontsize=ftSz3, loc=loc)
        ax.set_title(title, fontsize=ftSz1)
        ax.set_xlabel(r"Strain rate $\dot\gamma$ \; [1/s]", fontsize=ftSz2)

    lines_labels = [ax.get_legend_handles_labels() for ax in axs]
    lines, labels = [sum(line_or_label, []) for line_or_label in zip(*lines_labels)]
    # _, idxs = np.unique(labels, return_index=True)

    _ = axs[0].legend(
        lines, labels, #bbox_to_anchor=(1.0, 0.25, 0.18, 0.5),
        facecolor='wheat', framealpha=0.25, fancybox=True,
        #labelspacing=1., mode='expand', 
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
        fig.savefig(path + "fluid_classification.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()
    return


if __name__ == "__main__":
    save_global = True
    path = "./figures/"
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.usetex"] = save_global

    # plot_interior_point(save_global)
    # plot_shape_fct_1D(save_global)
    plot_fluid_models(save_global)
