import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nx, ny = 500, 500
    x1, y1 = np.linspace(0.8, 2.3, nx), np.linspace(0.4, 1.4, ny) # 1.3, 0.9
    xs = np.linspace(1.3, 2.3)
    x, y = np.meshgrid(x1, y1)

    nlevels = 15
    levels_list = [np.linspace(0.1, 3, nlevels), np.linspace(1.3, 4, nlevels), np.linspace(11.5, 20., nlevels)]
    

    nsteps = 101
    mu_range = np.geomspace(0.1, 10., nsteps)
    x_path, y_path = np.empty(nsteps), np.empty(nsteps)
    for i, mu in enumerate(mu_range):
        fmu = x / mu - np.log(2.*x+y-3.) - np.log(1. - 2.*x + 4*y) - np.log(5. - x - 3*y)
        fmu = np.nan_to_num(fmu, nan=100.)
        idx = np.unravel_index(fmu.argmin(), fmu.shape)
        x_path[i], y_path[i] = x[idx], y[idx]

    fig, axs = plt.subplots(1, 3, figsize=(12., 4.), constrained_layout=True, sharey='all')
    for mu, levels, ax in zip([10., 1., 0.1], levels_list, axs):
        ax.plot(x1, (5. - x1) / 3., color='black')  # 1st side
        ax.plot((3 - y1) / 2., y1, color='black')  # 2nd side
        ax.plot(xs, (2 * xs - 1) / 4., color='black', label='constraints')  # 3rd side

        fmu = x / mu - np.log(2.*x+y-3.) - np.log(1. - 2.*x + 4*y) - np.log(5. - x - 3*y)
        fmu = np.nan_to_num(fmu, nan=100.)

        idx = np.unravel_index(fmu.argmin(), fmu.shape)
        ax.plot(x_path[mu_range >= mu], y_path[mu_range >= mu], ls='-', color='C1', label='central path')
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
    # plt.savefig("./figures/interior_point_example.svg", format="svg", bbox_inches="tight")
    plt.show()