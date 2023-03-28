import numpy as np
import matplotlib.pyplot as plt

save = True
ftSz1, ftSz2, ftSz3 = 16, 14, 12
plt.rcParams["text.usetex"] = save

if __name__ == "__main__":

    xi = np.linspace(-1., 1., 200)
    xi_1 = np.linspace(-1., 1., 2)
    xi_2 = np.linspace(-1., 1., 3)

    fig, axs = plt.subplots(1, 2, figsize=(8., 3.), constrained_layout=True, sharey=True, sharex=True)

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
        ax.plot(xi_1, np.zeros_like(xi_1), marker='|', ls='', markersize=20, color='black')  # bounds
        ax.plot(this_xi, np.zeros_like(this_xi), marker='o', lw=4, alpha=0.5, color='black')  # nodes
        ax.legend(fontsize=ftSz2, loc="center right")
        ax.grid(ls=':')
    
    if save:
        fig.savefig("./figures/shape_fct_1D.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()