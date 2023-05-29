import matplotlib.pyplot as plt
import numpy as np
from bingham_structure import *
from bingham_run import load_solution


def get_profile(model, variant, x, y, n_pts):
    if variant == 500:
        model += "_cheat"

    gmsh.initialize()

    parameters, u_field, p_field, d_field, coords = load_solution(model, str(variant))
    sim = Simulation_2D(parameters, new_coords=coords)

    velocity = np.c_[u_field, np.zeros(sim.n_node)].flatten()

    # gmsh.fltk.initialize()

    tag_v = gmsh.view.add("Velocity", tag=1)
    gmsh.view.addHomogeneousModelData(
        tag_v, 0, sim.model_name, "NodeData",
        sim.node_tags + 1, velocity, numComponents=3
    )
    gmsh.view.option.setNumber(tag_v, "Visible", 0)

    gmsh.plugin.setNumber('Gradient', option="View", value=tag_v - 1)
    tag_grad = gmsh.plugin.run('Gradient')

    x_expr = f"{x[0]:.5f}"
    y_expr = "-u*u"
    u_min, u_max = y[0], y[1]
    # datas = [None, None]

    # for i, tag in enumerate([tag_v, tag_grad]):
    gmsh.plugin.setString('CutParametric', option="X", value=x_expr)
    gmsh.plugin.setString('CutParametric', option="Y", value=y_expr)
    gmsh.plugin.setString('CutParametric', option="Z", value="0")
    gmsh.plugin.setNumber('CutParametric', option="MinU", value=u_min)
    gmsh.plugin.setNumber('CutParametric', option="MaxU", value=u_max)
    gmsh.plugin.setNumber('CutParametric', option="NumPointsU", value=n_pts)
    gmsh.plugin.setNumber('CutParametric', option="MinV", value=0.)
    gmsh.plugin.setNumber('CutParametric', option="MaxV", value=0.)
    gmsh.plugin.setNumber('CutParametric', option="NumPointsV", value=1)
    gmsh.plugin.setNumber('CutParametric', option="ConnectPoints", value=0)
    gmsh.plugin.setNumber('CutParametric', option="View", value=tag_v - 1)
    tag_cut = gmsh.plugin.run('CutParametric')
    gmsh.view.option.setNumber(tag_cut, "Visible", 1)
    gmsh.view.option.setNumber(tag_cut, "ArrowSizeMax", 500)

    _, _, data = gmsh.view.getListData(tag_cut)

    data = np.array(data).reshape((n_pts, -1))
    # datas[i] = data

    # gmsh.fltk.run()
    # gmsh.fltk.finalize()

    gmsh.finalize()
    return data, sim.tau_zero


def plot_profiles(model, variants, n_pts=200, save=False):

    def find_stagnation():
        min_y = -0.4
        idx_switch = np.where((x_data[:-1] <= 0.) & (x_data[1:] > 0.) & (y_data[:-1] > min_y))
        idx_switch = idx_switch[0][0]

        ym, yp = y_data[idx_switch], y_data[idx_switch + 1]
        um, up = x_data[idx_switch], x_data[idx_switch + 1]
        alpha = -um / (up - um)
        return ym * (1. - alpha) + yp * alpha

    fig, axs = plt.subplots(1, 2, figsize=(7., 7.*2./3.), constrained_layout=True, sharey='all')
    colors = plt.get_cmap(cmap_name)(np.linspace(0., 1., len(variants)))

    for i, (variant, color) in enumerate(zip(variants, colors)):
        print(i)
        data, bn = get_profile(model, variant, x=[0.5], y=[-1., 0.], n_pts=n_pts)

        x_data = data[:, 3 + 0]
        y_data = data[:, 0 + 1]
        y_stag = find_stagnation()
        axs[0].plot(x_data, y_data, color=color, label=r"$Bn={:.0f}$".format(bn))
        axs[0].plot([0.], [y_stag], 'o', color=color, ms=5, zorder=-i)

        axs[1].plot(np.gradient(x_data, y_data), y_data, color=color, zorder=-i)

    axs[0].set_title(r"Velocity profile", fontsize=ftSz1)
    axs[0].set_xlabel(r"$u$", fontsize=ftSz2)
    axs[0].set_ylabel(r"$y$", fontsize=ftSz2)
    axs[0].axvline(x=0., ls='--', color='k', alpha=0.75)
    axs[0].set_xlim(axs[0].get_xlim()[0], 0.45)
    axs[0].set_ylim(-1., 0.)
    axs[0].legend(fontsize=ftSz3, ncols=1)

    axs[1].set_title(r"Velocity gradient profile", fontsize=ftSz1)
    axs[1].set_xlabel(r"$\partial_y u$", fontsize=ftSz2)
    # axs[1].set_ylabel(r"$y/H$", fontsize=ftSz2)
    axs[1].set_xlim(-1.5, 8.5)
    axs[1].axvline(x=0., ls='--', color='k', alpha=0.75)
    # axs[1].set_xlim(0., 1.)

    for ax in axs:
        ax.grid(ls=':')
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)

    if save:
        fig.savefig(path + f"profile_{model:s}.svg", format="svg", transparent=True)
    else:
        plt.show()

    return


def plot_comparison(model, variants, n_pts=200, save=False):

    n_variants_overall = 6
    kwargs1 = dict(marker="^", ls="",)
    kwargs2 = dict(marker="o", ls="", markerfacecolor=(1., 1., 1., 0.), markeredgewidth=1.)
    papers = ["Syrakos et al.", "Bleyer et al."]

    data_papers = np.loadtxt("../docs/papers_profiles.txt")
    bn_papers = data_papers[:, 0]
    x_papers = data_papers[:, 1::2]
    y_papers = data_papers[:, 2::2] - 1.

    fig, ax = plt.subplots(1, 1, figsize=(6., 4.), constrained_layout=True)
    colors = plt.get_cmap(cmap_name)(np.linspace(0., 1., n_variants_overall))
    colors = colors[[0, 3, 5]]

    for i, (variant, color) in enumerate(zip(variants, colors)):
        data, bn = get_profile(model, variant, x=[0.5], y=[-1., 0.], n_pts=n_pts)
        x_data = data[:, 3 + 0]
        y_data = data[:, 0 + 1]
        ax.plot(x_data, y_data, color=color, label=r"$Bn={:.0f}$".format(bn))

    for i, (kwargs, label) in enumerate(zip([kwargs1, kwargs2], papers)):

        ax.plot([], [], color="k", markeredgecolor="k", label=label, **kwargs)
        for j, color in enumerate(colors):
            x_data = x_papers[3 * i + j]
            y_data = y_papers[3 * i + j]
            ax.plot(x_data, y_data, color=color, markeredgecolor=color, **kwargs)

    ax.set_xlabel(r"$u$", fontsize=ftSz2)
    ax.set_ylabel(r"$y$", fontsize=ftSz2)
    ax.axvline(x=0., ls='--', color='k', alpha=0.75)
    ax.set_xlim(ax.get_xlim()[0], 0.65)
    ax.set_ylim(-1., 0.)
    ax.legend(fontsize=ftSz3, ncols=1)
    ax.grid(ls=':')
    ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)

    if save:
        fig.savefig(path + f"profile_papers_{model:s}.svg", format="svg", transparent=True)
    else:
        plt.show()

    return


if __name__ == "__main__":

    save_global = False
    path = "../figures/"
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.usetex"] = True
    ftSz1, ftSz2, ftSz3 = 16, 14, 12
    cmap_name = "Spectral_r"

    plot_profiles("cavity", [0, 1, 5, 20, 100, 500], n_pts=500, save=save_global)
    plot_comparison("cavity", [0, 20, 500], n_pts=500, save=save_global)
