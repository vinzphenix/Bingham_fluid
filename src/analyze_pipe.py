import matplotlib.pyplot as plt
import numpy as np
import gmsh

# save_global = False
# path = "../figures/"
plt.rcParams['font.family'] = 'serif'
plt.rcParams["text.usetex"] = True
ftSz1, ftSz2, ftSz3 = 16, 14, 12


def save_profiles(tags, filename, n_pts=150):
    xy_exprs = [
        ("1/2", "u", 0., 1., "Inflow"),
        # ("2", "u", 0., 1., "End unyielded"),
        ("2 + 1/2", "u", 0., 1., "Yielded"),
        # ("2 + 1/2 + u*cos(pi*2/5)", "-1 + u*sin(pi*2/5)", 1., 2., "Start unyielded"),
        ("2 + 1/2 + u*cos(pi*1/100)", "-1 + u*sin(pi*1/100)", 1., 2., "Outflow"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(10., 4.))

    for i, (x_expr, y_expr, u_min, u_max, label) in enumerate(xy_exprs):
        gmsh.plugin.setString('CutParametric', option="X", value=x_expr)
        gmsh.plugin.setString('CutParametric', option="Y", value=y_expr)
        gmsh.plugin.setString('CutParametric', option="Z", value="0")
        gmsh.plugin.setNumber('CutParametric', option="MinU", value=u_min + 1.e-3)
        gmsh.plugin.setNumber('CutParametric', option="MaxU", value=u_max - 1.e-3)
        gmsh.plugin.setNumber('CutParametric', option="NumPointsU", value=n_pts)
        gmsh.plugin.setNumber('CutParametric', option="MinV", value=0.)
        gmsh.plugin.setNumber('CutParametric', option="MaxV", value=0.)
        gmsh.plugin.setNumber('CutParametric', option="NumPointsV", value=1)
        gmsh.plugin.setNumber('CutParametric', option="ConnectPoints", value=0)

        for j, (ax, tag) in enumerate(zip(axs, tags)):
            gmsh.plugin.setNumber('CutParametric', option="View", value=tag - 1)
            tag_cut = gmsh.plugin.run('CutParametric')

            _, _, data = gmsh.view.getListData(tag_cut)
            gmsh.view.remove(tag_cut)

            data = np.array(data).reshape((n_pts, -1))
            
            x_value = np.linspace(0., 1., n_pts)
            # x_value = data[:, 1]
            if j < 2:
                y_value = np.linalg.norm(data[:, 3:], axis=1)
            else:
                y_value = data[:, 3]

            ax.plot(x_value, y_value, color=f'C{i:d}', label=label)

    for ax, title in zip(axs, ["Tangential velocity", "Strain rate norm", "Vorticity"]):
        ax.set_title(title, fontsize=ftSz1)
        ax.set_xlabel(r"Radial position", fontsize=ftSz2)
        ax.legend(fontsize=ftSz3, ncols=1)
        ax.tick_params(axis='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.grid(ls=':')

    plt.tight_layout()
    if filename != "":
        fig.savefig(f"{filename:s}.svg", format="svg", transparent=False, bbox_inches="tight")
    else:
        plt.show()

    return
