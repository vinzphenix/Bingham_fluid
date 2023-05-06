import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sf_p1 = [
    (0., 0., lambda x, y: 1. - x - y),
    (1., 0., lambda x, y: x),
    (0., 1., lambda x, y: y),
]

sf_p2 = [
    (0.0, 0.0, lambda x, y: 2. * (1. - x - y) * (0.5 - x - y)),
    (0.5, 0., lambda x, y: 4. * x * (1. - x - y)),
    (0.0, 0.5, lambda x, y: 4. * y * (1. - x - y)),
    (0.5, 0.5, lambda x, y: 4. * x * y),
    (1.0, 0.0, lambda x, y: x * (2. * x - 1.)),
    (0.0, 1.0, lambda x, y: y * (2. * y - 1.)),
]

sf_bubble = [
    (0., 0., lambda x, y: (9. * x * y - 1.) * (x + y - 1)),
    (1., 0., lambda x, y: 9. * x * y * (x + y - 1) + x),
    (0., 1., lambda x, y: 9. * x * y * (x + y - 1) + y),
    (1. / 3., 1. / 3., lambda x, y: 27. * x * y * (1. - x - y)),
]


# Map rectangular grid (u, v) to triangle (x, y)
# Singularity at (0.5, 0.5)
def mapping_x(u, v): return u * (1. - 0.5 * v)
def mapping_y(u, v): return v * (1. - 0.5 * u)

# # Singularity at (0.5, 0.)
# mapping_x = lambda u, v: u * (1. - 0.5 * (1. - v))
# mapping_y = lambda u, v: v * (1. - u)
#
# # Singularity at (0., 0.5)
# mapping_x = lambda u, v: u * (1. - v)
# mapping_y = lambda u, v: v * (1. - 0.5 * (1. - u))


def parametric_surface(x, y, f, x0, y0): return [
    x0 + mapping_x(x, y),
    y0 + mapping_y(x, y),
    f(mapping_x(x, y), mapping_y(x, y))
]


def zero_fct(x, y): return 0. * x * y


positions = [
    (0., 0.),
    (1.5, 0.),
    (0., 1.5),
    (1.5, 1.5),
    (3., 0.),
    (0., 3.),
]


def plot_basis_fcts(sfs, figsize, save_name=""):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
    ax.view_init(elev=20., azim=240.)  # type: ignore

    # Make data.
    x = np.linspace(0., 1., 10)
    y = np.linspace(0., 1., 10)
    X, Y = np.meshgrid(x, y)

    x_tri = np.linspace(0., 1., 2)
    y_tri = np.linspace(0., 1., 2)
    x_tri, y_tri = np.meshgrid(x_tri, y_tri)
    x_tri, y_tri = x_tri * (1. - y_tri / 2.), y_tri * (1. - x_tri / 2.)

    for (_, _, sf), (x0, y0) in zip(sfs, positions):
        xx, yy, zz = parametric_surface(X, Y, sf, x0, y0)
        ax.plot_surface( # type: ignore
            xx, yy, zz, cmap=plt.get_cmap('coolwarm'),  # type: ignore
            linewidth=10, antialiased=False, alpha=0.75
        )
        ax.plot_wireframe( # type: ignore
            x0 + x_tri, y0 + y_tri, 0. * x_tri, rstride=10, cstride=10,
            linewidths=1., colors='black', alpha=0.5
        )
        for xi, yi, _ in sfs:
            ax.plot(
                [x0 + xi, x0 + xi], [y0 + yi, y0 + yi], [0., sf(xi, yi)],
                '-o', lw=1., color='black', alpha=0.5, ms=5
            )

    if sfs == sf_p2:
        ax.set_xlim(0.5, 4.)
        ax.set_ylim(0.5, 4.)
        ax.set_zlim(-1., 2.) # type: ignore
        top, bot = 2.3, -1.
    elif sfs == sf_p1:
        ax.set_xlim(0.3, 2.5)
        ax.set_ylim(0.2, 2.5) 
        ax.set_zlim(-0.5, 1.5) # type: ignore
        top, bot = 1.5, -0.4
    else:
        ax.set_xlim(0.3, 2.5)
        ax.set_ylim(0.2, 2.5) 
        ax.set_zlim(-0.5, 1.5) # type: ignore
        top, bot = 1.3, -0.3

    ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(top=top, bottom=bot)

    if save_name != "":
        fig.savefig(path + f"shape_fcts_2d_{save_name:s}.svg", format="svg", transparent=True)
        # fig.savefig(f"shape_fcts_2d_{save_name:s}.svg", format="svg")
    else:
        plt.show()

    return


if __name__ == "__main__":
    ftSz1, ftSz2, ftSz3 = 16, 14, 12

    save_global = True
    path = "./figures/"
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.usetex"] = save_global

    # plot_basis_fcts(sf_bubble, figsize=(5., 3.), save_name='bubble')
    # plot_basis_fcts(sf_p1, figsize=(5., 3.), save_name='P1')
    plot_basis_fcts(sf_p2, figsize=(10., 5.), save_name='P2')
