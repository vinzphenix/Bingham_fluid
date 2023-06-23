import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon, FancyArrow

ftSz1, ftSz2, ftSz3 = 20, 14, 12


def update_all(i):
    global x_axis_patch, y_axis_patch
    if i > 0:
        nodes[:] = nodes + dt * np.dot(tsfm, nodes.T).T
    if i < nt:
        particles[:, :, i + 1] = particles[:, :, i] + dt * np.dot(tsfm, particles[:, :, i].T).T

    shape.set_xy(nodes[:4])
    x_axis.set_data(dx=nodes[4, 0], dy=nodes[4, 1])
    y_axis.set_data(dx=nodes[5, 0], dy=nodes[5, 1])

    start = max(0, i - 10)
    for p in range(particles.shape[0]):
        ps[p].set_data([particles[p, 0, start:i + 1]], [particles[p, 1, start:i + 1]])

    return


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.usetex"] = True

    save = "none"
    # mode = "stretch"
    mode = "shear"
    # mode = "rotation"

    fps = 25
    t_anim = 2.
    tend, nt = 0.5, int(t_anim * fps)
    dt = tend / nt

    if mode == "stretch":
        text_str = r"$\mathbf{D}_{s}$"
        tsfm = np.array([
            [1., 0.],
            [0., 1.],
        ]) * 0.875
    elif mode == "shear":
        text_str = r"$\mathbf{D}_{d}$"
        tsfm = np.array([
            [0., 1.],
            [1., 0.],
        ])
    elif mode == "rotation":
        text_str = r"$\mathbf{W}$"
        tsfm = np.array([
            [0., -1.],
            [1., 0.],
        ]) * np.pi / 2
    else:
        raise ValueError("mode is unknown")

    fig, ax = plt.subplots(1, 1, figsize=(5., 5.))

    x_axis = FancyArrow(0., 0., 1., 0., width=0.01, color='k', zorder=10, head_width=0.1)
    y_axis = FancyArrow(0., 0., 0., 1., width=0.01, color='k', zorder=10, head_width=0.1)
    ax.add_patch(x_axis)
    ax.add_patch(y_axis)

    nx = 11
    particles = np.zeros((nx * nx, 2, nt + 1))
    velocity = np.zeros((nx * nx, 2))
    x = np.linspace(-1., 1., nx)
    X, Y = np.meshgrid(x, x)
    particles[:, 0, 0] = X.flatten()
    particles[:, 1, 0] = Y.flatten()
    ps = [
        ax.plot([], [], '-', ms=3., lw=1.5, color='k', zorder=20, markevery=[-1])[0]
        for p in range(nx * nx)
    ]

    nodes = np.array([
        [-1., -1.],
        [+1., -1.],
        [+1., +1.],
        [-1., +1.],
        [+1., +0.],
        [+0., +1.],
    ])
    shape = Polygon(nodes, color='lightgrey')
    ax.add_patch(shape)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")

    text = ax.text(0.10, 0.85, text_str, fontsize=25, transform=ax.transAxes)
    fig.tight_layout()
    path_anim = f"./anim/kinematics_{mode:s}"

    if save == "none":
        anim = FuncAnimation(fig, update_all, nt + 1, interval=20, repeat=False)
        plt.show()

    elif save == "gif":
        anim = FuncAnimation(fig, update_all, nt + 1, interval=20, repeat=False)
        writerGIF = PillowWriter(fps=fps)
        anim.save(f"{path_anim}.gif", writer=writerGIF, savefig_kwargs={"transparent": True})

    elif save == "mp4":
        anim = FuncAnimation(fig, update_all, nt + 1, interval=20, repeat=False)
        writerMP4 = FFMpegWriter(fps=fps)
        anim.save(f"{path_anim}.mp4", writer=writerMP4, savefig_kwargs={"transparent": True})
