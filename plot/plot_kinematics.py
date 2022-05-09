"""
Dump plots of kinematic distributions shaded by net output.


Usage:

python plot/plot_kinematics.py


"""
import h5py
import matplotlib
import numpy
import plot_lib
from matplotlib import pyplot

FILENAME = "results/kinematics.h5"

# data

LABELS = ["AX", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
label_index = LABELS.index


def main():
    with h5py.File(FILENAME) as file_:
        data = file_["data"][:]
        phi = file_["phi"][:]

    plot_lib.set_default_context()

    cmap = make_cmap_2col(
        plot_lib.cmap_purple_orange(0),
        plot_lib.cmap_purple_orange(1),
    )

    plot_combo_t("kin_net_1_t.png", data, phi, cmap)
    plot_combo_z("kin_net_1_z.png", data, phi, cmap)


def plot_combo_t(out_path, data, phi, cmap):
    labels_y = ["BY", "CY"]
    labels_x = ["BY", "CY", "AX", "BX", "CX"]

    label_lim = {
        "AX": (0, 1600),
        "BX": (-1300, 300),
        "BY": (-800, 800),
        "CX": (-800, 800),
        "CY": (-800, 800),
    }.get

    label_ticks = {
        "AX": (0, 1000),
        "BX": (-1000, 0),
        "BY": (-500, 0, 500),
        "CX": (-500, 0, 500),
        "CY": (-500, 0, 500),
    }.get

    label_tex = {
        "AX": r"$\vec q^{\,j_1}_x$",
        "BX": r"$\vec q^{\,j_2}_x$",
        "BY": r"$\vec q^{\,j_2}_y$",
        "CX": r"$\vec q^{\,j_3}_x$",
        "CY": r"$\vec q^{\,j_3}_y$",
    }.get

    norm = matplotlib.colors.CenteredNorm(0, 1, clip=True)
    colors = cmap(norm(phi))

    colorbar_width = 0.073
    colorbar_bot = 0.23

    figure, axes = pyplot.subplots(
        len(labels_y),
        len(labels_x),
        figsize=(4.8, 2.0),
        dpi=400,
        gridspec_kw={
            "top": 0.99,
            "right": 0.99 - colorbar_width,
            "bottom": 0.22,
            "left": 0.135,
            "hspace": 0.1,
            "wspace": 0.1,
        },
    )

    for i, laby in enumerate(labels_y):
        for j, labx in enumerate(labels_x):
            axis = axes[i, j]
            axis.scatter(
                data[:, label_index(labx)],
                data[:, label_index(laby)],
                lw=0,
                c=colors,
                marker=",",
                s=0.2,
            )

            axis.set_aspect("equal")

            xmin, xmax = label_lim(labx)
            assert xmax - xmin == 1600
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(*label_lim(laby))

            xticks = label_ticks(labx)
            axis.set_xticks(xticks)

            yticks = label_ticks(laby)
            axis.set_yticks(yticks)

            if i == len(labels_y) - 1:
                axis.set_xlabel(label_tex(labx))
                axis.set_xticklabels(label_ends(xticks))
            else:
                axis.set_xticklabels([])

            if j == 0:
                axis.set_ylabel(label_tex(laby))
                axis.set_yticklabels(label_ends(yticks))
            else:
                axis.set_yticklabels([])

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    axis_colorbar = figure.add_axes(
        [1 - colorbar_width, colorbar_bot, 0.018, 0.975 - colorbar_bot]
    )

    mappable = matplotlib.cm.ScalarMappable(norm, cmap)
    figure.colorbar(mappable, cax=axis_colorbar)
    axis_colorbar.plot(
        [0.5, 0.5],
        [-1, 1],
        color="k",
        linewidth=1,
        linestyle="--",
        zorder=-0.5,
    )
    axis_colorbar.set_yticks([-1, 1])
    axis_colorbar.set_ylabel("$f$")
    axis_colorbar.yaxis.set_label_coords(1.9, 0.5)

    figure.text(
        0.999,  # x
        0.01,  # y
        r"$\textrm{/ GeV}$",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
    )

    plot_lib.save_fig(figure, out_path)


def plot_combo_z(out_path, data, phi, cmap):
    labels_y = ["BY", "CY"]
    labels_x = ["AZ", "BZ", "CZ"]

    label_lim = {
        "BY": (-800, 800),
        "CY": (-800, 800),
        "AZ": (0, 4400),
        "BZ": (-2200, 2200),
        "CZ": (-2200, 2200),
    }.get

    label_ticks = {
        "BY": (-500, 0, 500),
        "CY": (-500, 0, 500),
        "AZ": (0, 2000),
        "BZ": (-1000, 0, 1000),
        "CZ": (-1000, 0, 1000),
    }.get

    label_tex = {
        "BY": r"$\vec q^{\,j_2}_y$",
        "CY": r"$\vec q^{\,j_3}_y$",
        "AZ": r"$\vec q^{\,j_1}_z$",
        "BZ": r"$\vec q^{\,j_2}_z$",
        "CZ": r"$\vec q^{\,j_3}_z$",
    }.get

    norm = matplotlib.colors.CenteredNorm(0, 1, clip=True)
    colors = cmap(norm(phi))

    colorbar_width = 0.073
    colorbar_bot = 0.23

    figure, axes = pyplot.subplots(
        len(labels_y),
        len(labels_x),
        figsize=(4.8, 2.0),
        dpi=400,
        gridspec_kw={
            "top": 0.99,
            "right": 0.99 - colorbar_width,
            "bottom": 0.22,
            "left": 0.135,
            "hspace": 0.1,
            "wspace": 0.1,
        },
    )

    for i, laby in enumerate(labels_y):
        for j, labx in enumerate(labels_x):
            axis = axes[i, j]
            axis.scatter(
                data[:, label_index(labx)],
                data[:, label_index(laby)],
                lw=0,
                c=colors,
                marker=",",
                s=0.2,
            )

            axis.set_aspect(1.6)

            xmin, xmax = label_lim(labx)
            assert xmax - xmin == 4400
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(*label_lim(laby))

            xticks = label_ticks(labx)
            axis.set_xticks(xticks)

            yticks = label_ticks(laby)
            axis.set_yticks(yticks)

            if i == len(labels_y) - 1:
                axis.set_xlabel(label_tex(labx))
                axis.set_xticklabels(label_ends(xticks))
            else:
                axis.set_xticklabels([])

            if j == 0:
                axis.set_ylabel(label_tex(laby))
                axis.set_yticklabels(label_ends(yticks))
            else:
                axis.set_yticklabels([])

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    axis_colorbar = figure.add_axes(
        [1 - colorbar_width, colorbar_bot, 0.018, 0.975 - colorbar_bot]
    )

    mappable = matplotlib.cm.ScalarMappable(norm, cmap)
    figure.colorbar(mappable, cax=axis_colorbar)
    axis_colorbar.plot(
        [0.5, 0.5],
        [-1, 1],
        color="k",
        linewidth=1,
        linestyle="--",
        zorder=-0.5,
    )
    axis_colorbar.set_yticks([-1, 1])
    axis_colorbar.set_ylabel("$f$")
    axis_colorbar.yaxis.set_label_coords(1.9, 0.5)

    figure.text(
        0.999,  # x
        0.01,  # y
        r"$\textrm{/ GeV}$",
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
    )

    plot_lib.save_fig(figure, out_path)


# utilities


def label_ends(ticks):
    labels = []
    labels.append(r"$%r$" % ticks[0])
    labels += [""] * (len(ticks) - 2)
    labels.append(r"$%r$" % ticks[-1])
    return labels


def rgba_to_hue(rgba):
    rgb = rgba[:3]
    return matplotlib.colors.rgb_to_hsv(rgb)[0]


def make_cmap_2col(rgb_lo, rgb_hi, *, max_alpha=0.8):
    hsv_lo = matplotlib.colors.rgb_to_hsv(rgb_lo[:3])
    hsv_hi = matplotlib.colors.rgb_to_hsv(rgb_hi[:3])

    def _cmap(x):
        if x > 0.5:
            hsv = hsv_hi
        else:
            hsv = hsv_lo

        alpha = max_alpha * (2 * abs(x - 0.5)) ** 2

        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (*rgb, alpha)

    return matplotlib.colors.ListedColormap(
        [_cmap(x) for x in numpy.linspace(0, 1, 511)]
    )


if __name__ == "__main__":
    main()
