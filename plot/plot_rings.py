"""
Produce rings plots.

Usage:

python plot/plot_rings.py

"""
import json

import numpy
import plot_lib
from matplotlib import pyplot

COLOR_ORIGINAL = plot_lib.cmap_purple_orange(0)
COLOR_PARITY = plot_lib.cmap_purple_orange(1)


PROCESS_TO_LABEL = {
    "sm": r"$\mathrm{Standard}$" + "\n" + r"$\mathrm{Model}$",
    "pv_msme": r"$\textrm{PV-mSME}$",
    "pv_msme_c03_c30_1_1": r"$c_{03}=1$" + "\n" + r"$c_{30}=1$",
    "pv_msme_c03_c30_m2_2": r"$c_{03}=-2$" + "\n" + r"$c_{30}=\hphantom{-}2$",
    "pv_msme_c13_1": r"$c_{13}=1$",
    "pv_msme_diag_100m1": r"$c_{00}=\hphantom{-}1$" + "\n" + r"$c_{33}=-1$",
}

PDG_TO_TEX = {
    1: r"\mathrm{d}",
    -1: r"\bar{\mathrm{d}}",
    2: r"\mathrm{u}",
    -2: r"\bar{\mathrm{u}}",
    21: r"\mathrm{g}",
}


def main():
    plot_lib.set_default_context()

    events = [
        "liv_1_0",
        "liv_1_110",
    ]

    processes = [
        "sm",
        "pv_msme",
        "pv_msme_c03_c30_1_1",
        "pv_msme_c03_c30_m2_2",
        "pv_msme_diag_100m1",
        "pv_msme_c13_1",
    ]

    event_to_process_to_tree = {}

    # load serialized data
    for event in events:
        process_to_tree = {}
        for process in processes:
            name = event + "_" + process
            filename = "results/rings/rings_%s.json" % name
            process_to_tree[process] = json.load(open(filename))

        # ensure assumptions: same eveent by "event" and more.
        trees = iter(process_to_tree.values())
        first = next(trees)

        flavours = first["flavors"]
        momenta = first["momenta"]
        helicities = first["helicities"]

        assert first["helicity_sum"]

        for tree in trees:
            assert tree["flavors"] == flavours
            assert tree["momenta"] == momenta
            assert tree["helicities"] == helicities

        event_to_process_to_tree[event] = process_to_tree

    # relative variations?
    print("sm")
    check_variation(event_to_process_to_tree["liv_1_0"]["sm"])
    plot_variation(
        event_to_process_to_tree["liv_1_0"]["pv_msme"],
        PROCESS_TO_LABEL["pv_msme"],
    )
    print("pv-msme")
    check_variation(event_to_process_to_tree["liv_1_0"]["pv_msme"])
    plot_variation(
        event_to_process_to_tree["liv_1_0"]["sm"],
        PROCESS_TO_LABEL["sm"].replace("\n", " "),
    )

    # plot
    kw_top = 0.83
    kw_right = 0.999
    kw_bottom = 0.1
    kw_left = 0.13
    kw_hspace = -0.1
    kw_wspace = 0.2

    figure, axes = pyplot.subplots(
        len(events),  # rows
        len(processes),  # columns
        figsize=(4.8, 2.0),
        dpi=400,
        gridspec_kw={
            "top": kw_top,
            "right": kw_right,
            "bottom": kw_bottom,
            "left": kw_left,
            "hspace": kw_hspace,
            "wspace": kw_wspace,
        },
    )

    # top left is (0, 0), bottom right is (1, 5)
    for i, event in enumerate(events):
        for j, process in enumerate(processes):
            axis = axes[i][j]
            tree = event_to_process_to_tree[event][process]
            lines = plot_one(axis, tree)

    for i, event in enumerate(events):
        mid = kw_top - kw_bottom

        tree = event_to_process_to_tree[event][processes[0]]
        flavours = tree["flavors"]

        figure.text(
            0.001,  # x
            kw_top - mid * (i + 0.5) / len(events),  # y
            flavours_str(flavours),
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=9,
        )

    for i, process in enumerate(processes):
        mid = kw_right - kw_left

        x = kw_left + mid * (i + 0.5) / len(processes)
        # special case for ugliness
        if process == "sm":
            x -= 0.01

        figure.text(
            x,  # x
            0.85,  # y
            PROCESS_TO_LABEL[process],
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
        )

    figure.legend(
        lines,
        [
            r"$S(\phi, +)$",
            r"$S(\phi, -)$",
            r"$\mathrm{P} S(\phi, +)$",
            r"$\mathrm{P} S(\phi, -)$",
        ],
        frameon=False,
        ncol=4,
        fontsize=9,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        borderpad=0.0,
        handlelength=1.5,
    )

    plot_lib.save_fig(figure, "rings_combo.png")


def plot_one(axis, tree):

    ring = numpy.array(tree["ring"])
    ring_flip = numpy.array(tree["ring_flip"])
    pring = numpy.array(tree["pring"])
    pring_flip = numpy.array(tree["pring_flip"])

    ngrid = tree["ngrid"]
    assert (
        len(ring) == len(ring_flip) == len(pring) == len(pring_flip) == ngrid
    )

    thetas = numpy.linspace(0, 2 * numpy.pi, len(ring))

    lims = []

    def xy(r):
        x, y = xy_from(thetas, r)
        lims.append(max(abs(x).max(), abs(y).max()))
        return x, y

    axis.scatter(
        [0],
        [0],
        c="k",
        marker="o",
        s=2,
        lw=0,
    )

    lw = 0.7
    alpha = 1.0
    pring_obj = axis.plot(*xy(pring), color=COLOR_PARITY, lw=lw, alpha=alpha)
    pring_flip_obj = axis.plot(
        *xy(pring_flip), linestyle="--", color=COLOR_PARITY, lw=lw, alpha=alpha
    )

    ring_obj = axis.plot(*xy(ring), color=COLOR_ORIGINAL, lw=lw, alpha=alpha)
    ring_flip_obj = axis.plot(
        *xy(ring_flip),
        linestyle="--",
        color=COLOR_ORIGINAL,
        lw=lw,
        alpha=alpha,
    )

    axlim = max(max(lims), 1e-300) * 1.05
    axis.set_xlim(-axlim, axlim)
    axis.set_ylim(-axlim, axlim)

    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)

    return [ring_obj[0], ring_flip_obj[0], pring_obj[0], pring_flip_obj[0]]


def check_variation(tree):
    rx = tree["ring"] + tree["ring_flip"]
    prx = tree["pring"] + tree["pring_flip"]

    max_rx = max(rx)
    min_rx = min(rx)

    max_prx = max(prx)
    min_prx = min(prx)

    variation = 2 * (max_rx - min_rx) / (max_rx + min_rx)
    pvariation = 2 * (max_prx - min_prx) / (max_prx + min_prx)

    print(f"{variation = }")
    print(f"{pvariation = }")


def plot_variation(tree, label=""):
    figure, axis = pyplot.subplots(
        figsize=(4.8, 2.4),
        dpi=400,
        gridspec_kw={
            "top": 0.92,
            "right": 0.98,
            "bottom": 0.18,
            "left": 0.1,
        },
    )

    ring = numpy.array(tree["ring"])
    ring_flip = numpy.array(tree["ring_flip"])
    pring = numpy.array(tree["pring"])
    pring_flip = numpy.array(tree["pring_flip"])

    ngrid = tree["ngrid"]
    assert (
        len(ring) == len(ring_flip) == len(pring) == len(pring_flip) == ngrid
    )

    theta = numpy.linspace(0, 2 * numpy.pi, len(ring))

    def variation(x1, x2):
        central = numpy.mean([x1, x2])
        return x1 / central - 1, x2 / central - 1

    y, yflip = variation(ring, ring_flip)
    py, pyflip = variation(pring, pring_flip)

    lw = 0.7
    alpha = 1.0

    axis.plot(theta, py, color=COLOR_PARITY, lw=lw, alpha=alpha)
    axis.plot(
        theta, pyflip, linestyle="--", color=COLOR_PARITY, lw=lw, alpha=alpha
    )

    axis.plot(theta, y, color=COLOR_ORIGINAL, lw=lw, alpha=alpha)
    axis.plot(
        theta, yflip, linestyle="--", color=COLOR_ORIGINAL, lw=lw, alpha=alpha
    )

    scale = numpy.max(numpy.abs([y, yflip, py, pyflip]))
    axis.set_ylim(1.05 * scale, -1.05 * scale)
    axis.set_xlim(0, 2 * numpy.pi)
    axis.set_xticks([0, numpy.pi, 2 * numpy.pi])
    axis.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])

    axis.set_xlabel(r"$\phi$")
    axis.set_ylabel(r"$f(\phi) / \langle f \rangle - 1$")

    figure.text(
        0.995,  # x
        0.99,  # y
        label,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
    )

    plot_lib.save_fig(figure, "rings_tinyerror_%s.png" % tree["name"])


def flavours_str(flavours):
    tex = PDG_TO_TEX.get
    r = r"$%s %s \rightarrow" % (tex(flavours[0]), tex(flavours[1]))
    for n in flavours[2:]:
        r += r" %s" % tex(n)
    r += r"$"
    return r


# ripped from rings_lib
def xy_from(theta, radius):
    """Return cartesian versions of polar coordinates."""
    return numpy.array(
        [
            numpy.cos(theta) * radius,
            numpy.sin(theta) * radius,
        ]
    )


if __name__ == "__main__":
    main()
