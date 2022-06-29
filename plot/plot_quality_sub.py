"""
Dump quality vs lambda plots with ratio subplots.


Usage:

python plot/plot_quality_sub.py

"""
import numpy
import plot_lib
from matplotlib import pyplot


def main():
    plot_lib.set_default_context()

    plot_quality_jets()
    plot_quality_truth_net()
    plot_quality_flavour()
    plot_quality_images()
    plot_quality_both()


def plot_quality_jets():
    cmap = plot_lib.cmap_purple_orange

    label_to_spec = {
        r"NN\hphantom{C} truth-jet": (
            "results/jet_net_truth_private_test.csv",
            "o",
            cmap(0),
        ),
        r"NN\hphantom{C} reco-jet": (
            "results/jet_net_reco_private_test.csv",
            "s",
            cmap(0.25),
        ),
        "BDT truth-jet": (
            "results/jet_bdt_truth_private_test.csv",
            "^",
            cmap(1),
        ),
        "BDT reco-jet": (
            "results/jet_bdt_reco_private_test.csv",
            "d",
            cmap(0.75),
        ),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_sub_jet.png")


def plot_quality_truth_net():
    cmap = plot_lib.cmap_purple_orange

    ml = "net"
    label = r"NN\hphantom{C}"

    label_to_spec = {
        label
        + " truth-jet": (
            "results/jet_%s_truth_private_test.csv" % ml,
            "o",
            cmap(0),
        ),
        label
        + " truth-jet + flavour": (
            "results/jet_%s_truth_flav_private_test.csv" % ml,
            "s",
            cmap(0.25),
        ),
        label
        + " truth-jet + helicity": (
            "results/jet_%s_truth_hel_private_test.csv" % ml,
            "^",
            cmap(1),
        ),
        label
        + " truth-jet + flavour + helicity": (
            "results/jet_%s_truth_flav_hel_private_test.csv" % ml,
            "d",
            cmap(0.75),
        ),
    }

    figure, axis = plot_qualities(
        label_to_spec, select_lo, xlim=(-0.01, 0.21), space=0.01
    )
    plot_lib.save_fig(figure, "quality_sub_truth_%s.png" % ml)


def plot_quality_flavour():
    cmap = plot_lib.cmap_purple_orange

    label_to_spec = {
        r"NN\hphantom{C} truth-jet": (
            "results/jet_net_truth_private_test.csv",
            "o",
            cmap(0),
        ),
        r"NN\hphantom{C} truth-jet + flavour": (
            "results/jet_net_truth_flav_private_test.csv",
            "s",
            cmap(0.25),
        ),
        "BDT truth-jet": (
            "results/jet_bdt_truth_private_test.csv",
            "^",
            cmap(1),
        ),
        "BDT truth-jet + flavour": (
            "results/jet_bdt_truth_flav_private_test.csv",
            "d",
            cmap(0.75),
        ),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_sub_flavour.png")


def plot_quality_images():
    cmap = plot_lib.cmap_greens

    label_to_spec = {
        "CNN truth-jet": (
            "results/jet_cnn_truth_private_test.csv",
            "p",
            cmap(1),
        ),
        "CNN reco-jet": (
            "results/jet_cnn_reco_private_test.csv",
            "<",
            cmap(0.5),
        ),
        "CNN calo-image": (
            "results/tower_cnn_reco_private_test.csv",
            ">",
            cmap(0),
        ),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_sub_images.png")


def plot_quality_both():
    cmap_jet = plot_lib.cmap_purple_orange
    cmap_image = plot_lib.cmap_greens

    label_to_spec = {
        r"NN\hphantom{C} truth-jet": (
            "results/jet_net_truth_private_test.csv",
            "o",
            cmap_jet(0),
        ),
        r"NN\hphantom{C} reco-jet": (
            "results/jet_net_reco_private_test.csv",
            "s",
            cmap_jet(0.25),
        ),
        "BDT truth-jet": (
            "results/jet_bdt_truth_private_test.csv",
            "^",
            cmap_jet(1),
        ),
        "BDT reco-jet": (
            "results/jet_bdt_reco_private_test.csv",
            "d",
            cmap_jet(0.75),
        ),
        "CNN truth-jet": (
            "results/jet_cnn_truth_private_test.csv",
            "p",
            cmap_image(1),
        ),
        "CNN reco-jet": (
            "results/jet_cnn_reco_private_test.csv",
            "<",
            cmap_image(0.5),
        ),
        "CNN calo-image": (
            "results/tower_cnn_reco_private_test.csv",
            ">",
            cmap_image(0),
        ),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_sub_both.png")


def plot_qualities(
    label_to_spec, select_func=None, *, sigma=1, xlim=(-0.05, 1.05), space=0.1
):
    figure, (axis, axis_lo) = pyplot.subplots(
        2,
        1,
        sharex=True,
        figsize=(4.8, 3.05),
        dpi=400,
        gridspec_kw={
            "height_ratios": [3, 1],
            "wspace": 0,
            "hspace": 0,
            "top": 0.99,
            "right": 0.995,
            "bottom": 0.14,
            "left": 0.135,
        },
    )

    axis.plot([-0.05, 1.05], [0, 0], "k--", lw=1)
    axis_lo.plot([-0.05, 1.05], [0, 0], "k--", lw=1)

    ymin_lo = -6
    ymax_lo = +6
    marker_buffer = (ymax_lo - ymin_lo) * 0.02
    arrow_length = (ymax_lo - ymin_lo) * 0.12

    nlabels = len(label_to_spec)
    errorbars = []
    for i, (label, spec) in enumerate(label_to_spec.items()):
        csvpath, marker, color = spec
        values = numpy.loadtxt(csvpath, skiprows=1, delimiter=",")
        lambdas, ntest, qualities, quality_stds = values.T

        if select_func is None:
            select = slice(None)
        else:
            select = select_func(lambdas)

        offset = 0.25 * space * (i / nlabels - 0.5)

        bar = axis.errorbar(
            lambdas[select] + offset,
            qualities[select] * 1e6,
            yerr=quality_stds[select] * 1e6 * sigma,
            color=color,
            marker=marker,
            markersize=4,
            markeredgewidth=1,
            markerfacecolor="w",
            linewidth=0,
            elinewidth=1,
        )
        errorbars.append(bar)

        # ratio subplot
        ratio = _safe_divide(qualities[select], quality_stds[select])
        axis_lo.errorbar(
            lambdas[select] + offset,
            ratio,
            color=color,
            marker=marker,
            markersize=4,
            markeredgewidth=1,
            markerfacecolor="w",
            linewidth=0,
        )

        # ratio subplot outer arrows
        edge_line_offset = (ymax_lo - ymin_lo) * 0.02
        head_width = (xlim[1] - xlim[0]) * 0.005

        above = ratio > (ymax_lo + marker_buffer)
        for x in (lambdas[select] + offset)[above]:
            axis_lo.arrow(
                x,
                ymax_lo - arrow_length - edge_line_offset,
                0,
                arrow_length,
                edgecolor=color,
                facecolor="w",
                width=0,
                head_width=head_width,
                head_length=0.4 * arrow_length,
                length_includes_head=True,
                linewidth=1,
                alpha=0.7,
            )

        below = ratio < (ymin_lo - marker_buffer)
        for x in (lambdas[select] + offset)[below]:
            axis_lo.arrow(
                x,
                ymin_lo + arrow_length + edge_line_offset,
                0,
                -arrow_length,
                edgecolor=color,
                facecolor="w",
                width=0,
                head_width=head_width,
                head_length=0.4 * arrow_length,
                length_includes_head=True,
                linewidth=1,
                alpha=0.7,
            )

    labels = [r"$\textrm{%s}$" % label.replace(" ", "~") for label in label_to_spec]
    axis.legend(
        errorbars,
        labels,
        frameon=False,
        loc="upper left",
        borderpad=0,
    )

    axis.set_xlim(*xlim)
    axis_lo.set_ylim(ymin_lo, ymax_lo)

    if sigma != 1:
        axis.set_ylabel(r"$Q \pm %r\sigma~~(\times10^6)$" % sigma)
    else:
        axis.set_ylabel(r"$Q \pm \sigma~~(\times10^6)$")
    axis_lo.set_ylabel(r"$Q / \sigma$")
    axis_lo.set_xlabel(r"$\lambda_\mathrm{PV}$")

    # align labels
    axis.yaxis.set_label_coords(-0.11, 0.5)
    axis_lo.yaxis.set_label_coords(-0.11, 0.5)

    return figure, axis


# selectors


def select_pi(lambdas):
    keep = set(i / 10 for i in range(11))
    return numpy.array([li in keep for li in lambdas])


def select_lo(lambdas):
    return lambdas <= 0.2


# utility


def _safe_divide(a, b):
    """Return a / b except:

    if a is 0 and b is a number, return 0
    """
    return a / (b + (a == 0))


if __name__ == "__main__":
    main()
