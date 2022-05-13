"""
Dump quality vs lambda plots.


Usage:

python plot/plot_quality.py

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
    plot_lib.save_fig(figure, "quality_jet.png")


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
    plot_lib.save_fig(figure, "quality_truth_%s.png" % ml)


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
    plot_lib.save_fig(figure, "quality_flavour.png")


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
    plot_lib.save_fig(figure, "quality_images.png")


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
    plot_lib.save_fig(figure, "quality_both.png")


def plot_qualities(
    label_to_spec, select_func=None, *, sigma=1, xlim=(-0.05, 1.05), space=0.1
):
    figure, axis = pyplot.subplots(
        figsize=(4.8, 2.4),
        dpi=400,
        gridspec_kw={
            "top": 0.99,
            "right": 0.995,
            "bottom": 0.18,
            "left": 0.135,
        },
    )

    axis.plot([-0.05, 1.05], [0, 0], "k--", lw=1)

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

    labels = [
        r"$\textrm{%s}$" % label.replace(" ", "~") for label in label_to_spec
    ]
    axis.legend(
        errorbars,
        labels,
        frameon=False,
        loc="upper left",
        borderpad=0,
    )

    axis.set_xlim(*xlim)

    axis.set_xlabel(r"$\lambda_\mathrm{PV}$")
    axis.set_ylabel(r"$Q \pm %r\sigma$" % sigma)

    return figure, axis


# selectors


def select_pi(lambdas):
    keep = set(i / 10 for i in range(11))
    return numpy.array([li in keep for li in lambdas])


def select_lo(lambdas):
    return lambdas <= 0.2


if __name__ == "__main__":
    main()
