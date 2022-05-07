"""
Dump quality vs lambda plots.


Usage:

python plot/plot_quality.py

"""
import numpy
import plot_lib
from plot_lib import plot_qualities


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
        "NN\hphantom{C} truth-jet": (
            "results/jet_net_truth_private_test.csv",
            "o",
            cmap(0),
        ),
        "NN\hphantom{C} reco-jet": (
            "results/jet_net_reco_private_test.csv",
            "s",
            cmap(0.25),
        ),
        "BDT truth-jet": ("results/jet_bdt_truth_private_test.csv", "^", cmap(1)),
        "BDT reco-jet": ("results/jet_bdt_reco_private_test.csv", "d", cmap(0.75)),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_jet.png")


def plot_quality_truth_net():
    cmap = plot_lib.cmap_purple_orange

    ml = "net"
    label = "NN\hphantom{C}"

    label_to_spec = {
        label
        + " truth-jet": ("results/jet_%s_truth_private_test.csv" % ml, "o", cmap(0)),
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
        "NN\hphantom{C} truth-jet": (
            "results/jet_net_truth_private_test.csv",
            "o",
            cmap(0),
        ),
        "NN\hphantom{C} truth-jet + flavour": (
            "results/jet_net_truth_flav_private_test.csv",
            "s",
            cmap(0.25),
        ),
        "BDT truth-jet": ("results/jet_bdt_truth_private_test.csv", "^", cmap(1)),
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
        "CNN truth-jet": ("results/jet_cnn_truth_private_test.csv", "p", cmap(1)),
        "CNN reco-jet": ("results/jet_cnn_reco_private_test.csv", "<", cmap(0.5)),
        "CNN calo": ("results/tower_cnn_reco_private_test.csv", ">", cmap(0)),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_images.png")


def plot_quality_both():
    cmap_jet = plot_lib.cmap_purple_orange
    cmap_image = plot_lib.cmap_greens

    label_to_spec = {
        "NN\hphantom{C} truth-jet": (
            "results/jet_net_truth_private_test.csv",
            "o",
            cmap_jet(0),
        ),
        "NN\hphantom{C} reco-jet": (
            "results/jet_net_reco_private_test.csv",
            "s",
            cmap_jet(0.25),
        ),
        "BDT truth-jet": ("results/jet_bdt_truth_private_test.csv", "^", cmap_jet(1)),
        "BDT reco-jet": ("results/jet_bdt_reco_private_test.csv", "d", cmap_jet(0.75)),
        "CNN truth-jet": ("results/jet_cnn_truth_private_test.csv", "p", cmap_image(1)),
        "CNN reco-jet": ("results/jet_cnn_reco_private_test.csv", "<", cmap_image(0.5)),
        "CNN calo": ("results/tower_cnn_reco_private_test.csv", ">", cmap_image(0)),
    }

    figure, axis = plot_qualities(label_to_spec, select_pi)
    plot_lib.save_fig(figure, "quality_both.png")


# selectors


def select_pi(lambdas):
    keep = set(i / 10 for i in range(11))
    return numpy.array([li in keep for li in lambdas])


def select_lo(lambdas):
    return lambdas <= 0.2


if __name__ == "__main__":
    main()
