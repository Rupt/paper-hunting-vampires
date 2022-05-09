"""
Dump plots of parity variables.


Usage:

python plot/plot_parity.py

"""
import json

import hist
import numpy
import plot_lib
from matplotlib import pyplot
from plot_lib import hist_err, save_fig


def main():
    plot_lib.set_default_context()

    # Normalize to SM cross-section
    lumi_ifb = 1.0  # ifb
    # TODO? 139.0  # ifb

    # standard model xs
    lambda_, xss_fb = numpy.loadtxt(
        "results/reco_xs_fb.csv", skiprows=1, delimiter=","
    ).T
    xs_fb = xss_fb[lambda_.tolist().index(0.0)]

    # number of events put into the histogram (not all end up in bins)
    nhist = 2304276

    # train-val-test split
    frac_test = 0.2

    print(f"{lumi_ifb = }")
    print(f"{xs_fb = }")
    print(f"{frac_test = }")
    print(f"{nhist = }")

    scale = frac_test * lumi_ifb * xs_fb / nhist

    plot_both(
        scale=scale,
        lumi_ifb=lumi_ifb,
    )


def plot_both(*, scale, lumi_ifb):

    label = "1"  # only LIV_1 supported
    cmap = plot_lib.cmap_purple_orange

    figure, (axis1, axis2) = pyplot.subplots(
        1,
        2,
        figsize=(4.8, 2.4),
        dpi=400,
        gridspec_kw={
            "top": 0.99,
            "right": 0.975,
            "bottom": 0.18,
            "left": 0.115,
            "hspace": 0.9,
        },
        sharex=False,
        sharey=False,
    )

    # axis1: net f raw
    x, edges = load_hist_original(label)
    yield_ = x * scale
    err = (x ** 0.5) * scale

    lw = 1.5

    poly_hist_real, _ = hist_err(
        axis1,
        yield_,
        err,
        edges,
        normed=True,
        color=cmap(0),
        lw=lw,
    )

    poly_hist_fake, _ = hist_err(
        axis1,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
        normed=True,
        color=cmap(1),
        lw=lw,
    )

    axis1.legend(
        [poly_hist_real, poly_hist_fake],
        [r"$f > 0$", r"$f < 0$"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    axis1.set_xlabel(r"$|f|$")
    axis1.set_ylabel(r"$\mathrm{normalized}$")

    axis1.set_xlim(0, 1.05)
    axis1.set_ylim(5e-7, 5e1)
    axis1.set_yscale("log")

    # axis2: net f transformed
    x, edges = load_hist_transformed(label)
    yield_ = x * scale
    err = (x ** 0.5) * scale

    poly_hist_real, _ = hist_err(
        axis2,
        yield_,
        err,
        edges,
        normed=True,
        color=cmap(0),
        lw=lw,
    )

    poly_hist_fake, _ = hist_err(
        axis2,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
        normed=True,
        color=cmap(1),
        lw=lw,
    )

    axis2.legend(
        [poly_hist_real, poly_hist_fake],
        [r"$f > 0$", r"$f < 0$"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    axis2.set_xlabel(r"$F(|f|)$")

    axis2.set_xlim(0.0, 1.0)
    axis2.set_ylim(0, 0.9)

    axis2.set_xticks([0, 0.5, 1])

    outname = "hist_both_net"
    outname += ".png"
    save_fig(figure, outname)


# utility


def load_hist_transformed(label, histname="reco_net_parity_transformed"):
    nbins = 20

    hist_dict = json.load(
        open("results/hist/liv_%s/%s.json" % (label, histname))
    )
    hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
    x, edges = hist.arrays(hist_dict)

    return x, edges


def load_hist_original(label, histname="reco_net_parity"):
    nbins = 40

    hist_dict = json.load(
        open("results/hist/liv_%s/%s.json" % (label, histname))
    )
    hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
    x, edges = hist.arrays(hist_dict)

    return x, edges


if __name__ == "__main__":
    main()
