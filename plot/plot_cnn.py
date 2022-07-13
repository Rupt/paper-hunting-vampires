"""
Dump plots of parity variables (including alpha!)


Usage:

python plot/plot_cnn.py

"""
import json

import hist
import numpy
import plot_lib
from plot_lib import dual_subplots, hist_err, hist_ratio, save_fig
import matplotlib
import os

def main():
    do_linear = False
    plot_lib.set_default_context()

    # n.b. need different xlim for different lambda, currently set for lambda = 1
    # for label in ("0", "p2", "p4", "p6", "p8", "1"):
    for label in ("1"):

        model = "liv_{}".format(label)

        print(model)
        plot_parity_alpha(model, "reco_alpha")
        plot_parity_reco_net(model, "tower_cnn_parity", xlim=(0, 1.5), title=r"Energy Deposits")
        plot_parity_reco_net(model, "jets_cnn_parity", xlim=(0, 1.05), title=r"Reco-jet $p_T$")
        plot_parity_reco_net(model, "truth_cnn_parity", xlim=(0, 1.05), title=r"Truth-jet $p_T$")
        plot_parity_reco_net(model, "truth_rot_cnn_parity", xlim=(0, 1.05), title=r"Truth-jet $p_T$ rotated")
        plot_parity_reco_net(model, "reco_net_parity", xlim=(0, 1.05))
        # if do_linear:
        #     plot_parity_reco_net(model, "test_hist", linear=True)


def plot_parity_reco_net(label, histname, *, xlim=(0, 1), ratiolim = (0.9,2), linear=False, title = "" ):
    cmap = plot_lib.cmap_purple_orange

    x, edges = load_hists(label, histname, True)
    scale = get_scale()

    #####

    figure, (axis_hi, axis_lo) = dual_subplots()

    yield_ = x * scale
    err = (x**0.5) * scale

    poly_hist_real, _ = hist_err(
        axis_hi,
        yield_,
        err,
        edges,
        # normed=True,
        color=cmap(0),
    )

    poly_hist_fake, _ = hist_err(
        axis_hi,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
        color=cmap(1),
        # normed=True,
    )

    hist_ratio(
        axis_lo,
        (yield_, edges),
        (numpy.flip(yield_), edges),
        normed=False,
        color=cmap(0),
    )

    axis_hi.legend(
        [poly_hist_real, poly_hist_fake],
        [r"$f > 0$", r"$f < 0$"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    axis_hi.set_ylim(1, 10**5)


    axis_hi.set_ylabel(r"$\mathrm{events}$")
    # axis_hi.set_ylabel(r"$\mathrm{normalised}$"))
    axis_hi.minorticks_on()

    if not linear:
        axis_hi.set_yscale("log")

    axis_lo.plot(
        [xlim[0], xlim[1]],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
    )

    ####
    locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1,2,3,4,5))
    locmaj = matplotlib.ticker.LogLocator(base=10.0, numticks=7)
    axis_hi.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)

    axis_hi.yaxis.set_minor_locator(locmin)
    axis_hi.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis_hi.set_title(title)

    axis_lo.set_ylabel(r"$\mathrm{ratio}$")
    axis_lo.set_xlabel(r"$|f|$")
    axis_lo.set_xlim(*xlim)
    axis_lo.set_ylim(*ratiolim)
    axis_lo.minorticks_on()

    outname = "hist_%s_%s" % (histname, label)
    if linear:
        outname += "_linear"
    outname += ".png"
    fullpath = os.path.join("plots", outname)
    figure.savefig(
        fullpath,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
    )

def plot_parity_alpha(label, histname, *, linear=False):
    cmap = plot_lib.cmap_purple_orange

    x, edges = load_hists(label, histname, False)
    scale = get_scale()
    #####

    figure, (axis_hi, axis_lo) = dual_subplots()

    yield_ = x * scale
    err = (x**0.5) * scale

    poly_hist_real, _ = hist_err(
        axis_hi,
        yield_,
        err,
        edges,
        # normed=True,
        color=cmap(0),
    )

    poly_hist_fake, _ = hist_err(
        axis_hi,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
        color=cmap(1),
        # normed=True,
    )

    hist_ratio(
        axis_lo,
        (yield_, edges),
        (numpy.flip(yield_), edges),
        normed=False,
        color=cmap(0),
    )

    axis_hi.legend(
        [poly_hist_real, poly_hist_fake],
        [r"$\alpha > 0$", r"$\alpha < 0$"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    axis_hi.set_ylim(10, 10**4)

    axis_hi.set_ylabel(r"$\mathrm{events}$")
    axis_hi.minorticks_on()

    if not linear:
        axis_hi.set_yscale("log")

    xlim = [0,1.6]
    axis_lo.plot(
        [xlim[0], xlim[1]],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
    )
    axis_lo.set_ylabel(r"$\mathrm{ratio}$")
    axis_lo.set_xlabel(r"$\alpha$")
    axis_lo.set_xlim(*xlim)
    axis_lo.set_ylim(0.9,1.1)
    axis_lo.minorticks_on()

    outname = "hist_%s_%s" % (histname, label)
    if linear:
        outname += "_linear"
    outname += ".png"

    fullpath = os.path.join("plots", outname)
    figure.savefig(
        fullpath,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
    )

# utility


def load_hists(label, histname, rebin):
    nbins = 40

    hist_dict = json.load(open("results/hist/%s/%s.json" % (label, histname)))
    if rebin: hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
    x, edges = hist.arrays(hist_dict)

    return x, edges

def get_scale():

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

    scale = frac_test * lumi_ifb * xs_fb / nhist

    return scale


if __name__ == "__main__":
    main()
