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


def main():
    do_linear = False
    plot_lib.set_default_context()

    # n.b. need different xlim for different lambda, currently set for lambda = 1
    for label in ("0", "p2", "p4", "p6", "p8", "1"):

        model = "liv_{}".format(label)

        print(model)
        # plot_parity_reco_net(model, "tower_cnn_parity", xlim=(0, 1.5))
        # plot_parity_reco_net(model, "jets_cnn_parity", xlim=(0, 1.05))
        # plot_parity_reco_net(model, "truth_cnn_parity", xlim=(0, 1.05))
        # plot_parity_reco_net(model, "truth_rot_cnn_parity", xlim=(0, 1.05))
        plot_parity_reco_net(model, "reco_net_parity", xlim=(0, 1.05))
        if do_linear:
            plot_parity_reco_net(model, "test_hist", linear=True)


def plot_parity_reco_net(label, histname, *, xlim=(0, 1), linear=False):
    cmap = plot_lib.cmap_purple_orange

    x, edges = load_hists(label, histname)

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
    #####

    figure, (axis_hi, axis_lo) = dual_subplots()

    yield_ = x * scale
    err = (x**0.5) * scale

    poly_hist_real, _ = hist_err(
        axis_hi,
        yield_,
        err,
        edges,
        normed=True,
        color=cmap(0),
    )

    poly_hist_fake, _ = hist_err(
        axis_hi,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
        color=cmap(1),
        normed=True,
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

    # axis_hi.set_ylim(1,max(yield_) *10 )
    axis_hi.set_ylim(5e-7, 5e1)

    # axis_hi.set_ylabel(r"$\mathrm{events}$"))
    axis_hi.set_ylabel(r"$\mathrm{normalized}$")
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

    axis_lo.set_ylabel(r"$\mathrm{ratio}$")
    axis_lo.set_xlabel(r"$|f|$")

    axis_lo.set_xlim(*xlim)
    axis_lo.set_ylim(0, 5.0)
    axis_lo.minorticks_on()

    outname = "hist_%s_%s" % (histname, label)
    if linear:
        outname += "_linear"
    outname += ".png"
    save_fig(figure, outname)


# utility


def load_hists(label, histname):
    nbins = 40

    hist_dict = json.load(open("results/hist/%s/%s.json" % (label, histname)))
    hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
    x, edges = hist.arrays(hist_dict)

    return x, edges


if __name__ == "__main__":
    main()
