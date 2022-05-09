"""
Dump plots of parity variables (including alpha!) quantile transformed.


Usage:

python plot/plot_parity_transformed.py

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
        "reco_alpha_transformed",
        "reco_net_parity_transformed",
        scale=scale,
        lumi_ifb=lumi_ifb,
        linear=True,
    )


def plot_both(alphaname, netname, *, scale, lumi_ifb, linear=False):

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
            "hspace": 0.02,
        },
        sharex=True,
        sharey=True,
    )

    # axis1: alpha
    x, edges = load_hists(label, alphaname)
    yield_ = x * scale
    err = (x * scale) ** 0.5

    lw = 1.5

    poly_hist_real, _ = hist_err(
        axis1,
        yield_,
        err,
        edges,
        color=cmap(0),
        lw=lw,
    )

    poly_hist_fake, _ = hist_err(
        axis1,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
        color=cmap(1),
        lw=lw,
    )

    axis1.legend(
        [poly_hist_real, poly_hist_fake],
        [r"$\alpha > 0$", r"$\alpha < 0$"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    axis1.set_xlabel(r"$F(|\alpha|)$")
    axis1.set_ylabel(r"$\mathrm{events}$")

    # axis2: net phi
    x, edges = load_hists(label, netname)
    yield_ = x * scale
    err = (x * scale) ** 0.5

    poly_hist_real, _ = hist_err(
        axis2,
        yield_,
        err,
        edges,
        color=cmap(0),
        lw=lw,
    )

    poly_hist_fake, _ = hist_err(
        axis2,
        numpy.flip(yield_),
        numpy.flip(err),
        edges,
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
    axis2.set_ylim(0, 3800)
    axis2.set_xticks([0, 0.5, 1])

    if not linear:
        axis2.set_yscale("log")

    # text
    axis1.text(
        0.05,
        0.20,
        r"$\alpha="
        r"\arcsin\!\left("
        r"\frac{"
        r"\vec p^{\,j_1} \times \vec p^{\,j_2}"
        r"}{"
        r"|\vec p^{\,j_1} \times \vec p^{\,j_2}|"
        r"}"
        r"\cdot"
        r"\frac{"
        r"\vec p^{\,j_3}"
        r"}{"
        r"|\vec p^{\,j_3}|"
        r"}"
        r"\right)$",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axis1.transAxes,
        fontsize=9,
    )

    axis2.text(
        0.05,
        0.22,
        r"$f(x) = g(x) - g(\mathrm{P}x)$",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=axis2.transAxes,
        fontsize=9,
    )

    for ax in (axis1, axis2):
        ax.text(
            0.02,
            0.98,
            r"$\textrm{PV-mSME}~~\lambda_\textrm{PV} = 1$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.text(
            0.02,
            0.09,
            r"$p\textrm{--}p~~\sqrt{s}=13~\mathrm{TeV}~~%r~\mathrm{fb}^{-1}$"
            % lumi_ifb,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.text(
            0.02,
            0.02,
            r"$3~\textrm{jets}~~p_\mathrm{T} > 220~\mathrm{GeV}~~|\eta|<2.8$",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=6,
        )

    outname = "hist_both_alpha_net_transformed"
    if linear:
        outname += "_linear"
    outname += ".png"
    save_fig(figure, outname)


# utility


def load_hists(label, histname):
    nbins = 20

    hist_dict = json.load(
        open("results/hist/liv_%s/%s.json" % (label, histname))
    )
    hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
    x, edges = hist.arrays(hist_dict)

    return x, edges


if __name__ == "__main__":
    main()
