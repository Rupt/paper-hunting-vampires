"""
Dump p_T histogram plots.

Usage:

python plot/plot_pt.py

"""
import json

import hist
import plot_lib
from plot_lib import (
    cmap_purple_orange,
    dual_subplots,
    hist_ratio,
    hist_sqrterr,
    save_fig,
)


def main():
    plot_lib.set_default_context()

    for jet in "abcd":
        figure, (axis_hi, axis_lo) = plot_pt(load_hists("truth", jet))
        save_fig(figure, "hist_pt_%s_truth.png" % jet)

    for jet in "abcd":
        figure, (axis_hi, axis_lo) = plot_pt(load_hists("reco", jet))
        save_fig(figure, "hist_pt_%s_reco.png" % jet)


def plot_pt(lambda_to_x_and_edges):
    sm_arrays = lambda_to_x_and_edges[0.0]

    figure, (axis_hi, axis_lo) = dual_subplots()

    poly_hist_sm, _ = hist_sqrterr(
        axis_hi, sm_arrays, normed=True, color=cmap_purple_orange(0)
    )

    liv_lambdas = []
    liv_polys = []
    for lambda_i, arrays_i in lambda_to_x_and_edges.items():
        if lambda_i == 0:
            continue

        color = cmap_purple_orange(lambda_i)
        liv_lambdas.append(lambda_i)

        poly_hist_i, _ = hist_sqrterr(
            axis_hi,
            arrays_i,
            normed=True,
            color=color,
        )
        liv_polys.append(poly_hist_i)

        hist_ratio(
            axis_lo,
            arrays_i,
            sm_arrays,
            normed=True,
            color=poly_hist_i.get_edgecolor(),
        )

    axis_hi.legend(
        [poly_hist_sm, liv_polys[-1]],
        [
            r"$\mathrm{SM}$",
            r"$\lambda_\mathrm{PV} = %g$" % liv_lambdas[-1],
        ],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    xmin = 200
    xmax = 4_000

    axis_lo.plot(
        [xmin, xmax],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
    )

    axis_hi.set_ylabel(r"$\textrm{normalized}~/~\mathrm{Ge\kern-0.15ex V}$")
    axis_hi.set_yscale("log")

    axis_lo.set_ylabel(r"$\textrm{ratio~to~SM}$")
    axis_lo.set_xlabel(r"$p_\mathrm{T}~/~\mathrm{Ge\kern-0.15ex V}$")
    axis_lo.set_xlim(xmin, xmax)
    axis_lo.set_ylim(0, 7)

    return figure, (axis_hi, axis_lo)


def load_hists(tag, suffix):
    nbins = 50

    if suffix == "a":
        binning = (
            list((i,) for i in range(24))
            + list((i, i + 1) for i in range(24, 30, 2))
            + [range(30, 40), range(40, nbins)]
        )
    elif suffix == "b":
        binning = (
            list((i,) for i in range(20))
            + list((i, i + 1) for i in range(20, 24, 2))
            + [range(24, 40), range(40, nbins)]
        )
    elif suffix == "c":
        binning = (
            list((i,) for i in range(11))
            + list((i, i + 1) for i in range(11, 13, 2))
            + [range(13, 23), range(23, 40), range(40, nbins)]
        )
    else:
        assert suffix == "d"
        binning = list((i,) for i in range(8)) + [
            range(8, 18),
            range(18, 40),
            range(40, nbins),
        ]

    out = {}
    for label in ("0", "p2", "p4", "p6", "p8", "1"):
        lambda_ = float(label.replace("p", "."))
        hist_dict = json.load(
            open("results/hist/liv_%s/%s_pt_%s.json" % (label, tag, suffix))
        )
        hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
        x_and_edges = hist.arrays(hist_dict)
        x_and_edges = hist.combine_array_bins(x_and_edges, binning)
        # remove two leading empty bins and 4000+
        x, edges = x_and_edges
        x = x[2:-1]
        edges = edges[2:-1]
        out[lambda_] = x, edges

    return out


if __name__ == "__main__":
    main()
