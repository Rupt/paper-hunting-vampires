"""Dump H_T histogram plots.

Usage:

python plot/plot_ht.py

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

    for suffix in ("", "_3j", "_4j"):
        figure, (axis_hi, axis_lo) = plot_ht(load_hists("truth", suffix))
        save_fig(figure, "hist_ht%s_truth.png" % suffix)

    for suffix in ("", "_3j", "_4j"):
        figure, (axis_hi, axis_lo) = plot_ht(load_hists("reco", suffix))
        save_fig(figure, "hist_ht%s_reco.png" % suffix)


def plot_ht(lambda_to_x_and_edges):
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

    xmin = 500
    xmax = 8_000

    axis_lo.plot(
        [xmin, xmax],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
    )

    axis_hi.set_ylabel(r"$\mathrm{normalized}~/~\mathrm{GeV}$")
    axis_hi.set_yscale("log")

    axis_lo.set_ylabel(r"$\mathrm{ratio~to~SM}$")
    axis_lo.set_xlabel(r"$\mathrm{H_T}~/~\mathrm{GeV}$")
    axis_lo.set_xlim(xmin, xmax)
    axis_lo.set_ylim(0, 7)

    return figure, (axis_hi, axis_lo)


def load_hists(tag, suffix=""):
    nbins = 40

    binning = (
        list((i,) for i in range(18))
        + list((i, i + 1) for i in range(18, 24, 2))
        + [range(24, nbins)]
    )

    out = {}
    for label in ("0", "p2", "p4", "p6", "p8", "1"):
        lambda_ = float(label.replace("p", "."))
        hist_dict = json.load(
            open("results/hist/liv_%s/%s_ht%s.json" % (label, tag, suffix))
        )
        hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
        x_and_edges = hist.arrays(hist_dict)
        x_and_edges = hist.combine_array_bins(x_and_edges, binning)
        # reduce upper edge with 0 events
        x_and_edges[1][-1] = 8_000
        # remove two leading empty bins
        x, edges = x_and_edges
        x = x[2:]
        edges = edges[2:]
        out[lambda_] = x, edges

    return out


if __name__ == "__main__":
    main()
