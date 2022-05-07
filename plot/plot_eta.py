"""Dump eta histogram plots.

Usage:

python plot/plot_eta.py

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
        figure, (axis_hi, axis_lo) = plot_eta(load_hists("truth", jet))
        save_fig(figure, "hist_eta_%s_truth.png" % jet)

    for jet in "abcd":
        figure, (axis_hi, axis_lo) = plot_eta(load_hists("reco", jet))
        save_fig(figure, "hist_eta_%s_reco.png" % jet)


def plot_eta(lambda_to_x_and_edges):
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
        loc="lower center",
    )

    xmin = -3
    xmax = 3

    axis_lo.plot(
        [xmin, xmax],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
    )

    axis_hi.set_ylabel(r"$\mathrm{normalized}~/~\mathrm{unit}$")

    axis_lo.set_ylabel(r"$\mathrm{ratio~to~SM}$")
    axis_lo.set_xlabel(r"$\mathrm{\eta}$")
    axis_lo.set_xlim(xmin, xmax)
    axis_lo.set_ylim(0.7, 1.3)

    return figure, (axis_hi, axis_lo)


def load_hists(tag, suffix):

    out = {}
    for label in ("0", "p2", "p4", "p6", "p8", "1"):
        lambda_ = float(label.replace("p", "."))
        hist_dict = json.load(
            open("results/hist/liv_%s/%s_eta_%s.json" % (label, tag, suffix))
        )
        x_and_edges = hist.arrays(hist_dict)
        # x_and_edges = hist.combine_array_bins(x_and_edges, binning)
        # crop to (-2.8, 2.8)
        x, edges = x_and_edges
        x = x[20:-20]
        edges = edges[20:-20]
        out[lambda_] = x, edges

    return out


if __name__ == "__main__":
    main()
