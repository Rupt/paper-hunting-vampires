"""Dump p_T histogram plots.

Usage:

python plot/plot_pileup_comparison.py

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

# todo maybe full yields?


def main():
    plot_lib.set_default_context()

    # can do eta and ht here too
    for jet in ["_a", "_b", "_c", "_d"]:
        combined_hists = {}
        nopileup_hists = load_hists("reco_nopileup", jet, "pt")
        pileup_hists = load_hists("reco", jet, "pt")
        combined_hists["nopileup"] = nopileup_hists[0.0]
        combined_hists["pileup"] = pileup_hists[0.0]

        figure, (axis_hi, axis_lo) = plot_pt(combined_hists)
        save_fig(figure, "hist_pt_%s_nopileup_vs_pileup.png" % jet)

        # plot eta
        # def...
        combined_hists = {}
        nopileup_hists = load_hists("reco_nopileup", jet, "eta")
        pileup_hists = load_hists("reco", jet, "eta")
        combined_hists["nopileup"] = nopileup_hists[0.0]
        combined_hists["pileup"] = pileup_hists[0.0]

        figure, (axis_hi, axis_lo) = plot_eta(combined_hists)
        save_fig(figure, "hist_eta_%s_nopileup_vs_pileup.png" % jet)

    # plot ht

    for suffix in ("", "_3j", "_4j"):
        combined_hists = {}
        nopileup_hists = load_hists("reco_nopileup", suffix, "ht")
        pileup_hists = load_hists("reco", suffix, "ht")
        combined_hists["nopileup"] = nopileup_hists[0.0]
        combined_hists["pileup"] = pileup_hists[0.0]

        figure, (axis_hi, axis_lo) = plot_ht(combined_hists)
        save_fig(figure, "hist_ht_%s_nopileup_vs_pileup.png" % suffix)


def plot_pt(lambda_to_x_and_edges):
    nopileup_arrays = lambda_to_x_and_edges["nopileup"]
    pileup_arrays = lambda_to_x_and_edges["pileup"]

    figure, (axis_hi, axis_lo) = dual_subplots()

    poly_hist_nopileup, _ = hist_sqrterr(
        axis_hi, nopileup_arrays, normed=True, color=cmap_purple_orange(0)
    )

    color = cmap_purple_orange(1)

    poly_hist_i, _ = hist_sqrterr(
        axis_hi,
        pileup_arrays,
        normed=True,
        color=color,
    )
    hist_ratio(
        axis_lo,
        pileup_arrays,
        nopileup_arrays,
        normed=True,
        color=poly_hist_i.get_edgecolor(),
    )

    axis_hi.legend(
        [poly_hist_nopileup, poly_hist_i],
        ["nopileup", "pileup"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    xmin = 0
    xmax = 1000  # edit this

    axis_lo.plot(
        [xmin, xmax],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
    )

    axis_hi.set_ylabel(r"$\mathrm{normalized}~/~\mathrm{GeV}$")
    # axis_hi.set_yscale("log")

    axis_lo.set_ylabel(r"$\mathrm{ratio}$")
    axis_lo.set_xlabel(r"$\mathrm{p_T}~/~\mathrm{GeV}$")
    axis_lo.set_xlim(xmin, xmax)
    axis_lo.set_ylim(0.5, 1.5)

    return figure, (axis_hi, axis_lo)


def plot_eta(lambda_to_x_and_edges):

    nopileup_arrays = lambda_to_x_and_edges["nopileup"]
    pileup_arrays = lambda_to_x_and_edges["pileup"]

    figure, (axis_hi, axis_lo) = dual_subplots()

    figure, (axis_hi, axis_lo) = dual_subplots()

    poly_hist_nopileup, _ = hist_sqrterr(
        axis_hi, nopileup_arrays, normed=True, color=cmap_purple_orange(0)
    )

    color = cmap_purple_orange(1)

    poly_hist_i, _ = hist_sqrterr(
        axis_hi,
        pileup_arrays,
        normed=True,
        color=color,
    )
    hist_ratio(
        axis_lo,
        pileup_arrays,
        nopileup_arrays,
        normed=True,
        color=poly_hist_i.get_edgecolor(),
    )

    axis_hi.legend(
        [poly_hist_nopileup, poly_hist_i],
        ["nopileup", "pileup"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    xmin = 0
    xmax = 1000  # edit this

    axis_lo.plot(
        [xmin, xmax],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
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


def plot_ht(lambda_to_x_and_edges):

    nopileup_arrays = lambda_to_x_and_edges["nopileup"]
    pileup_arrays = lambda_to_x_and_edges["pileup"]

    figure, (axis_hi, axis_lo) = dual_subplots()

    figure, (axis_hi, axis_lo) = dual_subplots()

    poly_hist_nopileup, _ = hist_sqrterr(
        axis_hi, nopileup_arrays, normed=True, color=cmap_purple_orange(0)
    )

    color = cmap_purple_orange(1)

    poly_hist_i, _ = hist_sqrterr(
        axis_hi,
        pileup_arrays,
        normed=True,
        color=color,
    )
    hist_ratio(
        axis_lo,
        pileup_arrays,
        nopileup_arrays,
        normed=True,
        color=poly_hist_i.get_edgecolor(),
    )

    axis_hi.legend(
        [poly_hist_nopileup, poly_hist_i],
        ["nopileup", "pileup"],
        frameon=False,
        borderpad=0,
        labelspacing=0.2,
    )

    xmin = 0
    xmax = 1000  # edit this

    axis_lo.plot(
        [xmin, xmax],
        [1, 1],
        linestyle="--",
        linewidth=1,
        color="xkcd:grey",
        alpha=0.5,
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

    axis_hi.set_ylabel(r"$\textrm{normalized}~/~\mathrm{Ge\kern-0.15ex V}$")
    axis_hi.set_yscale("log")

    axis_lo.set_ylabel(r"$\textrm{ratio~to~SM}$")
    axis_lo.set_xlabel(r"$H_\mathrm{T}~/~\mathrm{Ge\kern-0.15ex V}$")
    axis_lo.set_xlim(xmin, xmax)
    axis_lo.set_ylim(0, 7)

    return figure, (axis_hi, axis_lo)


def load_hists(tag, suffix, var):
    nbins = 100

    out = {}
    label = "0"
    lambda_ = float(label.replace("p", "."))
    hist_dict = json.load(
        open("results/hist/liv_%s/%s_%s%s.json" % (label, tag, var, suffix))
    )
    hist_dict = hist.rebin(hist_dict, nbins, hist_dict["range"])
    out[lambda_] = hist.arrays(hist_dict)

    return out


if __name__ == "__main__":
    main()
