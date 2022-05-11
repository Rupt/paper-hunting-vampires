"""Utilities for paper plots."""
import os

import matplotlib
import numpy
from matplotlib import pyplot

PREFIX = "plots"


def cmap_purple_orange(x):
    return matplotlib.cm.plasma(x * 0.7)


def cmap_greens(x):
    return matplotlib.cm.viridis(0.4 + x * 0.4)


def set_default_context():
    # use latex text / fonts to match document
    # https://matplotlib.org/stable/tutorials/text/usetex.html
    pyplot.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 10,
            "figure.facecolor": "w",
        }
    )


# figures


def dual_subplots(*, dpi=400, figsize=(2.4, 2.4), hspace=0.05):
    """Return a standard plot-ratio plot figure, axis."""
    return pyplot.subplots(
        2,
        1,
        sharex=True,
        dpi=dpi,
        figsize=figsize,
        gridspec_kw={
            "height_ratios": [3, 1],
            "top": 0.99,
            "right": 0.94,
            "bottom": 0.18,
            "left": 0.24,
            "wspace": 0,
            "hspace": hspace,
        },
    )


def save_fig(figure, path, *, verbose=True):
    os.makedirs(PREFIX, exist_ok=True)
    fullpath = os.path.join(PREFIX, path)
    figure.savefig(fullpath)
    if verbose:
        print("wrote %r" % fullpath)
    pyplot.close(figure)


# histograms


def hist_sqrterr(
    axis,
    x_and_edges,
    normed=False,
    color=None,
    alpha=1.0,
    **kwargs,
):
    """Draw a histogram to axis with sqrt(n) errorbar shading."""
    x, edges = x_and_edges
    return hist_err(axis, x, x**0.5, edges, normed, color, alpha, **kwargs)


def hist_err(
    axis,
    x,
    xerr,
    edges,
    normed=False,
    color=None,
    alpha=1.0,
    **kwargs,
):
    x = x.astype(float)
    xerr = xerr.astype(float)

    if normed:
        scale = 1 / (edges[1:] - edges[:-1])
        norm = scale / x.sum()
        x *= norm
        xerr *= norm

    # appending an empty bin avoids vertical edge lines
    _, _, (poly_hist,) = axis.hist(
        numpy.append(edges[:-1], numpy.nan),
        bins=numpy.append(edges, numpy.nan),
        weights=numpy.append(x, numpy.nan),
        histtype="step",
        color=color,
        alpha=alpha,
        **kwargs,
    )

    poly_fill = axis.fill_between(
        interleave(edges[:-1], edges[1:]),
        interleave(x - xerr),
        interleave(x + xerr),
        alpha=0.2 * alpha,
        color=poly_hist.get_facecolor(),
        linewidth=0,
    )

    return poly_hist, poly_fill


def hist_ratio(
    axis,
    x_and_edges,
    y_and_edges,
    normed=False,
    color=None,
    alpha=1.0,
    **kwargs,
):
    """Draw a histogram ratio with "standard" errorbar shading."""
    x, edges = x_and_edges
    y, edges_check = y_and_edges
    numpy.testing.assert_array_equal(edges, edges_check)

    xerr = x**0.5
    yerr = y**0.5

    if normed:
        xnorm = 1 / x.sum()
        x = x * xnorm
        xerr *= xnorm
        ynorm = 1 / y.sum()
        y = y * ynorm
        yerr *= ynorm

    rat, rat_err = div_error_propagation(x, xerr, y, yerr)

    # nan weights appear to propagate when bins is a sequence of edges
    # so use plot as a substitute
    # match some default arguments
    plot_kwargs = dict(
        linewidth=1,
    )
    plot_kwargs.update(kwargs)

    (poly_hist,) = axis.plot(
        interleave(edges[:-1], edges[1:]),
        interleave(rat),
        color=color,
        alpha=alpha,
        **plot_kwargs,
    )

    poly_fill = axis.fill_between(
        interleave(edges[:-1], edges[1:]),
        interleave(rat - rat_err),
        interleave(rat + rat_err),
        alpha=0.2 * alpha,
        color=color,
        linewidth=0,
    )

    return poly_hist, poly_fill


def plot_qualities(
    label_to_spec, select_func=None, *, sigma=2, xlim=(-0.05, 1.05), space=0.1
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


# misc


def div_error_propagation(x, dx, y, dy):
    """Return x/y and a standard error assignment."""
    # this assignment is only valid in dx << x limits
    # and obviously fails for zero denominators
    # we assume d* >= 0
    bad = (dx > x) | (dy > y) | (x == 0) | (y == 0)
    x[bad] = 1
    y[bad] = 1
    r = x / y
    dr = r * ((dx / x) ** 2 + (dy / y) ** 2) ** 0.5
    r[bad] = numpy.nan
    dr[bad] = numpy.nan
    return r, dr


def interleave(x, y=None):
    if y is None:
        y = x
    return numpy.stack([x, y], axis=1).ravel()
