"""
Make illustrations of how a histogram analysis can be done.

Usage:

python plot/plot_two_bin_stats.py

"""
import numpy
import plot_lib
from matplotlib import pyplot
from numpy import float32
from scipy.special import xlogy


def main():
    plot_lib.set_default_context()

    plot_twobin()


def plot_twobin():
    nminus = 19
    nplus = 42

    mu_max = (nminus + nplus) / 2
    post_fit_yields = (mu_max - 2.3, mu_max + 1.8)

    print(f"{nplus = }")
    print(f"{nminus = }")
    print()

    significance = sigma(twobin_poisson_llr(nplus, nminus))
    print(f"{significance = :.1f}")
    significance_gauss = sigma(twobin_gaussian_llr(nplus, nminus))
    print(f"{significance_gauss = :.1f}")
    print()

    print(f"{post_fit_yields = }")
    significance_post_fit = sigma(
        poisson_llr(nminus, post_fit_yields[0])
        + poisson_llr(nplus, post_fit_yields[1])
    )
    print(f"{significance_post_fit = :.1f}")
    print()

    figure, axis = pyplot.subplots(
        dpi=400,
        figsize=(2.6, 2.4),
        gridspec_kw={
            "top": 0.99,
            "right": 0.99,
            "bottom": 0.08,
            "left": 0.16,
        },
    )

    # one-sigma main
    lo_minus, hi_minus = poisson_interval(nminus, llr(1))
    lo_plus, hi_plus = poisson_interval(nplus, llr(1))
    errors = [
        [nminus - lo_minus, nplus - lo_plus],
        [hi_minus - nminus, hi_plus - nplus],
    ]

    data = axis.errorbar(
        [-0.5, 0.5],
        [nminus, nplus],
        yerr=errors,
        xerr=0.5,
        color="k",
        marker="o",
        markersize=5,
        markeredgewidth=0,
        linewidth=0,
        elinewidth=1,
    )
    # two-sigma
    lo_minus_2sig, hi_minus_2sig = poisson_interval(nminus, llr(2))
    lo_plus_2sig, hi_plus_2sig = poisson_interval(nplus, llr(2))
    axis.plot(
        [-0.5, -0.5],
        [lo_minus_2sig, hi_minus_2sig],
        "k:",
        lw=1,
        zorder=1.9,
    )
    axis.plot(
        [0.5, 00.5],
        [lo_plus_2sig, hi_plus_2sig],
        "k:",
        lw=1,
        zorder=1.9,
    )

    # background fit histograms
    _, _, (max_like,) = axis.hist(
        [-0.5, 0.5],
        weights=(mu_max, mu_max),
        bins=2,
        range=(-1, 1),
        histtype="step",
        zorder=1.8,
        color=plot_lib.cmap_purple_orange(0),
    )

    background = (*plot_lib.cmap_purple_orange(1)[:3], 0.05)
    _, _, (post_fit,) = axis.hist(
        [-0.5, 0.5],
        weights=post_fit_yields,
        bins=2,
        range=(-1, 1),
        histtype="stepfilled",
        zorder=1.7,
        color=background,
        edgecolor=plot_lib.cmap_purple_orange(1),
    )

    axis.plot(
        [0, 0],
        [0, max(post_fit_yields)],
        color=plot_lib.cmap_purple_orange(1),
        lw=1,
        zorder=1.7,
    )

    # bin labels
    axis.text(
        0.25,
        -0.02,
        r"$\textrm{SR-minus}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=axis.transAxes,
    )

    axis.text(
        0.75,
        -0.02,
        r"$\textrm{SR-plus}$",
        horizontalalignment="center",
        verticalalignment="top",
        transform=axis.transAxes,
    )

    # legend
    axis.legend(
        [data, max_like, post_fit],
        [
            r"$\textrm{data}\pm\!1\sigma\pm\!2\sigma$",
            r"$\textrm{best fit}, %.1f\sigma$" % significance,
            r"$\textrm{+ systematics}, %.1f\sigma$" % significance_post_fit,
        ],
        frameon=False,
        loc="upper left",
        borderpad=0,
        labelspacing=0.2,
    )

    axis.set_ylim(0, 82)
    axis.set_xlim(-1, 1)
    axis.set_xticks([-1, 0, 1])
    axis.set_xticklabels([])
    axis.set_ylabel(r"$\textrm{events}$")

    plot_lib.save_fig(figure, "two_bin_stats.png")


# significance measures


def twobin_gaussian_llr(n1, n2):
    if n1 == n2 == 0:
        return 0.0

    return -0.5 * (n1 - n2) ** 2 / (n1 + n2)


def twobin_poisson_llr(n1, n2):
    # bracket to ensure n1 n2 commutativity
    return xlogy(n1 + n2, 0.5 * (n1 + n2)) - (xlogy(n1, n1) + xlogy(n2, n2))


# stats


def poisson_interval(n, llr_):
    """Return (lo, hi) between which poisson_llr >= llr."""
    assert n >= 0
    assert llr_ <= 0

    # special cases
    if llr_ == 0:
        return (float(n), float(n))

    if n == 0:
        return (0.0, -float(llr_))

    # bisection to float32 precision
    def func(x):
        return poisson_llr(n, x) - llr_

    delta = 2 * sigma(llr_) * n

    while func(n + delta) > 0:
        delta *= 2

    hi = bisectf(func, n + delta, n)
    lo = bisectf(func, 0, n)

    return lo, hi


def bisectf(func, lo, hi):
    func_lo = func(lo)
    func_hi = func(hi)
    assert func_lo <= 0, func_lo
    assert func_hi >= 0, func_hi

    while float32(lo) != float32(hi):
        mid = 0.5 * (lo + hi)
        func_mid = func(mid)

        if func_mid >= 0:
            hi = mid
            func_hi = func_mid

        if func_mid <= 0:
            lo = mid
            func_lo = func_mid

    return lo


def llr(sigma):
    """Return the Gaussian sigma for log likelihood ratio llr."""
    return -0.5 * sigma**2


def sigma(llr):
    """Return the inverse of llr."""
    return (-2 * llr) ** 0.5


def poisson_llr(n, mu):
    """Return the poisson log likelihood difference from maximum.

    Arguments:
        n: observed events >= 0
        mu: expectation
    """
    return n - mu + xlogy(n, mu / numpy.maximum(1, n))


if __name__ == "__main__":
    main()
