"""
Generate and serialize histograms.
"""
import json
from collections.abc import Callable
from numbers import Integral

import numpy


def histogram(func_bins_ranges, arrays):
    """Return jsonnable histograms of functions applied to arrays.

    Attempt to be data-efficient by accumulating functions mapped over given
    arrays (which may be iteratively read from disk) in one pass over the data.

    Arguments:
        func_bins_ranges:
            iterable of functions and numpy histogram arguments in trios
            funcs do not modify their arguments
        arrays:
            iterable of array-like objects to histogram
    """
    # arrange, validate and standardize arguments for json format
    func_bins_ranges_checked = []
    for func, bins, range_ in func_bins_ranges:
        assert isinstance(func, Callable)
        assert isinstance(bins, Integral)
        range_ = numpy.array(range_, dtype=float)
        assert range_.shape == (2,)

        func_bins_ranges_checked.append((func, bins, range_))

    # accumulate the histograms
    def gen_func_hists(array):
        for func, bins, range_ in func_bins_ranges_checked:
            yield numpy.histogram(func(unwritable(array)), bins, range_)[0]

    iter_arrays = iter(arrays)
    func_hists = list(gen_func_hists(next(iter_arrays)))
    for array in iter_arrays:
        for func_hist, func_hist_i in zip(func_hists, gen_func_hists(array)):
            func_hist += func_hist_i

    return [
        {"bins": bins, "range": range_.tolist(), "hist": func_hist.tolist()}
        for func_hist, (_, bins, range_) in zip(
            func_hists, func_bins_ranges_checked
        )
    ]


def arrays(hist_dict):
    """Return hist, bin_edges as from numpy.histogram."""
    bins = numpy.array(hist_dict["bins"], dtype=int)
    range_ = numpy.array(hist_dict["range"], dtype=float)
    hist = numpy.array(hist_dict["hist"], dtype=int)
    assert bins.shape == ()
    bins = bins.item()
    assert range_.shape == (2,)
    assert hist.shape == (bins,)

    bin_edges = numpy.linspace(*range_, bins + 1)
    return hist, bin_edges


def rebin(hist_dict, bins, range_):
    """Return hist_dict reduced to new implicable bins and range arguments."""
    assert isinstance(bins, Integral)
    range_ = numpy.array(range_, dtype=float)
    assert range_.shape == (2,)

    hist, bin_edges = arrays(hist_dict)

    i, j = bin_edges.searchsorted(range_)
    # values below range give 0
    assert i > 0 or range_[0] == bin_edges[0]
    assert j > 0 or range_[1] == bin_edges[0]
    # values above range give len
    assert i < len(bin_edges)
    assert j < len(bin_edges)
    ratio = (j - i) // bins

    new_hist = hist[i:j].reshape(bins, ratio).sum(axis=1)

    return {"bins": bins, "range": range_.tolist(), "hist": new_hist.tolist()}


def histogram2d(func_bins_ranges, arrays):
    """Return jsonnable 2d histograms of functions applied to arrays.

    Similar to histogram(...), but for numpy.histogram2d

    Arguments:
        func_bins_ranges:
            iterable of functions and numpy histogram2d arguments in trios
            funcs do not modify their arguments
            each func takes and returns x, y arrays
            bins are supported only in the int and [int, int] forms
        arrays:
            iterable of (x, y) array-like objects to histogram
    """
    # arrange, validate and standardize arguments for json format
    func_bins_ranges_checked = []
    for func, bins, range_ in func_bins_ranges:
        assert isinstance(func, Callable)
        if isinstance(bins, Integral):
            bins = [bins, bins]
        bins = numpy.array(bins, dtype=int)
        assert bins.shape == (2,)
        range_ = numpy.array(range_, dtype=float)
        assert range_.shape == (2, 2)

        func_bins_ranges_checked.append((func, bins, range_))

    # accumulate the histograms
    def gen_func_hists(array_x, array_y):
        for func, bins, range_ in func_bins_ranges_checked:
            x, y = func(unwritable(array_x), unwritable(array_y))
            # although numpy.histogram default to int dtype, numpy.histogram2d
            # returns float64; cast back for consistency
            yield numpy.histogram2d(x, y, bins, range_)[0].astype(int)

    iter_arrays = iter(arrays)
    func_hists = list(gen_func_hists(*next(iter_arrays)))
    for array_x, array_y in iter_arrays:
        func_hists_add = gen_func_hists(array_x, array_y)
        for func_hist, func_hist_i in zip(func_hists, func_hists_add):
            func_hist += func_hist_i

    return [
        {
            "bins": bins.tolist(),
            "range": range_.tolist(),
            "hist": func_hist.ravel().tolist(),
        }
        for func_hist, (_, bins, range_) in zip(
            func_hists, func_bins_ranges_checked
        )
    ]


def arrays2d(hist_dict):
    """Return hist, xedges, yedges as from numpy.histogram2d."""
    bins = numpy.array(hist_dict["bins"], dtype=int)
    range_ = numpy.array(hist_dict["range"], dtype=float)
    hist = numpy.array(hist_dict["hist"], dtype=int)
    assert bins.shape == (2,)
    assert range_.shape == (2, 2)
    hist = hist.reshape(*bins)
    xedges = numpy.linspace(*range_[0], bins[0] + 1)
    yedges = numpy.linspace(*range_[1], bins[1] + 1)
    return hist, xedges, yedges


def rebin2d(hist_dict, bins, range_):
    """Return hist_dict reduced to new implicable bins and range arguments."""
    bins = numpy.array(bins, dtype=int)
    assert bins.shape == (2,)
    xbins, ybins = bins
    range_ = numpy.array(range_, dtype=float)
    assert range_.shape == (2, 2)

    hist, xedges, yedges = arrays2d(hist_dict)

    # sum right to left (y then x)
    # values below range give 0
    # values above range give len
    xi, xj = xedges.searchsorted(range_[0])
    assert xi > 0 or range_[0, 0] == xedges[0]
    assert xj > 0 or range_[0, 1] == xedges[0]
    assert xi < len(xedges)
    assert xj < len(xedges)
    xratio = (xj - xi) // xbins

    yi, yj = yedges.searchsorted(range_[1])
    assert yi > 0 or range_[1, 0] == yedges[0]
    assert yj > 0 or range_[1, 1] == yedges[0]
    assert yi < len(yedges)
    assert yj < len(yedges)
    yratio = (yj - yi) // ybins

    new_hist = hist[xi:xj, yi:yj].reshape(-1, ybins, yratio).sum(axis=2)
    new_hist = new_hist.reshape(xbins, xratio, ybins).sum(axis=1)

    return {
        "bins": bins,
        "range": range_.tolist(),
        "hist": new_hist.ravel().tolist(),
    }


# utility


def unwritable(arr):
    """Return an unwritable view of arr."""
    # malicious funcs could modify array in place;
    # this is discouraged by passing a non writeable view
    unw = arr.view()
    unw.flags.writeable = False
    return unw


# testing


def test_histogram():
    # python -c "import hist; hist.test_histogram()"
    funcs_bins_ranges = [
        (lambda x: x, 10, (0, 1)),
        (lambda x: x + 1, 20, (0.1, 1.2)),
        (lambda x: x * 0.5, 9, (0.2, 1.1)),
    ]

    arrays = [
        numpy.arange(10) / 10,
        numpy.arange(10) / 5,
    ]

    hists = histogram(funcs_bins_ranges, arrays)

    for item, (func, bins, range_) in zip(hists, funcs_bins_ranges):
        numpy.testing.assert_array_equal(item["bins"], bins)
        numpy.testing.assert_array_equal(item["range"], range_)

        hist_parts = (
            numpy.histogram(func(a), bins, range_)[0] for a in arrays
        )
        hist_check = next(hist_parts)
        for hist_part in hist_parts:
            hist_check += hist_part

        numpy.testing.assert_array_equal(item["hist"], hist_check)

        # item must be json serializable
        json.dumps(item)


def test_histogram2d():
    # python -c "import hist; hist.test_histogram2d()"
    funcs_bins_ranges = [
        (lambda x, y: (x, y), 10, ((0, 1), (0, 1))),
        (lambda x, y: (x + 1, y - 1), (20, 5), ((0.1, 1.2), (-1, 1))),
        (lambda x, y: (x * 0.5, y * 2.0), (5, 5), ((0.2, 1.1), (-2, 2))),
    ]

    arrays = [
        (numpy.arange(10) / 10, numpy.arange(10) / 3),
        (numpy.arange(10) / 5, numpy.ones(10)),
    ]

    hists = histogram2d(funcs_bins_ranges, arrays)

    for item, (func, bins, range_) in zip(hists, funcs_bins_ranges):
        numpy.testing.assert_array_equal(item["bins"], bins)
        numpy.testing.assert_array_equal(item["range"], range_)

        hist_parts = (
            numpy.histogram2d(*func(x, y), bins, range_)[0] for x, y in arrays
        )
        hist_check = next(hist_parts)
        for hist_part in hist_parts:
            hist_check += hist_part

        numpy.testing.assert_array_equal(item["hist"], hist_check.ravel())

        # item must be json serializable
        json.dumps(item)


def test_arrays():
    # python -c "import hist; hist.test_arrays()"
    # 1d
    array = numpy.ones(3)
    bins = 10
    range_ = (0, 3)

    h1d = histogram([(lambda x: x, bins, range_)], [array])
    hist_test, bin_edges_test = arrays(h1d[0])

    hist_check, bin_edges_check = numpy.histogram(array, bins, range_)
    numpy.testing.assert_array_equal(hist_test, hist_check)
    numpy.testing.assert_array_equal(bin_edges_test, bin_edges_check)

    # 2d
    xarray = numpy.ones(3)
    yarray = numpy.ones(3)
    bins = [3, 3]
    range_ = [(0, 3), (1, 4)]

    h2d = histogram2d(
        [(lambda x, y: (x, y), bins, range_)], [(xarray, yarray)]
    )
    hist_test, xedges_test, yedges_test = arrays2d(h2d[0])

    hist_check, xedges_check, yedges_check = numpy.histogram2d(
        xarray, yarray, bins, range_
    )
    numpy.testing.assert_array_equal(hist_test, hist_check)
    numpy.testing.assert_array_equal(xedges_test, xedges_check)
    numpy.testing.assert_array_equal(yedges_test, yedges_check)


def test_rebin():
    # python -c "import hist; hist.test_rebin()"
    array = numpy.linspace(0, 1, 10)
    bins = 4
    range_ = (0, 1)

    hist_dict = histogram([(lambda x: x, bins, range_)], [array])[0]

    rebin_args = [
        (2, (0, 1)),
        (4, (0, 1)),
        (3, (0.25, 1)),
        (2, (0.25, 0.75)),
    ]

    for bins, range_ in rebin_args:
        hist_dict_rebinned = rebin(hist_dict, bins, range_)
        hist_check = numpy.histogram(array, bins, range_)[0]

        numpy.testing.assert_array_equal(
            hist_dict_rebinned["hist"], hist_check
        )
        numpy.testing.assert_array_equal(hist_dict_rebinned["bins"], bins)
        numpy.testing.assert_array_equal(hist_dict_rebinned["range"], range_)


def test_rebin2d():
    # python -c "import hist; hist.test_rebin2d()"
    xarray = numpy.linspace(0, 1, 10)
    yarray = numpy.linspace(0, 1, 10) + 0.1
    bins = [4, 4]
    range_ = [(0, 1), (0, 1)]

    hist_dict = histogram2d(
        [(lambda x, y: (x, y), bins, range_)], [(xarray, yarray)]
    )[0]

    rebin_args = [
        ([2, 2], [(0, 1), (0, 1)]),
        ([2, 4], [(0, 1), (0, 1)]),
        ([4, 2], [(0, 1), (0, 1)]),
        ([3, 2], [(0, 0.75), (0, 1)]),
        ([3, 2], [(0.25, 1), (0, 1)]),
        ([4, 3], [(0, 1), (0.25, 1)]),
        ([4, 3], [(0, 1), (0, 0.75)]),
    ]

    for bins, range_ in rebin_args:
        hist_dict_rebinned = rebin2d(hist_dict, bins, range_)
        hist_check = numpy.histogram2d(xarray, yarray, bins, range_)[0]

        numpy.testing.assert_array_equal(
            hist_dict_rebinned["hist"], hist_check.ravel()
        )
        numpy.testing.assert_array_equal(hist_dict_rebinned["bins"], bins)
        numpy.testing.assert_array_equal(hist_dict_rebinned["range"], range_)


def combine_array_bins(x_and_edges, ibins):
    """Return x_and_edges arrays with bins combined according to indices in ibins

    Each entry of ibins must be a sequence of indices to combine.

    All bins must be consumed in order.
    """
    x, edges = x_and_edges
    xout = []
    edges_out = [0]
    icheck = []
    for bins in ibins:
        bins = list(bins)
        icheck += bins
        xout.append(x[bins].sum())
        edges_out.append(edges[bins[-1] + 1])
    assert all(icheck[i] == i for i in range(len(x)))
    return numpy.array(xout), numpy.array(edges_out)
