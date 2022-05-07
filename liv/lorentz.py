""" Lorentz transformations. """
import numba
import numpy
from numba import float64


@numba.njit(float64(float64))
def yamma(b2):
    """Return the y for which e = y*m from b2 = (v/c)**2."""
    return 1 / (1 - b2) ** 0.5


c0 = 1 / 2
c1 = 3 / 8
c2 = 5 / 16
c3 = 35 / 128
c4 = 63 / 256
c5 = 231 / 1024
c6 = 429 / 2048


@numba.njit(float64(float64))
def ym1_b2(b2):
    """Return (y - 1)/b2."""
    if b2 < 2 ** -6:
        # Taylor series about 0
        return c0 + b2 * (c1 + b2 * (c2 + b2 * (c3 + b2 * (c4 + b2 * (c5 + b2 * c6)))))
    return (yamma(b2) - 1) / b2


@numba.njit(float64[:, :](float64, float64, float64))
def make_boost(bx, by, bz):
    """Return the matrix for a passive Lorentz boost by (bx, by, by)"""
    b2 = bx * bx + by * by + bz * bz
    assert b2 < 1

    y = yamma(b2)
    g = ym1_b2(b2)
    return numpy.array(
        (
            (y, -y * bx, -y * by, -y * bz),
            (-y * bx, 1 + bx * bx * g, bx * by * g, bx * bz * g),
            (-y * by, bx * by * g, 1 + by * by * g, by * bz * g),
            (-y * bz, bx * bz * g, by * bz * g, 1 + bz * bz * g),
        )
    )


# testing
import itertools
import unittest

import mpmath
from mpmath import mp


def yamma_ref(b2):
    """Reference implementation for yamma(...)."""
    b2 = mp.mpf(b2)
    y = (1 - b2) ** -0.5
    return float(y)


def ym1_b2_ref(b2):
    """Reference implementation for ym1_b2(...)."""
    b2 = mp.mpf(b2)
    ym1 = (1 - b2) ** -0.5 - 1
    # Towards zero, we can get both /0 and 0/ errors. Avoid both.
    if ym1 and b2:
        return float(str(ym1 / b2))
    return 0.5


def make_boost_ref(bx, by, bz):
    """Reference implementation for make_boost(...)."""
    bx = mp.mpf(bx)
    by = mp.mpf(by)
    bz = mp.mpf(bz)
    # Convert to rapidity vector.
    b = (bx * bx + by * by + bz * bz) ** 0.5
    eta = -mp.atanh(b)
    x = b and bx / b
    y = b and by / b
    z = b and bz / b
    # Using the expanded exponential from from Wikipedia.
    k = mp.matrix(
        (
            (0, x, y, z),
            (x, 0, 0, 0),
            (y, 0, 0, 0),
            (z, 0, 0, 0),
        )
    )
    transform = mp.eye(4) + mp.sinh(eta) * k + (mp.cosh(eta) - 1) * k ** 2
    return numpy.array(transform, dtype=float).reshape((4, 4))


def mass(x):
    """4-vector mass."""
    x = numpy.asarray(x)
    if x.shape == (4,):
        x = x.reshape(1, 4)
    return (x[:, 0] ** 2 - numpy.einsum("ij,ij->i", x[:, 1:], x[:, 1:])) ** 0.5


def listify(iterable):
    """Return lists of the elements of items yielded from iterator.

    All items are assumed to have the same length.
    """
    it = iter(iterable)

    out = tuple([item_i] for item_i in next(it, ()))

    for item in it:
        for out_i, item_i in zip(out, item):
            out_i.append(item_i)

    return out


def betas():
    """A nice selection of beta values to test."""
    irange = range(1, 20)

    return itertools.chain(
        (2 ** -i for i in irange),
        (numpy.random.rand() * 2 ** -i for i in irange),
        (1 - 2 ** -i for i in irange),
        (1 - max(numpy.random.rand() * 2 ** -i, 2 ** -53) for i in irange),
    )


class TestLorentz(unittest.TestCase):
    def test_yamma(self):
        def yamma_checkref(b):
            """Return a yamma(...), reference pair."""
            return yamma(b), yamma_ref(b)

        numpy.testing.assert_allclose(*yamma_checkref(0.0))

        check, ref = listify(map(yamma_checkref, betas()))
        numpy.testing.assert_allclose(check, ref, rtol=1e-14)

    def test_ym1_b2(self):
        def ym1_b2_checkref(b):
            """Return a ym1_b2(...), reference pair."""
            return ym1_b2(b), ym1_b2_ref(b)

        numpy.testing.assert_allclose(*ym1_b2_checkref(0.0))

        check, ref = listify(map(ym1_b2_checkref, betas()))
        numpy.testing.assert_allclose(check, ref, rtol=1e-13)

    def test_make_boost(self):
        def make_boost_checkref(beta):
            """Return a make_boost(...), reference pair."""
            return make_boost(*beta), make_boost_ref(*beta)

        vec4s = numpy.array(
            (
                (3, 0, 0, 0),
                (2, 1, 0, 0),
                (5, 0, 2, 0),
                (7, 0, 0, 3),
                (7, 3, 2, 1),
                (7, 3, -2, 1),
                (7, -3, 2, 1),
                (7, -3, -2, 1),
                (7, -3, 2, -1),
                (1, 1 - 1e-15, 0, 0),
                (1, 1e-15, 0, 0),
            )
        )

        betas = vec4s[:, 1:] / vec4s[:, 0, numpy.newaxis]
        rng = numpy.random.Generator(numpy.random.Philox(123))
        betas = numpy.concatenate(
            [betas, numpy.tanh(rng.normal(size=len(betas)))[:, numpy.newaxis] * betas]
        )

        check, ref = listify(map(make_boost_checkref, betas))
        numpy.testing.assert_allclose(check, ref, rtol=1e-14, atol=1e-8)


if __name__ == "__main__":
    mp.prec = 255
    unittest.main()
