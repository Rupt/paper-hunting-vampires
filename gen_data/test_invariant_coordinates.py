"""Test invariant_coordinates.py.


Usage:

python gen_data/test_invariant_coordinates.py

"""
import unittest

import numpy
from invariant_coordinates import invariant_momenta, parity_flip


class Test_invariant_momenta(unittest.TestCase):
    def test_rotation_matrix(self):

        numpy.testing.assert_allclose(
            [1, 1, 0, 2] @ rotation_matrix(numpy.pi / 4),
            [1, 2**-0.5, 2**-0.5, 2],
        )

    def test_rotations(self):
        rng = numpy.random.Generator(numpy.random.Philox(321))

        shapes = (
            (3000, 3 * 3),
            (2000, 3 * 4),
            (1000, 3 * 5),
        )

        phis = numpy.linspace(0, 2 * numpy.pi, 10)

        def check_rotations(x):
            atol = numpy.finfo(x.dtype).eps * 8
            d = invariant_momenta(x)

            # rotate about x st z = -z, y = -y
            xrot = x.copy()
            xrot[:, 1::3] *= -1
            xrot[:, 2::3] *= -1

            drot = invariant_momenta(xrot)
            numpy.testing.assert_allclose(d, drot, atol=atol)

            # rotate about z st x = -y, y = x
            self.assertEqual(d.dtype, x.dtype)

            xrot = x.copy()
            xrot[:, ::3] = -x[:, 1::3]
            xrot[:, 1::3] = x[:, ::3]

            drot = invariant_momenta(xrot)
            numpy.testing.assert_allclose(d, drot, atol=atol)

            # other rotations
            for phi in phis:
                drot = invariant_momenta(rotate(x, phi))
                numpy.testing.assert_allclose(d, drot, atol=atol)

        for shape in shapes:
            momenta = rng.normal(size=shape)
            check_rotations(momenta)
            check_rotations(momenta.astype(numpy.float32))


class Test_parity_flip(unittest.TestCase):
    def test_parity(self):
        rng = numpy.random.Generator(numpy.random.Philox(0xB00))

        shapes = (
            (3000, 3 * 3),
            (2000, 3 * 4),
            (1000, 3 * 5),
        )

        for shape in shapes:
            momenta = rng.normal(size=shape)

            inv = invariant_momenta(momenta)
            pinv = invariant_momenta(-momenta)
            check_pinv = parity_flip(inv)

            numpy.testing.assert_equal(check_pinv, pinv)


def rotate(momenta, phi):
    """Return momenta rotated by theta about the z axis."""
    rot = rotation_matrix(phi)[1:, 1:].astype(momenta.dtype)
    ps = momenta.reshape(len(momenta), -1, 3)
    return (ps @ rot).reshape(*momenta.shape)


def rotation_matrix(phi):
    """Return a matrix M such that p @ M = rot_by_theta(p)."""
    c = numpy.cos(phi)
    s = numpy.sin(phi)
    return numpy.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ]
    ).T


if __name__ == "__main__":
    unittest.main()
