"""Transform to rotation-invariant coordinates."""
import numba
import numpy


@numba.njit(
    [
        numba.float32[:, :](numba.float32[:, :]),
        numba.float64[:, :](numba.float64[:, :]),
    ]
)
def invariant_momenta(momenta):
    """Return a collider-rotationally invariant representation of momenta.

    Output coordinates are in an orthonormal basis where:
        x is aligned to the transverse momentum of the first particle
        z is aligned to its z direction
        y is the cross product of z and x

    Here the first particle has zero y component, which we drop.

    Arguments:
        momenta:
            shape (n, 3 * k)
            where the 3 is for (px, py, pz)
            and k >= 1

    Returns:
        shape (n, 3 * k - 1)
    """
    assert momenta.shape[1] > 0
    assert momenta.shape[1] % 3 == 0
    dtype = momenta.dtype
    new = numpy.zeros((len(momenta), momenta.shape[1] - 1), dtype=dtype)

    for i, features in enumerate(momenta):
        p = features.copy().reshape(-1, 3)
        a = p[0]

        # x in direction of hardest jet
        at = numpy.array((a[0], a[1], 0.0), dtype=dtype)
        e0 = at * at.dot(at) ** numpy.array(-0.5, dtype=dtype)
        # z in direction of hardest jet
        e2 = numpy.array((0.0, 0.0, numpy.copysign(1.0, a[2])), dtype=dtype)
        # y right handed from z an x
        e1 = numpy.cross(e2, e0)

        basis = numpy.stack((e0, e1, e2)).T
        e = p.dot(basis).ravel()

        # py_a is 0
        new[i] = numpy.delete(e, 1)

    return new


def parity_flip(inv_momenta):
    """Return an equivalent to invariant_momenta(-momenta)."""
    # the effect of parity is to change the sign of y components
    out = inv_momenta.copy()
    # x z x y z ...
    out[:, range(3, inv_momenta.shape[1], 3)] *= -1
    return out
