"""
Create a PV-mSME model for MadGraph, with configured parameters.

Find usage examples are in README.md.

"""
import argparse
import glob
import itertools
import json
import os
import subprocess

import numpy

RNG = numpy.random.Generator(numpy.random.Philox(0xBAD))

TEMPLATE_DIR = "model_template"
VEUP = "FFV1smeVEUP*.f"
VEDN = "FFV1smeVEDN*.f"
AXUP = "FFV1smeAXUP*.f"
AXDN = "FFV1smeAXDN*.f"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Lorentz Invariance Violating (LIV) "
            "standard model extension for MadGraph"
        )
    )
    parser.add_argument("parameters", type=str, help="parameters in json format")
    parser.add_argument("output_dir", type=str, help="output directory path")
    parser.add_argument(
        "--rotation_angle",
        type=float,
        help="rotate about the y axis by these many radians",
        default=None,
    )

    args = parser.parse_args()

    with open(args.parameters, "r") as file_:
        parameters = json.load(file_)

    # dump
    configure(parameters, args.output_dir, rotation_angle=args.rotation_angle)


def configure(parameters, output_dir, *, rotation_angle=None):
    """Write an LIV SME model with given parameters to output_dir.

    Only supports 0, 0 generation indices.

    (we use these for all generations)

    Arguments:
        ...
        rotation_angle:
            rotate about the y axis (Earth's pole axis in certain coordinates)
            by this angle in radians
    """
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["cp", "-r", "-T", template_dir(), output_dir])
    path = os.path.join(output_dir, "Fortran")

    def update(wildcard, prefix, couplings):
        select = glob.glob(os.path.join(path, wildcard))

        for mu, nu in itertools.product(range(4), range(4)):
            cname = f"{prefix}{mu}{nu}"
            cmunu = couplings[mu, nu, 0, 0]
            coupling = f"({cmunu.real}D0, {cmunu.imag}D0)"
            command = [
                "sed",
                "-i",
                f"s/{cname} = .*/{cname} = {coupling}/g",
                *select,
            ]
            subprocess.run(command)

    q, u, d = dict_to_qud(parameters)
    if rotation_angle is not None:
        rot = _rotate_matrix(rotation_angle)
        q = rot(q)
        u = rot(u)
        d = rot(d)
    check_traceless_hermitian(q)
    check_traceless_hermitian(u)
    check_traceless_hermitian(d)
    axup, axdn, veup, vedn = qud_to_avud(q, u, d)
    verbose = True
    if verbose:
        print(axup[:, :, 0, 0].real)
        print(axdn[:, :, 0, 0].real)
        print(veup[:, :, 0, 0].real)
        print(vedn[:, :, 0, 0].real)
    update(AXUP, "ca", axup)
    update(AXDN, "ca", axdn)
    update(VEUP, "cv", veup)
    update(VEDN, "cv", vedn)


def dump_random(output_path, scale=1, seed=0xBAD):
    """Write a random example couplings file to output_path."""
    rng = numpy.random.Generator(numpy.random.Philox(seed))
    q = random_traceless_hermitian(rng) * scale
    u = random_traceless_hermitian(rng) * scale
    d = random_traceless_hermitian(rng) * scale
    qud_dict = qud_to_dict(q, u, d)

    with open(output_path, "w") as file_:
        json.dump(qud_dict, file_, indent="\t")
    print("Wrote %r." % output_path)


def dump_zero(output_path):
    """Write an empty example couplings file to output_path."""
    q = numpy.zeros((4, 4, 3, 3), dtype=complex)
    qud_dict = qud_to_dict(q, q, q)

    with open(output_path, "w") as file_:
        json.dump(qud_dict, file_, indent="\t")
    print("Wrote %r." % output_path)


def random_traceless_hermitian(rng):
    """Return a random traceless hermitian matrix."""
    c = numpy.zeros((4, 4, 3, 3), dtype=complex)
    # only filling first generation; 0, 0 must be real for hermitian
    c00 = rng.normal(size=(4, 4))
    c00 -= numpy.eye(4) * (c00.trace() / 4)
    c[:, :, 0, 0] = c00
    return c


# earth rotation business


def _rotate_matrix(theta):
    """Return a function which rotates fourmatrices by angle theta.

    In original frame
        A = x.T @ M @ x

    In transformed frame rotated by (orthogonal) matrix R, x' = R @ x

        A = x.T @ R.T @ C @ R @ x

        such that C' = R.T @ C @ R

    Convention: if starting at positive z [0, 0, 0, 1], a rotation of pi/2
    takes us to positive x [0, 1, 0, 0].


    Test:
    python -c "import configure; configure._test_rotate_matrix()"

    """
    # rotate x and x about y axis
    sint = numpy.sin(theta)
    cost = numpy.cos(theta)

    big_r = numpy.array(
        [
            # e
            [
                1,
                0,
                0,
                0,
            ],
            # x
            [
                0,
                cost,
                0,
                sint,
            ],
            # y
            [
                0,
                0,
                1,
                0,
            ],
            # z
            [
                0,
                -sint,
                0,
                cost,
            ],
        ],
        dtype=float,
    )

    def rot(m):
        # big_r @ m @ big_r.T, but allow extended dimensions of m
        return numpy.einsum("ij,jk...,kl->il...", big_r, m, big_r.T)

    return rot


def _test_rotate_matrix():
    rng = numpy.random.Generator(numpy.random.Philox(123))

    c = rng.normal(size=16).reshape(4, 4)
    x = rng.normal(size=4)

    acheck = x.T @ c @ x

    for theta in [0.0, 0.1, 0.5 * numpy.pi, numpy.pi, 2 * numpy.pi]:
        rot = _rotate_matrix(theta)
        xdash = numpy.array(
            [
                x[0],
                x[1] * numpy.cos(theta) + x[3] * numpy.sin(theta),
                x[2],
                x[3] * numpy.cos(theta) - x[1] * numpy.sin(theta),
            ]
        )
        numpy.testing.assert_allclose(acheck, xdash.T @ rot(c) @ xdash)

    numpy.testing.assert_allclose(_rotate_matrix(2 * numpy.pi)(c), c)

    c2 = rng.normal(size=16).reshape(4, 4)

    rot = _rotate_matrix(0.321)
    rot_stack = numpy.stack([c, c2], axis=-1)[..., numpy.newaxis]
    assert rot_stack.shape == (4, 4, 2, 1)
    a_stack = rot(rot_stack)
    numpy.testing.assert_array_equal(a_stack[:, :, 0, 0], rot(c))
    numpy.testing.assert_array_equal(a_stack[:, :, 1, 0], rot(c2))


# utility


def template_dir():
    """Return the template model directory path."""
    return os.path.join(os.path.dirname(__file__), TEMPLATE_DIR)


def qud_to_avud(q, u, d):
    """Translate q u d parameter matrices to (ax ve) x (up dn)."""
    axup = u - q
    axdn = d - q
    veup = u + q
    vedn = d + q
    return axup, axdn, veup, vedn


def qud_to_dict(q, u, d):
    """Return a dict representation of parameter matrices."""
    out = {}

    for label, matrix in [("q", q), ("u", u), ("d", d)]:
        for mu, nu, a, b in itertools.product(range(4), range(4), range(3), range(3)):
            value = matrix[mu, nu, a, b]
            if value == 0:
                continue
            out[f"{label}{mu}{nu}{a}{b}"] = repr(value)

    return out


def dict_to_qud(qud_dict):
    """Return a matrix representation of parameter matrices."""
    # two lorentz indices, two generation indices
    q = numpy.empty((4, 4, 3, 3), dtype=complex)
    u = numpy.empty_like(q)
    d = numpy.empty_like(q)

    for label, matrix in [("q", q), ("u", u), ("d", d)]:
        for mu, nu, a, b in itertools.product(range(4), range(4), range(3), range(3)):
            matrix[mu, nu, a, b] = complex(qud_dict.get(f"{label}{mu}{nu}{a}{b}", 0))

    return q, u, d


def check_traceless_hermitian(c):
    """Assert that c has certain required properties:

    Traceless in lorentz indices (first two)

    Hermitian in generation indices (latter two)
    """
    # traceless c[mu, nu]
    for a, b in itertools.product(range(3), range(3)):
        c_ab = c[:, :, a, b]
        numpy.testing.assert_allclose(
            c_ab.trace(), 0, atol=numpy.linalg.norm(c_ab.diagonal()) * 1e-13
        )
    # hermitian c[mu, nu, a, b] in {a, b}
    for a, b in itertools.product(range(3), range(3)):
        c_ab = c[:, :, a, b]
        c_ba = c[:, :, b, a]
        numpy.testing.assert_allclose(c_ab, c_ba.conj(), rtol=1e-13)


if __name__ == "__main__":
    main()
