"""Inspect matrix element values as symmetries are explored.

Example setup:

conda activate symbdt
cd /home/tombs/Cambridge/lester_flips/symmetry_violation/liv

python configure.py parameter/axial1.json MG5_aMC_v3_3_0/models/liv_axial1

python madcontrol.py output_standalone standalone/standard_jjj

python madcontrol.py output_standalone standalone/liv_axial1 --model liv_axial1


Usage:

python rings.py


"""
import json
import math
import os
from numbers import Integral

import element
import lhe
import numpy

PDG_ID_TO_STRING = {
    -2: "u~",
    -1: "d~",
    -3: "s~",
    -4: "c~",
    -5: "b~",
    -6: "t~",
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    21: "g",
    23: "Z",
}


def theta_elements(thetas, process, pdg_id, momentum, helicity):
    """Return arrays for process at rotated and parity flipped momenta."""
    pdg_id = numpy.asarray(pdg_id)
    momentum = numpy.asarray(momentum)
    # presort to accelerate calculations
    isort = process.isort(pdg_id)
    pdg_id = pdg_id[isort]
    momentum = momentum[isort]

    if helicity == -1:
        ihel = -1
    else:
        helicity = helicity[isort]
        ihel = element.helicity_index(pdg_id, helicity)

    # prepare transformed versions
    momentum_flip = rotyz_pi(momentum)
    pmomentum = parity(momentum)
    pmomentum_flip = parity(rotyz_pi(momentum))

    # original
    ring = numpy.empty(len(thetas))
    ring_flip = numpy.empty(len(thetas))
    # parity inverted
    pring = numpy.empty(len(thetas))
    pring_flip = numpy.empty(len(thetas))

    def call(mom):
        return process.unchecked(pdg_id, mom, ihel)

    for i, theta in enumerate(thetas):
        ring[i] = call(rotxy(momentum, theta))
        ring_flip[i] = call(rotxy(momentum_flip, theta))
        pring[i] = call(rotxy(pmomentum, theta))
        pring_flip[i] = call(rotxy(pmomentum_flip, theta))

    return ring, ring_flip, pring, pring_flip


def event_stats(event):
    """Return some useful properties of an event in arrays."""
    nparticles = len(event.particles)
    pdg_id = numpy.empty(nparticles)
    momentum = numpy.empty([nparticles, 4])
    helicity = numpy.empty(nparticles)

    for i, p in enumerate(event.particles):
        pdg_id[i] = p.pdg_id
        momentum[i] = [p.e, p.px, p.py, p.pz]
        helicity[i] = p.helicity

    return pdg_id, momentum, helicity


def lodds(process, pdg_id, momentum, helicity, ngrid=36):
    """Return a `which is real?' log odds.

    Standalone version.
    """
    # special case: pure gluon processes are symmetric
    if all(pdg_id) == 21:
        return 0.0

    # presort to accelerate calculations
    isort = process.isort(pdg_id)
    pdg_id = pdg_id[isort]
    momentum = momentum[isort]
    if isinstance(helicity, Integral):
        ihel = helicity
    else:
        helicity = helicity[isort]
        ihel = element.helicity_index(pdg_id, helicity)

    # prepare transformed versions
    momentum_flip = rotyz_pi(momentum)
    pomentum = parity(momentum)
    pomentum_flip = rotyz_pi(pomentum)

    original = 0.0
    inverted = 0.0

    def call(mom):
        return process.unchecked(pdg_id, mom, ihel)

    for theta in numpy.linspace(0, 2 * math.pi, ngrid, endpoint=False):
        original += call(rotxy(momentum, theta)) + call(rotxy(momentum_flip, theta))
        inverted += call(rotxy(pomentum, theta)) + call(rotxy(pomentum_flip, theta))

    if original == 0 or inverted == 0:
        return 0.0

    return math.log(original / inverted)


def rotxy(momentum, theta):
    """Return a copy of momentum rotated in the xy plane by theta."""
    assert momentum.shape == (len(momentum), 4)
    c = math.cos(theta)
    s = math.sin(theta)
    return momentum.dot(
        [
            [1, 0, 0, 0],
            [0, c, s, 0],
            [0, -s, c, 0],
            [0, 0, 0, 1],
        ]
    )


def rotyz_pi(momentum):
    """Return a copy of momentum rotated by pi about the x axis."""
    m = momentum.copy()
    m[:, 2:] *= -1
    return m


def parity(momentum):
    """Return a copy of momentum with parity inverted."""
    m = momentum.copy()
    m[:, 1:] *= -1
    return m


def xy_from(theta, radius):
    """Return cartesian versions of polar coordinates."""
    return numpy.array(
        [
            numpy.cos(theta) * radius,
            numpy.sin(theta) * radius,
        ]
    )


# testing


def test_rotxy():
    a = numpy.array(
        [
            [7, 0, 1, 7],
        ]
    )

    for ai in a:
        e, x, y, z = ai

        angle_target = [
            (math.pi, [e, -x, -y, z]),
            (math.pi / 2, [e, -y, x, z]),
            (math.pi / 4, [e, (x - y) * 2 ** -0.5, (y + x) * 2 ** -0.5, z]),
        ]
        for angle, target in angle_target:
            adash = rotxy(a, angle)
            numpy.testing.assert_allclose(adash, [target], atol=1e-13)


def test_rotyz_pi():
    a = numpy.array(
        [
            [7, 0, 1, 7],
            [6, 1, 2, 3],
            [1.23, 4.56, -12.0, 6.77],
        ]
    )

    ayzx = a[:, [0, 2, 3, 1]]
    target = rotxy(ayzx, math.pi)[:, [0, 3, 1, 2]]
    numpy.testing.assert_allclose(rotyz_pi(a), target, atol=1e-13)


# serialization
def dump(
    name,
    process,
    flavors,
    momenta,
    helicities,
    ngrid=128,
    helicity_sum=True,
):
    flavors = numpy.asarray(flavors)
    momenta = numpy.asarray(momenta)
    helicities = numpy.asarray(helicities)
    ngrid = int(ngrid)
    helicity_sum = bool(helicity_sum)

    thetas = numpy.linspace(0, 2 * math.pi, ngrid)
    ring, ring_flip, pring, pring_flip = theta_elements(
        thetas,
        process,
        flavors,
        momenta,
        -1 if helicity_sum else helicities,
    )

    tree = {
        "name": name,
        "flavors": flavors.tolist(),
        "momenta": momenta.tolist(),
        "helicities": helicities.tolist(),
        "ngrid": ngrid,
        "helicity_sum": helicity_sum,
        "ring": ring.tolist(),
        "ring_flip": ring_flip.tolist(),
        "pring": pring.tolist(),
        "pring_flip": pring_flip.tolist(),
    }

    filename = "rings_%s.json" % name
    with open(filename, "w") as file_:
        json.dump(tree, file_, indent=4)
    print("wrote %r" % filename)
