"""Write log odds from the matrix ring method on lhe inputs.

usage:

python rings_lodds.py FILENAME


e.g.

python rings_lodds.py /home/tombs/Cambridge/lester_flips/sm_atlas_200k.lhe

"""
import argparse
import math
import sys
from numbers import Integral

import element
import lhe
import numpy
import rings_lib
import standalone.liv_axial1 as pv_msme
from rings_lib import parity, rotxy, rotyz_pi

NGRID = 36


def main():
    process = element.Process(pv_msme)

    assert len(sys.argv) == 2
    filename_in = sys.argv[1]
    assert filename_in[-4:] == ".lhe"
    filename_out = filename_in[:-4] + "_rings_lodds.csv"
    write_rings_lodds(filename_in, process, filename_out)


def write_rings_lodds(filename_in, process, filename_out):
    """Write log odds for events in filename_in to (csv) filename_out."""

    def get_lodds(pdg_id, momentum, helicity):
        # all-gluon processes are symmetric here
        if all(pdg_id) == 21:
            return 0.0
        helicity_sum = True
        if helicity_sum:
            hel = -1
        else:
            hel = helicity
        return ring_lodds(process, pdg_id, momentum, hel, NGRID)

    lodds = []
    print("Evaluating rings lodds...")
    for i, event in enumerate(lhe.read_lhe(filename_in)):
        stats = rings_lib.event_stats(event)
        lodds.append(get_lodds(*stats))
        if (i + 1) % 100 == 0:
            print(f"\r {i + 1:7d} / ???????", end="")
    print(f"\r {i + 1:7d} / {i + 1:7d}")

    numpy.savetxt(filename_out, lodds, fmt="%.8e")
    print(f"Wrote {filename_out!r}")


def ring_lodds(process, pdg_id, momentum, helicity, ngrid=36):
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

    for theta in numpy.linspace(0, 2 * numpy.pi, ngrid, endpoint=False):
        original += call(rotxy(momentum, theta)) + call(rotxy(momentum_flip, theta))
        inverted += call(rotxy(pomentum, theta)) + call(rotxy(pomentum_flip, theta))

    if original == 0 or inverted == 0:
        return 0.0

    return math.log(original / inverted)


if __name__ == "__main__":
    main()
