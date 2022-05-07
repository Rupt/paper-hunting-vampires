"""
Demonstrate madgraph standalone matrix element calculation.

Borrowing from:

https://answers.launchpad.net/mg5amcnlo/+question/658745

https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-4

https://pdg.lbl.gov/2019/reviews/rpp2018-rev-monte-carlo-numbering.pdf


Usage:

python demo_violation.py

"""
import numpy
import standalone.liv_random_jjj as liv_random
import standalone.liv_zero_jjj as liv
import standalone.standard_jjj as standard
from element import PDG, Process


def main():
    print("standard")
    demo(standard)
    print()
    print("liv zero")
    demo(liv)
    print()
    print("liv random")
    demo(liv_random)


def demo(module):
    """Print some module matrix element values."""
    me2 = Process(module)

    # PDG IDs
    gg_guu = [
        # g g
        PDG.g,
        PDG.g,
        # -> g u u~
        PDG.g,
        PDG.u,
        -PDG.u,
    ]

    uu_gdd = [
        # u u~
        PDG.u,
        -PDG.u,
        # -> g d d~
        PDG.g,
        PDG.d,
        -PDG.d,
    ]

    gu_ggu = [
        # g u
        PDG.g,
        PDG.u,
        # -> g g u
        PDG.g,
        PDG.g,
        PDG.u,
    ]

    # construct example momenta and output results
    mom = make_momenta([100, 0, 10], [0, 200, -10])
    print("g g > g u u~")
    print("original %.4e" % me2(gg_guu, mom))
    print("flipped  %.4e" % me2(gg_guu, flip(mom)))
    print("u u~ > g d d~")
    print("original %.4e" % me2(uu_gdd, mom))
    print("flipped  %.4e" % me2(uu_gdd, flip(mom)))
    print("g u > g g u")
    print("original %.4e" % me2(gu_ggu, mom))
    print("flipped  %.4e" % me2(gu_ggu, flip(mom)))

    print()

    mom = make_momenta([100, 100, 300], [100, -100, 300])
    print("g g > g u u~")
    print("original %.4e" % me2(gg_guu, mom))
    print("flipped  %.4e" % me2(gg_guu, flip(mom)))
    print("u u~ > g d d~")
    print("original %.4e" % me2(uu_gdd, mom))
    print("flipped  %.4e" % me2(uu_gdd, flip(mom)))
    print("g u > g g u")
    print("original %.4e" % me2(gu_ggu, mom))
    print("flipped  %.4e" % me2(gu_ggu, flip(mom)))


def make_momenta(j1, j2, *, debug=True):
    """Return an array for p1 p2 -> j1 j2 j3 4-momenta, for given 3-momenta.

    j are massless, p are partons in z, j3 is set to conserve momentum.
    """
    j1 = numpy.asarray(j1)
    j2 = numpy.asarray(j2)
    j3 = -j1 - j2

    e1 = numpy.linalg.norm(j1)
    e2 = numpy.linalg.norm(j2)
    e3 = numpy.linalg.norm(j3)

    # choose to be in com rest frame
    halfe = 0.5 * (e1 + e2 + e3)
    p1 = [halfe, 0, 0, halfe]
    p2 = [halfe, 0, 0, -halfe]

    mom = numpy.array(
        [
            p1,
            p2,
            [e1, *j1],
            [e2, *j2],
            [e3, *j3],
        ]
    )

    if debug:
        # check conservation
        numpy.testing.assert_allclose(mom[:2].sum(axis=0), mom[2:].sum(axis=0))

    return mom


def flip(mom):
    """Return a copy of 4momenta mom with parity flipped."""
    new = mom.copy()
    new[..., 1:] = -new[..., 1:]
    return new


if __name__ == "__main__":
    main()
