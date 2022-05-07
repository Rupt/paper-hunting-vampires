"""
Demonstrate madgraph standalone matrix element calculation.

Borrowing from:

https://answers.launchpad.net/mg5amcnlo/+question/658745

https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-4

https://pdg.lbl.gov/2019/reviews/rpp2018-rev-monte-carlo-numbering.pdf


Usage:

python demo_matrix_element.py

"""
import numpy
import standalone.standard_jjj as jjj
from element import PDG, Process
from lorentz import make_boost


class pdg:
    """Some PDG Monte Carlo ID numbers."""

    d = 1
    u = 2
    s = 3
    c = 4
    b = 5
    t = 6
    g = 21


def main():
    """Print some jjj matrix element values."""
    me2 = Process(jjj)

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

    # construct example momenta and output results
    mom = make_momenta([100, 0, 10], [0, 200, -10])
    print("g g > g u u~")
    print("original %.4e" % me2(gg_guu, mom))
    print("scaled   %.4e" % me2(gg_guu, mom * 1.5))
    print("u u~ > g d d~")
    print("original %.4e" % me2(uu_gdd, mom))
    print("scaled   %.4e" % me2(uu_gdd, mom * 1.5))

    print()

    mom = make_momenta([100, 100, 300], [100, -100, 300])
    print("g g > g u u~")
    print("original %.4e" % me2(gg_guu, mom))
    print("scaled   %.4e" % me2(gg_guu, mom * 1.5))
    print("u u~ > g d d~")
    print("original %.4e" % me2(uu_gdd, mom))
    print("scaled   %.4e" % me2(uu_gdd, mom * 1.5))

    print()

    print("try boosting to new frames")
    betas = [
        (0.1, 0.2, 0.3),
        (0, 0, 1 - 1e-6),
        (0, 1 - 1e-6, 0),
        (1 - 1e-6, 0, 0),
        (0, 0.2, -0.3),
        (-0.8, 0.2, 0),
    ]
    for bx, by, bz in betas:
        print("beta = (%.3f, %.3f, %.3f)" % (bx, by, bz))
        bmom = mom @ make_boost(bx, by, bz)
        print("g g > g u u~")
        print("original %.8e" % me2(gg_guu, mom))
        print("boosted  %.8e" % me2(gg_guu, bmom))
        print("u u~ > g d d~")
        print("original %.8e" % me2(uu_gdd, mom))
        print("boosted  %.8e" % me2(uu_gdd, bmom))
        print()


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


if __name__ == "__main__":
    main()
