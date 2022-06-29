"""
Serialize data for rings plots.

Usage:

python rings_paper_dump.py

"""
import element
import numpy
import rings_lib


def main():
    import standalone.liv_axial1 as pv_msme
    import standalone.standard_jjj as sm
    import standalone.trial_c03_c30_1_1 as pv_msme_c03_c30_1_1
    import standalone.trial_c03_c30_m2_2 as pv_msme_c03_c30_m2_2
    import standalone.trial_c13_1 as pv_msme_c13_1
    import standalone.trial_diag_100m1 as pv_msme_diag_100m1

    events = {
        "liv_1_110": event_liv_1_110(),
        "liv_1_0": event_liv_1_0(),
    }

    processes = {
        "sm": element.Process(sm),
        "pv_msme": element.Process(pv_msme),
        "pv_msme_c03_c30_1_1": element.Process(pv_msme_c03_c30_1_1),
        "pv_msme_c03_c30_m2_2": element.Process(pv_msme_c03_c30_m2_2),
        "pv_msme_c13_1": element.Process(pv_msme_c13_1),
        "pv_msme_diag_100m1": element.Process(pv_msme_diag_100m1),
    }

    for event_name, event in events.items():
        for process_name, process in processes.items():
            name = event_name + "_" + process_name
            rings_lib.dump(
                name,
                process,
                *event,
                ngrid=256,
                helicity_sum=True,
            )


def event_liv_1_110():
    flavors = [2, -2, 21, 2, -2]  # u u~ > g u u~
    helicities = [1, 1, -1, 1, 1]
    momenta = make_momenta(
        [100, 250, 250],
        [200, 250, 400],
        -300,
    )
    assert_cuts(momenta)

    return flavors, momenta, helicities


def event_liv_1_0():
    flavors = [1, 2, 21, 2, 1]  # d u > g u d
    helicities = [-1, 1, -1, 1, -1]
    momenta = make_momenta(
        [200, 400, -600],
        [-200, -150, 200],
        1200,
    )
    assert_cuts(momenta)

    return flavors, momenta, helicities


# utilities


def make_momenta(j1, j2, j3z, *, debug=False):
    """Return an array for p1 p2 -> j1 j2 j3 4-momenta, for given 3-momenta.

    j are massless, p are partons in z, j3 is set to conserve momentum.
    """
    j1 = numpy.asarray(j1)
    j2 = numpy.asarray(j2)
    j3 = -j1 - j2
    j3[2] = j3z

    e1 = numpy.linalg.norm(j1)
    e2 = numpy.linalg.norm(j2)
    e3 = numpy.linalg.norm(j3)

    # choose to be in com rest frame
    tote = e1 + e2 + e3
    totz = j1[2] + j2[2] + j3[2]
    ep1 = 0.5 * (tote + totz)
    ep2 = 0.5 * (tote - totz)
    p1 = [ep1, 0, 0, ep1]
    p2 = [ep2, 0, 0, -ep2]

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


def assert_cuts(momenta):
    assert momenta.shape == (5, 4)
    j1, j2, j3 = momenta[2:]

    min_pt = min(pt(j1), pt(j2), pt(j3))
    min_delta_r = min(
        delta_r(j1, j2),
        delta_r(j1, j3),
        delta_r(j2, j3),
    )

    assert min_pt > 220, min_pt
    assert min_delta_r > 0.4, min_delta_r


def pt(p):
    return (p[1] ** 2 + p[2] ** 2) ** 0.5


def delta_r(p1, p2):
    deta = abs(eta(p1) - eta(p2))

    phi1 = numpy.arctan2(p1[2], p1[1])
    phi2 = numpy.arctan2(p2[2], p2[1])
    dphi = min(abs(phi1 - phi2), 2 * numpy.pi - abs(phi1 - phi2))

    return (dphi**2 + deta**2) ** 0.5


def eta(p):
    absp = p[1:].dot(p[1:]) ** 0.5
    return numpy.arctanh(p[2] / absp)


if __name__ == "__main__":
    main()
