"""Mimic boosting an event as if in the heart of madgraph."""
import lorentz
import numpy


def main():
    numpy.random.seed(3)

    ebeam = numpy.array([6500.0, 6500.0])
    xbk = numpy.array([0.123, 0.234])

    elab = ebeam * xbk

    plab = numpy.array(
        [
            [elab[0], 0, 0, elab[0]],
            [elab[1], 0, 0, -elab[1]],
            # a resonance particle
            [elab[0] + elab[1], 0, 0, elab[0] - elab[1]],
        ]
    )

    # boost to lab frame
    tote, totp = plab[:2, [0, 3]].sum(axis=0)
    beta = totp / tote

    pcom = plab.dot(lorentz.make_boost(0, 0, beta))

    print(f"{ebeam = }")
    print(f"{xbk = }")

    print("lab frame")
    print(plab)
    reslab = plab[2]
    masslab = (reslab[0] ** 2 - reslab[1:].dot(reslab[1:])) ** 0.5
    print(f"{masslab = }")
    print("center of mass frame")
    print(pcom)
    rescom = pcom[2]
    masscom = (rescom[0] ** 2 - rescom[1:].dot(rescom[1:])) ** 0.5
    print(f"{masscom = }")

    print("boost back to the lab")
    p1 = boost(pcom.T, xbk, ebeam).T
    print(p1)
    xbklab = p1[[0, 1], 0] / ebeam
    print(f"{xbklab = }")

    # flip z to check the other way
    ppcom = pcom.copy()
    ppcom[:, 3] *= -1

    print("invert z, then back to the lab again")
    pp1 = boost(ppcom.T, xbk, ebeam).T
    print(pp1)


def boost(pp, xbk, ebeam):
    """Return momenta boosted to the lab frame.

    Arranged like Fortran code in auto_dsig?.f

    Arguments:
        pp: array of 4momenta shape (4, nparticles)
        xbk: bjorken x for the incoming partons shape (2,)
        ebeam: beam energies shape (2,) (== 6500 GeV)


    We have massless incoming partons (e1, ±e1), (e2, ∓e2)
    We are in the com frame, so e1 == e2 == e
    We want the lab frame, where ebeam[i] * xbk[i] = e'

    Massless boost (active, +z direction):

        e' = (γ ± βγ)e

    let r = e' / e = ebeam[i] * xbk[i] / e;

    =>  r = γ + βγ = γ + sqrt(γ**2 - 1)
                   = sqrt((βγ)**2 + 1) - βγ

    =>  γ = (r**2 + 1) / (2r)
        |βγ| = (1 - r**2) / (2r)
    """
    r = ebeam[0] * xbk[0] / pp[0, 0]
    y = (r**2 + 1) / (2 * r)
    by = (r**2 - 1) / (2 * r)
    if pp[3, 0] < 0:
        by = -by

    p1 = pp.copy()
    for i in range(pp.shape[1]):
        p1[0, i] = y * pp[0, i] + by * pp[3, i]
        p1[3, i] = by * pp[0, i] + y * pp[3, i]

    return p1


if __name__ == "__main__":
    main()
