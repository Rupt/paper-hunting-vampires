"""Evaluate MadGraph standalone matrix element modules."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import ModuleType

import numpy

PARAM_CARD = "param_card.dat"

# strong coupling constant matched to nn23lo1
DEFAULT_ALPHA_S = 0.130
# i don't understand this one
DEFAULT_PROC_ID = -1
# for loops only, which we don't support
DEFAULT_SCALE2 = 0


class PDG:
    """Some PDG Monte Carlo ID numbers.

    https://pdg.lbl.gov/2019/reviews/rpp2018-rev-monte-carlo-numbering.pdf
    """

    d = 1
    u = 2
    s = 3
    c = 4
    b = 5
    t = 6
    g = 21
    y = 22
    a = 22
    Z = 23
    W = 24

    @staticmethod
    def is_fermion(pid):
        """Return True if pid is a fermion PDG ID, and False otherwise."""
        # Update me if adding new particles
        return abs(pid) < 20


@dataclass
class Process:
    """Configure and evaluate a matrix element module."""

    # a MadGraph standalone module
    module: ModuleType
    # for given in particles, out particles are in fixed orders
    # map sorted parameters to the argsort of the target order
    in_to_sorted_to_argsort: Mapping = field(init=False)

    def __post_init__(self):
        self.module.initialise(PARAM_CARD)

        inout = defaultdict(dict)
        for pdg_id in self.module.get_pdg_order()[0]:
            sorted_ = tuple(sorted(pdg_id[2:]))
            # first argsort gives locations to insert
            # second argsort gives reorder such that insertion goes 0, 1, 2...
            # not sure this is optimal!
            argsort = numpy.argsort(numpy.argsort(pdg_id[2:]))
            previous = inout[tuple(pdg_id[:2])].setdefault(sorted_, argsort)
            assert numpy.array_equal(previous, argsort)

        self.in_to_sorted_to_argsort = dict(inout)

    def __call__(self, pdg_id, momentum, helicity=-1, alpha_s=DEFAULT_ALPHA_S):
        """Return a squared matrix element evaluated by module.

        Arguments:
            pdg_id: PDG id codes for the particles
            momentum: particle momenta shape (nparticles, 4)
            helicity: helicity codes for the particles (or -1 for sum over all)
        """
        pdg_id, momentum, ihel = self.canon(pdg_id, momentum, helicity)
        return self.unchecked(pdg_id, momentum, ihel)

    def unchecked(self, pdg_id, momentum, ihel, alpha_s=DEFAULT_ALPHA_S):
        """Return a squared matrix element without standardizing arguments."""
        return self.module.smatrixhel(
            pdg_id,
            DEFAULT_PROC_ID,
            momentum.T,
            alpha_s,
            DEFAULT_SCALE2,
            ihel,
        )

    def canon(self, pdg_id, momentum, helicity):
        """Return the arguments in canonical form. May be views, not copies."""
        pdg_id = numpy.array(pdg_id)
        isort = self.isort(pdg_id)

        pdg_id = pdg_id[isort]
        momentum = numpy.asarray(momentum)[isort]

        if isinstance(helicity, Integral):
            ihel = helicity
        else:
            helicity = numpy.asarray(helicity)[isort]
            ihel = helicity_index(pdg_id, helicity)

        return pdg_id, momentum, ihel

    def isort(self, pdg_id):
        """Return indices which put inputs into canonical order."""
        in_ = tuple(pdg_id[:2])
        if in_ in self.in_to_sorted_to_argsort:
            insort = [0, 1]
        else:
            in_ = in_[::-1]
            assert in_ in self.in_to_sorted_to_argsort
            insort = [1, 0]

        out = tuple(pdg_id[2:])
        sorted_ = tuple(sorted(out))
        argsort = self.in_to_sorted_to_argsort[in_][sorted_]
        outsort = numpy.argsort(out)[argsort]

        return numpy.concatenate([insort, outsort + 2])


def helicity_index(pdg_id, helicity):
    """Return the helicity index for given helicities and ids.

    Each helicity is +1 or -1.
    Each pdg_id is nonzero signed integer.

    The index is a binary encoding of the helicities.

    Incoming fermion have negated helicities.
    Antiparticles have negated helicities.
    """
    ihel = 0
    for i, (pid, hel) in enumerate(zip(pdg_id, helicity)):
        if i < 2 and PDG.is_fermion(pid):
            hel = -hel
        ihel = (ihel << 1) | (hel * pid > 0)
    # one-counting index for Fortran
    return ihel + 1


# testing


def test_ihel():
    def test(pdg_id, helicity, ihel_chk):
        ihel_test = helicity_index(pdg_id, helicity)
        assert ihel_test == ihel_chk, (
            f"ihel_test == {ihel_test} but ihel_chk == {ihel_chk} "
            f"for pdg_id == {pdg_id}, helicity == {helicity}"
        )

    pdg_id = [PDG.u, -PDG.u, PDG.g, PDG.g, PDG.g]
    checks = [
        ([1, -1, -1, -1, -1], 1),
        ([1, 1, -1, -1, -1], 9),
        ([1, 1, -1, -1, 1], 10),
        ([1, 1, -1, 1, -1], 11),
        ([-1, -1, -1, 1, 1], 20),
        ([-1, 1, 1, 1, -1], 31),
        ([-1, 1, 1, 1, 1], 32),
    ]
    for helicity, ihel_chk in checks:
        test(pdg_id, helicity, ihel_chk)

    pdg_id = [PDG.u, -PDG.u, PDG.g, PDG.u, -PDG.u]
    checks = [
        ([1, -1, -1, -1, -1], 2),
        ([1, -1, 1, -1, -1], 6),
        ([1, -1, 1, 1, 1], 7),
        ([1, 1, 1, -1, 1], 13),
        ([-1, -1, -1, -1, 1], 17),
        ([-1, -1, -1, 1, 1], 19),
        ([-1, 1, 1, -1, 1], 29),
    ]
    for helicity, ihel_chk in checks:
        test(pdg_id, helicity, ihel_chk)

    pdg_id = [PDG.g, -PDG.d, PDG.d, -PDG.d, -PDG.d]
    checks = [
        ([-1, -1, -1, 1, 1], 1),
        ([-1, -1, -1, 1, -1], 2),
        ([-1, -1, -1, -1, 1], 3),
        ([-1, -1, 1, 1, 1], 5),
        ([-1, 1, -1, 1, -1], 10),
        ([-1, 1, 1, -1, 1], 15),
        ([1, -1, -1, -1, -1], 20),
        ([1, 1, -1, 1, 1], 25),
        ([1, 1, 1, 1, -1], 30),
        ([1, 1, 1, -1, -1], 32),
    ]
    for helicity, ihel_chk in checks:
        test(pdg_id, helicity, ihel_chk)

    pdg_id = [PDG.g, PDG.g, PDG.g, PDG.g, PDG.g]
    checks = [
        ([-1, -1, -1, -1, -1], 1),
        ([-1, -1, -1, -1, 1], 2),
        ([1, -1, -1, 1, 1], 20),
        ([1, 1, 1, 1, 1], 32),
    ]
    for helicity, ihel_chk in checks:
        test(pdg_id, helicity, ihel_chk)

    pdg_id = [PDG.u, PDG.d, PDG.g, PDG.u, PDG.d]
    checks = [
        ([1, 1, 1, 1, -1], 7),
        ([1, 1, 1, 1, 1], 8),
        ([1, -1, 1, 1, -1], 15),
        ([1, -1, 1, 1, 1], 16),
        ([-1, 1, -1, -1, -1], 17),
        ([-1, -1, 1, 1, -1], 31),
        ([-1, -1, 1, 1, 1], 32),
    ]
    for helicity, ihel_chk in checks:
        test(pdg_id, helicity, ihel_chk)


if __name__ == "__main__":
    test_ihel()
