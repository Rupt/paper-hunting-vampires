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
import math
import os

import element
import lhe
import matplotlib
import numpy
import rings_lib
import standalone.liv_axial1 as pv_msme
import standalone.standard_jjj as standard
from matplotlib import pyplot
from rings_lib import PDG_ID_TO_STRING, xy_from

# import standalone.trial_c03_c30_1_1 as pv_msme
# import standalone.trial_c03_c30_1_m1 as pv_msme
# import standalone.trial_c03_c30_m1_0 as pv_msme
# import standalone.trial_c03_c30_m1_2 as pv_msme
# import standalone.trial_c03_c30_m2_2 as pv_msme
# import standalone.trial_c02_1 as pv_msme
# import standalone.trial_c13_1 as pv_msme
# import standalone.trial_diag_1m1m11 as pv_msme
# import standalone.trial_diag_100m1 as pv_msme

# work without graphics
matplotlib.use("Agg")

# use latex text / fonts to match document
# https://matplotlib.org/stable/tutorials/text/usetex.html
pyplot.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["Helvetica"],
    }
)


def main():
    filename = "/home/tombs/Downloads/lhe/liv_3j_4j_1_0.lhe"
    events = []

    for event in lhe.read_lhe(filename):
        # gg ggg events are symmetric; skip
        if all(p.pdg_id == 21 for p in event.particles):
            continue
        events.append(event)
        if len(events) == 200:
            break

    liv = element.Process(pv_msme)

    os.makedirs("plots", exist_ok=True)

    for i in range(len(events)):
        stats = event_stats(events[i])
        try:
            plot_rings(f"liv_{i}", liv, *stats)
        except BaseException:
            print("oopsie")
            print(*stats)

    # all SM look like circles, so don't do all
    sm = element.Process(standard)

    for i in range(len(events))[:0]:
        stats = event_stats(events[i])
        plot_rings(f"sm_{i}", sm, *stats)


def plot_rings(
    name, process, pdg_id, momentum, helicity, ngrid=128, helicity_sum=True
):
    """Make plots of matrix elements under rotations and parity."""
    title = (
        r" ".join(map(PDG_ID_TO_STRING.get, pdg_id[:2]))
        + r" > "
        + r" ".join(map(PDG_ID_TO_STRING.get, pdg_id[2:]))
        + " (%s)" % (name.split("_")[0].upper())
    )
    print()
    print(title)

    thetas = numpy.linspace(0, 2 * math.pi, ngrid)
    ring, ring_rotyz_pi, ping, ping_rotyz_pi = rings_lib.theta_elements(
        thetas,
        process,
        pdg_id,
        momentum,
        -1 if helicity_sum else helicity,
    )

    print(list(ring))

    # integrals
    total, total_err = trapz_err(ring + ring_rotyz_pi)
    potal, potal_err = trapz_err(ping + ping_rotyz_pi)
    if potal > 0:
        lodds = math.log(total / potal)
    else:
        lodds = numpy.nan
    lscale = int(math.log10(max(total, potal)))
    scale = 10 ** -lscale
    total *= scale
    total_err *= scale
    potal *= scale
    potal_err *= scale
    print(f"total            : {total: .2f} +- {total_err:.2f} x 10^{lscale}")
    print(f"total (P flipped): {potal: .2f} +- {potal_err:.2f}")
    print(f"lodds            : {lodds: .2e}")

    # plot
    figure, axis = pyplot.subplots(
        figsize=(6, 3), dpi=200, tight_layout=(0, 0, 0)
    )

    lims = []

    def xy(ring):
        x, y = xy_from(thetas, ring)
        lims.append(max(abs(x).max(), abs(y).max()))
        return x, y

    axis.plot(*xy(ring), color="b", lw=2, alpha=0.6)
    axis.plot(*xy(ring_rotyz_pi), color="b", linestyle="--", lw=2, alpha=0.5)

    axis.plot(*xy(ping), color="r", lw=2, alpha=0.6)
    axis.plot(*xy(ping_rotyz_pi), color="r", linestyle="--", lw=2, alpha=0.5)

    axis.set_aspect("equal")
    axis.scatter([0], [0], c="k", marker="o", s=2)

    axlim = max(lims) * 1.05
    # ensure nonzero limits
    axlim = max(axlim, 1e-45)
    axis.set_xlim(-axlim, axlim * 3)
    axis.set_ylim(-axlim, axlim)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)

    def latex2e(x):
        if not math.isfinite(x):
            return str(x)
        if abs(x) < 1e-15:
            return "0"
        mantissa, exponent = f"{x:.2e}".split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)
        return r"%.2f \times 10^{%d}" % (mantissa, exponent)

    # text
    axis.set_title("matrix elements: rotations vs parity")

    axis.text(
        0.55,
        0.97,
        (
            title.replace(">", r"$\rightarrow$").replace("~", "x")
            + "\n"
            + r"$A = \int\!\mathcal{M}(x) ds(x) = %.2f \pm %.2f ~~(\times 10^{%d})$"
            % (total, total_err, lscale)
            + "\n"
            + r"$B = \int\!\mathcal{M}(Px) ds(x) = %.2f \pm %.2f$"
            % (potal, potal_err)
            + "\n"
            + r"$\rightarrow \log A/B = %s$" % latex2e(lodds)
        ),
        horizontalalignment="left",
        verticalalignment="top",
        transform=axis.transAxes,
    )

    axis.legend(
        [
            r"rotate",
            r"swap beams, rotate",
            r"$P$arity, rotate",
            r"$P$arity, swap beams, rotate",
        ],
        frameon=False,
        loc="lower right",
    )

    # data table
    data = []
    for i in range(len(pdg_id)):
        part = PDG_ID_TO_STRING[pdg_id[i]]
        mom = momentum[i]
        hel = helicity[i]
        data.append(
            "%s (%.1f, %.1f, %.1f, %.1f) %+d"
            % (part.replace("~", r"x"), *mom, hel)
        )
    data = "\n".join(data)

    axis.text(
        0.55,
        0.65,
        data,
        fontsize=6,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axis.transAxes,
    )

    savefig(figure, f"plots/rings_{name}.png")


# utiltities


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


def trapz_err(y):
    """Return an integral estimate for evenly spaced points y."""
    y = numpy.asarray(y)
    tot = numpy.trapz(y, dx=1 / (len(y) - 1))
    err = abs(y[1:] - y[:-1]).mean()
    return tot, err


def savefig(figure, filename):
    figure.savefig(filename)
    print("Wrote %r" % filename)
    pyplot.close(figure)


# testing

if __name__ == "__main__":
    rings_lib.test_rotxy()
    rings_lib.test_rotyz_pi()
    main()
