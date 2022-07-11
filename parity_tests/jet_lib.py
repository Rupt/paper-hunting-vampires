"""Tools for analysing jet-momenta data."""
import glob
import json
import os

import h5py
import numpy

from gen_data.invariant_coordinates import invariant_momenta

PARAMS_NAME = "params.dat.gz"
MODEL_NAME = "model.dat.gz"
META_NAME = "meta.json"
TEST_NAME = "test.json"
PRIVATE_TEST_NAME = "private_test.json"

N_PARTICLES = 4


# data preparation


def load_invariant_momenta(
    pathglob, *, nmax=None, nparticles=N_PARTICLES, with_truth=False
):
    """Return momenta loaded from pathglob in invariant coordinates.

    Input data are h5 files with arrays at ["events"]
    with 4momenta in shape [nevents, 4 * nparticles]
    where the four components are (px, py, pz, e).
    """
    assert nparticles >= 1

    # sort to ensure stability
    paths = sorted(glob.glob(pathglob))
    assert paths

    def files():
        for path in paths:
            with h5py.File(path, "r") as file_:
                yield file_

    # learn size we need
    nevents = sum(len(file_["events"]) for file_ in files())

    # fill that size with momenta, losing one from leading y component
    data = numpy.empty((nevents, 3 * nparticles - 1), dtype=numpy.float32)
    if with_truth:
        flav = numpy.empty((nevents, nparticles), dtype=numpy.int8)
        hel = numpy.empty((nevents, nparticles), dtype=numpy.int8)

    momentum_indices = []
    for i in range(nparticles):
        momentum_indices += [4 * i + 0, 4 * i + 1, 4 * i + 2]

    i = 0
    for file_ in files():
        events = file_["events"]
        momenta = events[:, momentum_indices]
        data[i : i + len(events)] = invariant_momenta(momenta)

        if with_truth:
            flav[i : i + len(events)] = file_["flavors"][:, :nparticles]
            hel[i : i + len(events)] = file_["helicities"][:, :nparticles]

        i += len(events)

        if nmax is not None and i >= nmax:
            data = data[:nmax]
            if with_truth:
                flav = flav[:nmax]
                hel = hel[:nmax]
            break

    if with_truth:
        return data, flav, hel

    return data


def load_fourmomenta(pathglob, *, nmax=None, nparticles=N_PARTICLES):
    """Return 4momenta loaded from pathglob.

    Input data are h5 files with arrays at ["events"]
    with 4momenta in shape [nevents, 4 * nparticles]
    where the four components are (px, py, pz, e).
    """
    assert nparticles >= 1

    # sort to ensure stability
    paths = sorted(glob.glob(pathglob))
    assert paths

    def files():
        for path in paths:
            with h5py.File(path, "r") as file_:
                yield file_

    # learn size we need
    nevents = sum(len(file_["events"]) for file_ in files())

    # fill that size with momenta, losing one from leading y component
    data = numpy.empty((nevents, 4 * nparticles), dtype=numpy.float32)

    i = 0
    for file_ in files():
        events = file_["events"]
        data[i : i + len(events)] = events[:, : 4 * nparticles]
        i += len(events)

        if nmax is not None and i >= nmax:
            data = data[:nmax]
            break

    return data


def parity_flip(inv_momenta):
    """Return an equivalent to invariant_momenta(-momenta)."""
    # the effect of parity is to change the sign of y components
    out = inv_momenta.copy()
    # x z x y z ...
    nmomenta = 3 * N_PARTICLES - 1
    out[:, range(3, nmomenta, 3)] *= -1
    return out


def stitch_parts(data_or_parts, with_flav, with_hel):
    if not with_flav and not with_hel:
        return data_or_parts

    data, flav, hel = data_or_parts

    parts = [data]
    if with_flav:
        parts.append(flav)
    if with_hel:
        parts.append(hel)

    return numpy.concatenate(parts, axis=1)


# serialization


def result_dump(result, dirname, *, private=False, verbose=True):
    """Serialize json data into directory dirname."""
    os.makedirs(dirname, exist_ok=True)

    if private:
        out_name = PRIVATE_TEST_NAME
    else:
        out_name = TEST_NAME

    result_name = os.path.join(dirname, out_name)

    json.dump(result, open(result_name, "w"), indent=4)
    if verbose:
        print("wrote %r" % result_name)


# display


def print_llr(meta):
    log_r = meta["log_r_val"]
    quality, quality_std = meta["best_quality"].values()
    print(f"llr     : Q*nval = {log_r:.1f}")
    print(
        f"mean llr:      Q = {1e6 * quality:.1f} +- {1e6 * quality_std:.1f} "
        "ppm"
    )


# rotation business
def load_rot(path, *, nper=None):
    if nper is None or isinstance(nper, int):
        nper = [nper] * len(paths)

    parts = []
    for hour, nper_i in zip(range(24), nper):
        pathglob = os.path.join(path, "liv_rot_%d_*.h5" % hour)
        part = load_invariant_momenta(pathglob, nmax=nper_i)
        assert len(part) == nper_i, (len(part), nper_i)

        theta = hour / 24 * (2 * numpy.pi)
        # expand to full arrays to append
        sint = numpy.full(len(part), numpy.sin(theta), dtype=part.dtype)
        cost = numpy.full(len(part), numpy.cos(theta), dtype=part.dtype)

        part = numpy.concatenate(
            [
                part,
                sint[:, numpy.newaxis],
                cost[:, numpy.newaxis],
            ],
            axis=1,
        )

        parts.append(part)

    return numpy.concatenate(parts)


def _file_hour(filepath):
    base = os.path.basename(filepath)
    hour = int(base.split("_")[2])
    assert 0 <= hour < 24
    return hour


def load_rot_xs():
    hour, xs_pre, *_ = numpy.loadtxt(
        "results/rot_xs.csv",
        skiprows=1,
        delimiter=",",
    ).T
    _, nsamples, ngenerated = numpy.loadtxt(
        "results/rot_acceptance_truth.csv",
        skiprows=1,
        delimiter=",",
    ).T
    acc = nsamples / ngenerated

    return xs_pre * acc
