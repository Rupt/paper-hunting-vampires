"""
Dump serialized histograms of certain jet properties.

Usage:

e.g.


python plot/hist_jets.py --tag truth \
    --inpath /home/tombs/Downloads/truth_ktdurham200_cut/sm_3j_4j/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_jets.py --tag truth \
        --inpath /home/tombs/Downloads/truth_ktdurham200_cut/liv_3j_4j_${LAMBDA}/ \
        --outpath results/hist/liv_${LAMBDA}/
done


python plot/hist_jets.py --tag reco \
    --inpath /home/tombs/Downloads/reco_ktdurham200/sm_3j_4j/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_jets.py --tag reco \
        --inpath /home/tombs/Downloads/reco_ktdurham200/liv_3j_4j_${LAMBDA}/ \
        --outpath results/hist/liv_${LAMBDA}/
done

"""
import argparse
import glob
import json
import os

import h5py
import hist
import numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    hist_jets(args.inpath, args.outpath, args.tag)


def hist_jets(inpath, outpath, tag, *, verbose=True):
    # we can always combine bins later: choose fine grained on nice boundaries
    pt_range = (0, 5_000)
    pt_bins = 500

    ht_range = (0, 10_000)
    ht_bins = 1_000

    eta_range = (-5, 5)
    eta_bins = 100

    func_bins_ranges = [
        (ht, ht_bins, ht_range),
        (ht_3j, ht_bins, ht_range),
        (ht_4j, ht_bins, ht_range),
        (pt_a, pt_bins, pt_range),
        (pt_b, pt_bins, pt_range),
        (pt_c, pt_bins, pt_range),
        (pt_d, pt_bins, pt_range),
        (eta_a, eta_bins, eta_range),
        (eta_b, eta_bins, eta_range),
        (eta_c, eta_bins, eta_range),
        (eta_d, eta_bins, eta_range),
    ]

    # input preparation

    def gen_arrays(inpaths):
        for path in inpaths:
            with h5py.File(path, "r") as file_:
                events = file_["events"][:]
            yield events

    inpaths = sorted(
        glob.glob(os.path.join(inpath, "train", "*.h5"))
        + glob.glob(os.path.join(inpath, "test", "*.h5"))
    )

    results = hist.histogram(func_bins_ranges, gen_arrays(inpaths))

    for (func, _, _), result in zip(func_bins_ranges, results):
        funcname = func.__name__
        filename = os.path.join(outpath, tag + "_" + funcname + ".json")
        with open(filename, "w") as file_:
            json.dump(result, file_, indent=1)
        if verbose:
            print("wrote %r" % filename)


def select_njets(ps, njets):
    # px is zero iff a jet is missing
    # require the nth jet is not missing
    # and the n+1th jet is missing
    ijet = njets - 1

    # order@ px py pz e
    mask = ps[:, 4 * ijet + 0] != 0

    if 4 * ijet + 4 < ps.shape[1]:
        mask &= ps[:, 4 * ijet + 4] == 0

    return ps[mask]


# event variables


def pt(ps, iparticle):
    x = ps[:, 4 * iparticle]
    y = ps[:, 4 * iparticle + 1]
    return (x ** 2 + y ** 2) ** 0.5


def ht(ps):
    assert ps.shape[1] % 4 == 0
    out = pt(ps, 0)
    for i in range(1, ps.shape[1] // 4):
        out += pt(ps, i)
    return out


def ht_3j(ps):
    return ht(select_njets(ps, 3))


def ht_4j(ps):
    return ht(select_njets(ps, 4))


def pt_a(ps):
    return pt(ps, 0)


def pt_b(ps):
    return pt(ps, 1)


def pt_c(ps):
    return pt(ps, 2)


def pt_d(ps):
    r = pt(ps, 3)
    return r[r != 0]


def eta(ps, iparticle):
    x = ps[:, 4 * iparticle + 0]
    y = ps[:, 4 * iparticle + 1]
    z = ps[:, 4 * iparticle + 2]
    p = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return numpy.arctanh(z / p)


def eta_a(ps):
    return eta(ps, 0)


def eta_b(ps):
    return eta(ps, 1)


def eta_c(ps):
    return eta(ps, 2)


def eta_d(ps):
    nonzero = pt(ps, 3) != 0
    return eta(ps[nonzero], 3)


if __name__ == "__main__":
    main()
