"""
Dump serialized histograms of certain alpha variable.

Equation (2.2) of
https://link.springer.com/article/10.1007/JHEP12(2019)120


Usage:

e.g.


python plot/hist_alpha.py --tag truth \
    --inpath /home/tombs/Downloads/truth_ktdurham200_cut/sm_3j_4j/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_alpha.py --tag truth \
        --inpath /home/tombs/Downloads/truth_ktdurham200_cut/liv_3j_4j_${LAMBDA}/ \
        --outpath results/hist/liv_${LAMBDA}/
done


python plot/hist_alpha.py --tag reco \
    --inpath /home/tombs/Downloads/reco_ktdurham200/sm_3j_4j/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_alpha.py --tag reco \
        --inpath /home/tombs/Downloads/reco_ktdurham200/liv_3j_4j_${LAMBDA}/ \
        --outpath results/hist/liv_${LAMBDA}/
done

"""
import argparse
import glob
import gzip
import json
import os

import h5py
import hist
import joblib
import numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    hist_alpha(args.inpath, args.outpath, args.tag)


def hist_alpha(inpath, outpath, tag, *, verbose=True):
    # we can always combine bins later: choose fine grained on nice boundaries
    alpha_range = (-1.6, 1.6)
    alpha_bins = 32

    def alpha(ps):
        a = alpha_rename(ps)
        return a[a != 0]

    func_bins_ranges = [
        (alpha, alpha_bins, alpha_range),
    ]

    # input preparation

    def gen_arrays(inpaths):
        for path in inpaths:
            with h5py.File(path, "r") as file_:
                events = file_["events"][:]
            yield events

    inpaths = sorted(glob.glob(os.path.join(inpath, "private_test", "*.h5")))

    results = hist.histogram(func_bins_ranges, gen_arrays(inpaths))

    for (func, _, _), result in zip(func_bins_ranges, results):
        funcname = func.__name__
        filename = os.path.join(outpath, tag + "_" + funcname + ".json")
        with open(filename, "w") as file_:
            json.dump(result, file_, indent=1)
        if verbose:
            print("wrote %r" % filename)


# event variables


def alpha(ps):
    pj1 = ps[:, 4 * 0 : 4 * 0 + 3]
    pj2 = ps[:, 4 * 1 : 4 * 1 + 3]
    pj3 = ps[:, 4 * 2 : 4 * 2 + 3]

    left = numpy.cross(pj1, pj2)
    left /= numpy.linalg.norm(left, axis=1).reshape(-1, 1)

    right = pj3 / numpy.linalg.norm(pj3, axis=1).reshape(-1, 1)

    sinalpha = numpy.einsum("ij,ij->i", left, right)

    return numpy.arcsin(sinalpha)


# hack to avoid name clash
alpha_rename = alpha


if __name__ == "__main__":
    main()
