"""
Dump serialized histograms of certain alpha variable, quantile transformed
for LIV_1

Equation (2.2) of
https://link.springer.com/article/10.1007/JHEP12(2019)120


Usage:

e.g.

python plot/hist_alpha_transformed.py --tag reco \
    --inpath /home/tombs/Downloads/reco_ktdurham200/liv_3j_4j_1/ \
    --outpath results/hist/liv_1/


"""
import argparse
import glob
import gzip
import json
import os

import h5py
import hist
import hist_alpha
import joblib
import numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    hist_alpha_transformed(args.inpath, args.outpath, args.tag)


def hist_alpha_transformed(inpath, outpath, tag, *, verbose=True):

    unit_range = (-1, 1)
    unit_bins = 2000

    transformer = joblib.load(gzip.open("results/transformer_alpha.dat.gz"))

    def alpha_transformed(ps):
        a = hist_alpha.alpha(ps)
        a = a[a != 0]
        t = transformer.transform(abs(a).reshape(-1, 1)).ravel()
        return numpy.copysign(t, a)

    func_bins_ranges = [(alpha_transformed, unit_bins, unit_range)]

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


if __name__ == "__main__":
    main()
