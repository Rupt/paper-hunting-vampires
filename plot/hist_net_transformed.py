"""
Dump serialized hists of a net parity variable quantile transformed.

Usage:

e.g.

python plot/hist_net_transformed.py --tag reco_net \
    --signal liv_3j_4j_1 \
    --modelpath /home/tombs/Downloads/models/jet_net_reco/ \
    --datapath /home/tombs/Downloads/reco_ktdurham200/ \
    --outpath results/hist/liv_1/


"""
import argparse
import glob
import gzip
import json
import os

import hist
import jet_lib
import jet_net_lib
import joblib
import numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=str, required=True)
    parser.add_argument("--modelpath", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    hist_net_transformed(
        args.signal, args.modelpath, args.datapath, args.outpath, args.tag
    )


def hist_net_transformed(
    signal, modelpath, datapath, outpath, tag, *, verbose=True
):

    unit_range = (-1, 1)
    unit_bins = 2000

    params, meta = jet_net_lib.fit_load(os.path.join(modelpath, signal))

    net_phi = jet_net_lib.make_phi_func(params, meta)

    transformer = joblib.load(gzip.open("results/transformer_net.dat.gz"))

    def parity_transformed(x):
        phi = net_phi(params, x)
        phi = phi[phi != 0]
        t = transformer.transform(abs(phi).reshape(-1, 1)).ravel()
        return numpy.copysign(t, phi)

    func_bins_ranges = [(parity_transformed, unit_bins, unit_range)]

    # input preparation
    def gen_arrays(inpaths):
        assert len(inpaths) == 1
        path = inpaths[0]

        momenta = jet_lib.load_invariant_momenta(path)

        # batch to avoid out-of-memory errors
        step = 1_000_000
        for i in range(0, len(momenta), step):
            yield momenta[i : i + step]

    inpaths = sorted(
        glob.glob(os.path.join(datapath, signal, "private_test", "*.h5"))
    )

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
