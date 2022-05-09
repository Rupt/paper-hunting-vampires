"""
Dump serialized histograms of bdt parity variables.

Usage:

e.g.


python plot/hist_bdt.py --tag truth_bdt \
    --signal sm_3j_4j \
    --modelpath /home/tombs/Downloads/models/jet_bdt_truth/ \
    --datapath /home/tombs/Downloads/truth_ktdurham200_cut/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_bdt.py --tag truth_bdt \
        --signal liv_3j_4j_${LAMBDA} \
        --modelpath /home/tombs/Downloads/models/jet_bdt_truth/ \
        --datapath /home/tombs/Downloads/truth_ktdurham200_cut/ \
        --outpath results/hist/liv_${LAMBDA}/
done


python plot/hist_bdt.py --tag reco_bdt \
    --signal sm_3j_4j \
    --modelpath /home/tombs/Downloads/models/jet_bdt_reco/ \
    --datapath /home/tombs/Downloads/reco_ktdurham200/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_bdt.py --tag reco_bdt \
        --signal liv_3j_4j_${LAMBDA} \
        --modelpath /home/tombs/Downloads/models/jet_bdt_reco/ \
        --datapath /home/tombs/Downloads/reco_ktdurham200/ \
        --outpath results/hist/liv_${LAMBDA}/
done


"""
import argparse
import glob
import json
import os

import hist
import jet_bdt_lib
import jet_lib

from sksym import sksym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=str, required=True)
    parser.add_argument("--modelpath", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    hist_bdt(
        args.signal, args.modelpath, args.datapath, args.outpath, args.tag
    )


def hist_bdt(signal, modelpath, datapath, outpath, tag, *, verbose=True):
    # we can always combine bins later: choose fine grained on nice boundaries
    # note that +0.0 goes to the right of zero; don't look at the central two!!
    bdt_range = (-2, 2)
    bdt_bins = 400

    model, meta = jet_bdt_lib.fit_load(os.path.join(modelpath, signal))

    which = sksym.WhichIsReal()

    def parity(x):
        test_pack = which.stack(x, [jet_lib.parity_flip(x)])
        zetas = sksym.predict_zeta(
            model,
            test_pack,
            iteration_range=(0, meta["best_iteration"]),
        )
        phi = zetas[0] - zetas[1]
        return phi[phi != 0]

    func_bins_ranges = [(parity, bdt_bins, bdt_range)]

    # input preparation
    def gen_arrays(inpaths):
        for path in inpaths:
            yield jet_lib.load_invariant_momenta(path)

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
