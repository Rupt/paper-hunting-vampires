"""
Dump serialized histograms of net parity variables.

Usage:

e.g.


python plot/hist_net.py --tag truth_net \
    --signal sm_3j_4j \
    --modelpath /home/tombs/Downloads/models/jet_net_truth/ \
    --datapath /home/tombs/Downloads/truth_ktdurham200_cut/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_net.py --tag truth_net \
        --signal liv_3j_4j_${LAMBDA} \
        --modelpath /home/tombs/Downloads/models/jet_net_truth/ \
        --datapath /home/tombs/Downloads/truth_ktdurham200_cut/ \
        --outpath results/hist/liv_${LAMBDA}/
done


python plot/hist_net.py --tag reco_net \
    --signal sm_3j_4j \
    --modelpath /home/tombs/Downloads/models/jet_net_reco/ \
    --datapath /home/tombs/Downloads/reco_ktdurham200/ \
    --outpath results/hist/liv_0/

for LAMBDA in p1 p2 p3 p4 p5 p6 p7 p8 p9 1
do
    python plot/hist_net.py --tag reco_net \
        --signal liv_3j_4j_${LAMBDA} \
        --modelpath /home/tombs/Downloads/models/jet_net_reco/ \
        --datapath /home/tombs/Downloads/reco_ktdurham200/ \
        --outpath results/hist/liv_${LAMBDA}/
done


"""
import argparse
import glob
import json
import os

import hist
import jet_lib
import jet_net_lib
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
    hist_net(
        args.signal, args.modelpath, args.datapath, args.outpath, args.tag
    )


def hist_net(signal, modelpath, datapath, outpath, tag, *, verbose=True):

    net_range = (-2, 2)
    net_bins = 400

    params, meta = jet_net_lib.fit_load(os.path.join(modelpath, signal))

    net_phi = jet_net_lib.make_phi_func(params, meta)

    def parity(x, nblock=100_000):
        if len(x) < nblock:
            phi = net_phi(params, x)
        else:
            # split up to blocks to avoid overloading memory
            phi_blocks = []
            for i in range(0, len(x), nblock):
                phi_blocks.append(net_phi(params, x[i : i + nblock]))
            phi = numpy.concatenate(phi_blocks)
        return phi[phi != 0]

    func_bins_ranges = [(parity, net_bins, net_range)]

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
