"""
Prepare QuantileTransformers for phi plots.


Usage:

e.g.

python plot/quantiles.py \
    --netpath /home/tombs/Downloads/models/jet_net_reco/ \
    --datapath /home/tombs/Downloads/reco_ktdurham200/ \
    --outpath results/


"""
import argparse
import glob
import gzip
import json
import os

import hist_alpha
import jet_lib
import jet_net_lib
import joblib
import numpy
from sklearn.preprocessing import QuantileTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netpath", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)
    dump_quantile_transformer(args.netpath, args.datapath, args.outpath)


def dump_quantile_transformer(netpath, datapath, outpath):
    signal = "liv_3j_4j_1"
    (inpath,) = sorted(glob.glob(os.path.join(datapath, signal, "test", "*.h5")))

    net_transformer = do_net(signal, netpath, inpath)
    dump(net_transformer, os.path.join(outpath, "transformer_net.dat.gz"))

    alpha_transformer = do_alpha(signal, inpath)
    dump(alpha_transformer, os.path.join(outpath, "transformer_alpha.dat.gz"))


def do_net(signal, netpath, inpath):
    params, meta = jet_net_lib.fit_load(os.path.join(netpath, signal))

    net_phi = jet_net_lib.make_phi_func(params, meta)

    def parity(x):
        phi = net_phi(params, x)
        return phi[phi != 0]

    x = jet_lib.load_invariant_momenta(inpath, nmax=int(1e6))
    p = parity(x)

    transformer = default_transformer()
    transformer.fit(abs(p).reshape(-1, 1))
    return transformer


def do_alpha(signal, inpath):
    x = jet_lib.load_fourmomenta(inpath, nmax=int(1e6))
    a = hist_alpha.alpha(x)
    a = a[a != 0]

    transformer = default_transformer()
    transformer.fit(abs(a).reshape(-1, 1))
    return transformer


# utilities


def default_transformer():
    return QuantileTransformer(
        n_quantiles=1000,
        output_distribution="uniform",
        subsample=1e6,
        random_state=123,
    )


def dump(transformer, path, *, verbose=True):
    joblib.dump(transformer, gzip.open(path, "w"))
    if verbose:
        print("wrote %r" % path)


if __name__ == "__main__":
    main()
