"""Test serialized nets against independent data on a rotating planet.

Usage:

e.g.

python parity_tests/jet_net_test_rot.py \
    --datapath /home/tombs/Downloads/liv_rot_smol/ \
    --outpath out_test_rot \
    --ntest 1_000_000

"""
import argparse
import os

import numpy
from jet_lib import load_rot, result_dump, load_rot_xs
from jet_net_lib import fit_load, net_test, zeta_20_20_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    jet_net_test(
        args.datapath,
        args.outpath,
        args.ntest,
        private=args.private,
    )


def jet_net_test(datapath, outpath, ntest, *, private=False, normalize=False):
    assert ntest is None or ntest > 0

    params, meta = fit_load(outpath)

    if private:
        test = "private_test"
    else:
        test = "test"

    xs = load_rot_xs()
    if normalize:
        xs[:] = 1
    probs = xs / xs.sum()

    rng = numpy.random.Generator(numpy.random.Philox(31415))

    path = os.path.join(datapath, test)
    nper = rng.multinomial(ntest, probs)
    test_parts = load_rot(path, nper=nper)
    rng.shuffle(test_parts)

    print(test_parts.shape)

    result = net_test(
        zeta_20_20_10,
        params,
        meta,
        test_parts,
        tag={
            "datapath": datapath,
        },
    )
    result_dump(result, outpath, private=private)



if __name__ == "__main__":
    main()
