"""Test serialized nets against independent data.

Usage:

e.g.

python parity_tests/jet_net_test.py \
    --datapath /home/tombs/Downloads/truth_ktdurham200/liv_3j_4j_1/ \
    --outpath out_test_3 \
    --ntest 1_000_000 \
    --with_flav --with_hel

"""
import argparse
import os

from jet_lib import load_invariant_momenta, result_dump, stitch_parts
from jet_net_lib import fit_load, net_test, zeta_100_100_d_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--with_flav", action="store_true")
    parser.add_argument("--with_hel", action="store_true")
    args = parser.parse_args()

    jet_net_test(
        args.datapath,
        args.outpath,
        args.ntest,
        args.with_flav,
        args.with_hel,
        private=args.private,
    )


def jet_net_test(
    datapath, outpath, ntest, with_flav, with_hel, *, private=False
):
    assert ntest is None or ntest > 0

    params, meta = fit_load(outpath)

    if private:
        test = "private_test"
    else:
        test = "test"

    with_truth = with_flav or with_hel

    test_parts = load_invariant_momenta(
        os.path.join(datapath, test, "*.h5"),
        nmax=ntest,
        with_truth=with_truth,
    )
    test_real = stitch_parts(test_parts, with_flav, with_hel)

    result = net_test(
        zeta_100_100_d_10,
        params,
        meta,
        test_real,
        tag={
            "datapath": datapath,
            "with_flav": with_flav,
            "with_hel": with_hel,
        },
    )
    result_dump(result, outpath, private=private)


if __name__ == "__main__":
    main()
