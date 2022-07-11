"""Test serialized BDT models against independent data.

Usage:

e.g.

python parity_tests/jet_bdt_test.py \
    --datapath /home/tombs/Downloads/truth_ktdurham200/liv_3j_4j_1/ \
    --outpath out_test_2 \
    --ntest 1_000_000 \
    --with_flav --with_hel

"""
import argparse
import os

from .jet_bdt_lib import bdt_test, fit_load
from .jet_lib import load_invariant_momenta, result_dump, stitch_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--with_flav", action="store_true")
    parser.add_argument("--with_hel", action="store_true")
    args = parser.parse_args()

    jet_bdt_test(
        args.datapath,
        args.outpath,
        args.ntest,
        args.with_flav,
        args.with_hel,
        private=args.private,
    )


def jet_bdt_test(
    datapath, outpath, ntest, with_flav, with_hel, *, private=False
):
    assert ntest is None or ntest > 0

    model, meta = fit_load(outpath)

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

    result = bdt_test(
        model,
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
