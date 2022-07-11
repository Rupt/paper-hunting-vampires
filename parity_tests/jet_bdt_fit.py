"""Fit and serialize BDT models.


Usage:

e.g.

python parity_tests/jet_bdt_fit.py \
    --datapath /home/tombs/Downloads/truth_ktdurham200/liv_3j_4j_1/ \
    --outpath out_test_2 \
    --ntrain 100_000 \
    --nval 50_000 \
    --seed 123 \
    --with_flav --with_hel


"""
import argparse
import os

from .jet_bdt_lib import bdt_fit, bdt_meta, fit_dump
from .jet_lib import load_invariant_momenta, stitch_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--nval", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    model_kwargs = dict(
        tree_method="hist",
        min_child_weight=10_000,
        n_estimators=1000,
        learning_rate=0.1,
        n_jobs=1,
    )

    jet_bdt_fit(
        args.datapath,
        args.outpath,
        args.ntrain,
        args.nval,
        args.seed,
        args.with_flav,
        args.with_hel,
        model_kwargs,
    )


def jet_bdt_fit(
    datapath, outpath, ntrain, nval, seed, with_flav, with_hel, model_kwargs
):
    with_truth = with_flav or with_hel

    train_parts = load_invariant_momenta(
        os.path.join(datapath, "train/*.h5"),
        nmax=ntrain,
        with_truth=with_truth,
    )
    train_real = stitch_parts(train_parts, with_flav, with_hel)

    # overloading "test" as validation
    val_parts = load_invariant_momenta(
        os.path.join(datapath, "test/*.h5"),
        nmax=nval,
        with_truth=with_truth,
    )
    val_real = stitch_parts(val_parts, with_flav, with_hel)

    model, iteration_to_quality = bdt_fit(
        seed,
        model_kwargs,
        train_real,
        val_real,
    )

    meta = bdt_meta(
        model,
        iteration_to_quality,
        len(train_real),
        len(val_real),
        tag={
            "datapath": datapath,
            "with_flav": with_flav,
            "with_hel": with_hel,
        },
    )

    fit_dump(model, meta, outpath)


if __name__ == "__main__":
    main()
