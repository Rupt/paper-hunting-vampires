"""Fit and serialize nets.

Usage:

e.g.

python parity_tests/jet_net_fit.py \
    --datapath /home/tombs/Downloads/truth_ktdurham200/liv_3j_4j_1/ \
    --outpath out_test_3 \
    --ntrain 100_000 \
    --nval 50_000 \
    --seed 123 \
    --with_flav --with_hel

"""
import argparse
import os

from jet_lib import load_invariant_momenta, stitch_parts
from jet_net_lib import fit_dump, net_fit, zeta_100_100_d_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--nval", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--with_flav", action="store_true")
    parser.add_argument("--with_hel", action="store_true")
    args = parser.parse_args()

    jet_net_fit(
        args.datapath,
        args.outpath,
        args.ntrain,
        args.nval,
        args.seed,
        args.with_flav,
        args.with_hel,
        nsteps_max=100_000,
        nsteps_round=1_000,
        batch_size=10_000,
        early_stopping_rounds=10,
        learning_rate=1e-3,
    )


def jet_net_fit(
    datapath,
    outpath,
    ntrain,
    nval,
    seed,
    with_flav,
    with_hel,
    *,
    batch_size,
    nsteps_max,
    nsteps_round,
    early_stopping_rounds,
    learning_rate,
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

    params, meta = net_fit(
        zeta_100_100_d_10,
        train_real,
        val_real,
        seed=seed,
        batch_size=batch_size,
        nsteps_max=nsteps_max,
        nsteps_round=nsteps_round,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        tag={
            "datapath": datapath,
            "with_flav": with_flav,
            "with_hel": with_hel,
        },
    )

    fit_dump(params, meta, outpath)


if __name__ == "__main__":
    main()
