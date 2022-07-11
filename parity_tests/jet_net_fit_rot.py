"""Fit and serialize nets on a rotating planet.

Usage:

e.g.

python parity_tests/jet_net_fit_rot.py \
    --datapath /home/tombs/Downloads/liv_rot_smol/ \
    --outpath out_test_rot_net \
    --ntrain 100_000 \
    --nval 50_000 \
    --seed 123

"""
import argparse
import os

import numpy

from .jet_lib import load_rot, load_rot_xs
from .jet_net_lib import fit_dump, net_fit, zeta_20_20_10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--nval", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    jet_net_fit(
        args.datapath,
        args.outpath,
        args.ntrain,
        args.nval,
        args.seed,
        nsteps_max=100_000,
        nsteps_round=1_000,
        batch_size=args.batch_size,
        early_stopping_rounds=10,
        learning_rate=1e-3,
        normalize=args.normalize,
    )


def jet_net_fit(
    datapath,
    outpath,
    ntrain,
    nval,
    seed,
    *,
    batch_size,
    nsteps_max,
    nsteps_round,
    early_stopping_rounds,
    learning_rate,
    normalize=False,
):
    xs = load_rot_xs()
    if normalize:
        xs[:] = 1
    probs = xs / xs.sum()

    rng = numpy.random.Generator(numpy.random.Philox(31415))

    path = os.path.join(datapath, "train")
    nper = rng.multinomial(ntrain, probs)
    train_parts = load_rot(path, nper=nper)
    rng.shuffle(train_parts)

    # overloading "test" as validation
    path = os.path.join(datapath, "test")
    nper = rng.multinomial(nval, probs)
    val_parts = load_rot(path, nper=nper)
    rng.shuffle(val_parts)

    params, meta = net_fit(
        zeta_20_20_10,
        train_parts,
        val_parts,
        seed=seed,
        batch_size=batch_size,
        nsteps_max=nsteps_max,
        nsteps_round=nsteps_round,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        tag={
            "datapath": datapath,
        },
    )

    fit_dump(params, meta, outpath)


if __name__ == "__main__":
    main()
