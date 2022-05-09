"""
Dump data to scatter for kinemtic distribution plots.

!!! Requires external data !!!

Usage:

python plot/prepare_kinematics.py ${DATAPATH} %{MODELPATH}

e.g.

python plot/prepare_kinematics.py \
/home/tombs/Downloads/reco_ktdurham200 \
/home/tombs/Downloads/models/jet_net_reco/

"""


import os
import sys

import h5py
import jax
import jet_lib
import jet_net_lib
import numpy
from jet_net_lib import make_net, parity_flip_jax, prescale, zeta_100_100_d_10

MODEL = "liv_3j_4j_1"

FILENAME = "results/kinematics.h5"


def main():
    assert (
        len(sys.argv) == 3
    ), "Usage: python plot/plot_kinematics ${DATAPATH} ${MODELPATH}"

    data_path, model_path = sys.argv[1:]
    data = jet_lib.load_invariant_momenta(
        os.path.join(data_path, MODEL, "private_test/*.h5"),
        nmax=1_000_000,
    )

    phi = evaluate_net(model_path, data)

    with h5py.File(FILENAME, "w") as file_:
        file_["data"] = data
        file_["phi"] = phi
    print("wrote %r" % FILENAME)


def evaluate_net(model_path, data):
    params, meta = jet_net_lib.fit_load(os.path.join(model_path, MODEL))

    net = make_net(zeta_100_100_d_10)

    pre_loc, pre_scale = meta["prescale"].values()
    pre_loc = jax.numpy.array(pre_loc, dtype=numpy.float32)
    pre_scale = jax.numpy.array(pre_scale, dtype=numpy.float32)

    @jax.jit
    def net_phi(params, x):
        x = prescale(x, pre_loc, pre_scale)
        zeta = net.net
        phi_1 = zeta.apply(params, x) - zeta.apply(params, parity_flip_jax(x))
        return phi_1.ravel()

    return numpy.array(net_phi(params, data))


if __name__ == "__main__":
    main()
