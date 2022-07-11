"""
Load an NN model and test it against example data.

Usage example:

source example_nn.sh

OR

make env_nn/bin/activate
source env_nn/bin/activate

python example_nn.py \
--datapath pv_msme_3j_4j_1_seed_80_truth_cut.h5 \
--modelpath results/models/jet_net_truth/liv_3j_4j_1/

"""
import argparse
import json
import os
import subprocess

from parity_tests.jet_lib import load_invariant_momenta, result_dump, stitch_parts
from parity_tests.jet_net_lib import fit_load, net_test, zeta_100_100_d_10

DEFAULT_DATAPATH = "pv_msme_3j_4j_1_seed_80_truth_cut.h5"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--modelpath", type=str, required=True)
    args = parser.parse_args()

    # ensure the default data file exists
    if args.datapath == DEFAULT_DATAPATH:
        subprocess.run(["make", DEFAULT_DATAPATH])

    example_nn(args.datapath, args.modelpath)


def example_nn(datapath, modelpath):
    params, meta = fit_load(modelpath)

    test_real = load_invariant_momenta(datapath)

    result = net_test(
        zeta_100_100_d_10,
        params,
        meta,
        test_real,
        tag={
            "datapath": datapath,
        },
    )
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
