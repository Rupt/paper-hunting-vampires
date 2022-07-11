"""
Load a BDT model and test it against example data.

Usage example:

make env_bdt/bin/activate
source env_bdt/bin/activate

python example_bdt.py \
--datapath pv_msme_3j_4j_1_seed_80_truth_cut.h5 \
--modelpath results/models/jet_bdt_truth/liv_3j_4j_1/

"""
import argparse
import json
import os
import subprocess

from parity_tests.jet_bdt_lib import bdt_test, fit_load
from parity_tests.jet_lib import load_invariant_momenta, result_dump, stitch_parts

DEFAULT_DATAPATH = "pv_msme_3j_4j_1_seed_80_truth_cut.h5"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--modelpath", type=str, required=True)
    args = parser.parse_args()

    # ensure the default data file exists
    if args.datapath == DEFAULT_DATAPATH:
        subprocess.run(["make", DEFAULT_DATAPATH])

    example_bdt(args.datapath, args.modelpath)


def example_bdt(datapath, modelpath):
    model, meta = fit_load(modelpath)

    test_real = load_invariant_momenta(datapath)

    result = bdt_test(
        model,
        meta,
        test_real,
        tag={
            "datapath": datapath,
        },
    )
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
