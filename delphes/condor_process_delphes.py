import argparse
import os

from process_delphes_all import run_processing


def main():

    parser = argparse.ArgumentParser(
        description="process delphes root files on condor"
    )

    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    in_dir = "."  # . on condor (file transferred over)
    out_dir = "."  # . on condor (set output dir in condor job file)

    run_processing(args.filename, in_dir, out_dir)


if __name__ == "__main__":
    main()
