"""
Process lhe files to h5 with truth level information.

# Example usage:


source setup.sh


MADPATH=/r10/atlas/symmetries/data/madgraph
TRUTHPATH=/r10/atlas/symmetries/data/truth
DATASETS='train test private_test'


# one screen per loop
for SIMSET in liv_3j_4j_p7 liv_3j_4j_p8 liv_3j_4j_p9
for SIMSET in liv_3j_4j_p3 liv_3j_4j_p4 liv_3j_4j_p5 liv_3j_4j_p6
for SIMSET in liv_3j_4j_1000 liv_3j_4j_p01 liv_3j_4j_p1 liv_3j_4j_p2
for SIMSET in sm_3j_4j liv_3j_4j_1 liv_3j_4j_10 liv_3j_4j_100
do
    for DATASET in ${DATASETS}
    do
        mkdir -p ${TRUTHPATH}/${SIMSET}/${DATASET}
        for ITEM in $(ls ${MADPATH}/${SIMSET}/${DATASET})
        do
            python gen_data/process_lhe_to_h5.py \
                ${MADPATH}/${SIMSET}/${DATASET}/${ITEM} \
                ${TRUTHPATH}/${SIMSET}/${DATASET}/${ITEM/.lhe.gz/_truth.h5}
        done
    done
done


"""
import argparse
import gzip

import h5lib
import lhe
import numpy


def main():
    """Process arguments and invoke actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    assert args.input_path.endswith(".lhe.gz")
    assert args.output_path.endswith(".h5")

    lhe_to_h5(args.input_path, args.output_path)
    print("wrote %r" % args.output_path)


def lhe_to_h5(input_path, output_path, **kwargs):
    """Write an h5 version of the lhe.gz in input_path to output_path."""
    nparticles = 5
    dtype = numpy.float32

    with gzip.open(input_path, "r") as file_:
        data_ps, data_fs, data_hs = read_events(file_, nparticles, dtype)

    content = {
        "events": data_ps,
        "flavors": data_fs,
        "helicities": data_hs,
    }
    h5lib.save_h5(content, output_path, **kwargs)


def read_events(file_name_or_object, nparticles, dtype):
    """Return arrays for particles in file_name_or_object.

    Returns:
        "events" (ndata, px py pz e...)
        "flavors" (ndata, flavor...)
        "helicities" (ndata, helicity...)

    """
    data_ps = []
    data_fs = []
    data_hs = []

    for event in lhe.read_lhe(file_name_or_object):
        # read decreasing-pt-sorted momenta
        # the first two elements are partons
        out_i = []
        for i, p in enumerate(event.particles[2:], 2):
            out_i.append((p.px ** 2 + p.py ** 2, i))

        ps = []
        fs = []
        hs = []
        for _, i in sorted(out_i, reverse=True):
            p = event.particles[i]
            # Note x y z e ordering!
            ps += [p.px, p.py, p.pz, p.e]
            fs.append(p.pdg_id)
            hs.append(p.helicity)

        # pad with zeros up to n
        nspare = nparticles - len(out_i)
        ps += [0.0, 0.0, 0.0, 0.0] * nspare
        fs += [0] * nspare
        hs += [0] * nspare

        data_ps.append(ps)
        data_fs.append(fs)
        data_hs.append(hs)

    data_ps = numpy.array(data_ps, dtype=dtype)
    data_fs = numpy.array(data_fs, dtype=numpy.int8)
    data_hs = numpy.array(data_hs, dtype=numpy.int8)

    return data_ps, data_fs, data_hs


if __name__ == "__main__":
    main()
