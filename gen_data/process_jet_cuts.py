"""
Read truth jets h5 and output truth jets h5 with pt > 200, eta < 2.8 cuts


# Example usage:


source setup.sh


INPATH=/r04/atlas/symmetries/data/truth_ktdurham200
OUTPATH=/r04/atlas/symmetries/data/truth_ktdurham200_cut
DATASETS='train test private_test'


# one screen per loop
for SIMSET in liv_3j_4j_p7 liv_3j_4j_p8 liv_3j_4j_p9
for SIMSET in liv_3j_4j_p3 liv_3j_4j_p4 liv_3j_4j_p5 liv_3j_4j_p6
for SIMSET in liv_3j_4j_1000 liv_3j_4j_p01 liv_3j_4j_p1 liv_3j_4j_p2
for SIMSET in sm_3j_4j liv_3j_4j_1 liv_3j_4j_10 liv_3j_4j_100
do
    for DATASET in ${DATASETS}
    do
        mkdir -p ${OUTPATH}/${SIMSET}/${DATASET}
        for ITEM in $(ls ${INPATH}/${SIMSET}/${DATASET})
        do
            python gen_data/process_jet_cuts.py \
                ${INPATH}/${SIMSET}/${DATASET}/${ITEM} \
                ${OUTPATH}/${SIMSET}/${DATASET}/${ITEM/.h5/_cut.h5}
        done
    done
done


"""
import argparse

import h5lib
import numpy


def main():
    """Process arguments and invoke actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    assert args.input_path.endswith(".h5")
    assert args.output_path.endswith(".h5")

    jet_cuts(args.input_path, args.output_path)
    print("wrote %r" % args.output_path)


def jet_cuts(input_path, output_path, **kwargs):
    inh5 = h5lib.load_h5(input_path)

    events = inh5["events"]
    flavors = inh5["flavors"]
    helicities = inh5["helicities"]
    assert len(events) == len(flavors) == len(helicities)

    nparticles = events.shape[1] // 4
    assert events.shape[1] == nparticles * 4
    dtype = events.dtype

    data_ps = []
    data_fs = []
    data_hs = []

    for event_i, flav_i, hel_i in zip(events, flavors, helicities):
        ps = []
        fs = []
        hs = []

        for i in range(nparticles):
            px, py, pz, e = event_i[4 * i : 4 * i + 4]

            pt = (px ** 2 + py ** 2) ** 0.5
            if not pt > 220:
                continue

            p = (px ** 2 + py ** 2 + pz ** 2) ** 0.5
            eta = numpy.arctanh(pz / p)
            if not abs(eta) < 2.8:
                continue

            ps += [px, py, pz, e]
            fs.append(flav_i[i])
            hs.append(hel_i[i])

        if len(fs) < 3:
            continue

        # pad with zeros up to n
        nspare = nparticles - len(fs)
        ps += [0.0, 0.0, 0.0, 0.0] * nspare
        fs += [0] * nspare
        hs += [0] * nspare

        data_ps.append(ps)
        data_fs.append(fs)
        data_hs.append(hs)

    data_ps = numpy.array(data_ps, dtype=dtype)
    data_fs = numpy.array(data_fs, dtype=numpy.int8)
    data_hs = numpy.array(data_hs, dtype=numpy.int8)

    content = {
        "events": data_ps,
        "flavors": data_fs,
        "helicities": data_hs,
    }
    h5lib.save_h5(content, output_path, **kwargs)


if __name__ == "__main__":
    main()
