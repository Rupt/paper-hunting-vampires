"""
Produce image representations of parton truth momenta.

Usage:

e.g.

python gen_data/process_truth_to_images.py \
--infile /home/tombs/Downloads/truth_ktdurham200/liv_3j_4j_1/train/liv_3j_4j_1_8_truth.h5 \
--outfile /home/tombs/Downloads/truth_ktdurham200_images/liv_3j_4j_1/train/liv_3j_4j_1_8_truth_images.h5

python gen_data/process_truth_to_images.py --random_rotate \
--infile /home/tombs/Downloads/truth_ktdurham200/liv_3j_4j_1/train/liv_3j_4j_1_8_truth.h5 \
--outfile /home/tombs/Downloads/truth_ktdurham200_images_rot/liv_3j_4j_1/train/liv_3j_4j_1_8_truth_images.h5

"""
import argparse
import os
import zlib

import h5py
import numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--random_rotate", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = zlib.adler32(args.infile.encode())

    truth_jets = load_truth(args.infile)

    if args.random_rotate:
        truth_jets = random_rotate(truth_jets, seed=args.seed)

    truth_images = truth_to_images(truth_jets)

    dump_truth_images(truth_images, args.outfile)


def load_truth(filename):
    with h5py.File(filename, "r") as file_:
        events = file_["events"][:]
    return events


def random_rotate(jets, *, seed):
    """Return jets randomly rotated about z and flipped along z."""
    rng = numpy.random.Generator(numpy.random.Philox(seed))

    # px py pz e order
    assert jets.shape[1] % 4 == 0
    nparticles = jets.shape[1] // 4

    out = jets.copy()
    for i in range(len(jets)):
        momenta = jets[i].reshape(-1, 4)[:, :3]

        # about z
        phi = rng.uniform(high=2 * numpy.pi)
        rot = rotation_matrix(phi)[1:, 1:].astype(momenta.dtype)
        momenta = momenta @ rot

        # along z (z <- -z, x <- -x) with probability 0.5
        xzsign = rng.integers(2) * 2 - 1
        momenta *= numpy.array((xzsign, 1, xzsign), dtype=momenta.dtype)

        for j in range(nparticles):
            out[i, 4 * j : 4 * j + 3] = momenta[j]

    return out


def truth_to_images(jets):
    # jets has shape (:, 4 * nparticles)
    # where the 4 are (px, py, pz, e)
    px = jets[:, 0::4]
    py = jets[:, 1::4]
    pz = jets[:, 2::4]
    e = jets[:, 3::4]

    pt = (px ** 2 + py ** 2) ** 0.5
    p = (pt ** 2 + pz ** 2) ** 0.5
    # missing particles have p=0 (and pz=0); these will be cleaned up later
    eta = numpy.arctanh(pz / numpy.where(p == 0, 1, p))
    phi = numpy.arctan2(py, px)
    mass = numpy.maximum(0, e ** 2 - p ** 2) ** 0.5

    images = numpy.zeros((len(pt), 32, 32), dtype=numpy.float32)
    for i, (pt_i, eta_i, phi_i, mass_i) in enumerate(zip(pt, eta, phi, mass)):
        images[i] = jet_image(pt_i, eta_i, phi_i, mass_i)[0]

    return images


def dump_truth_images(images, filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with h5py.File(filename, "w") as file_:
        file_.create_dataset("entries", data=images, compression="gzip")
    print("wrote %r" % filename)


# ripped from process_delphes_all.py
# (duplicated to avoid slow imports such as pandas)
def jet_image(pt, eta, phi, mass):
    """Return a calorimeter image for given jet properties."""
    bin_range = [(-3.2, 3.2), (-numpy.pi, numpy.pi)]  # eta x phi
    bins = [32, 32]  # eta x phi

    image, xedges, yedges = numpy.histogram2d(
        eta,
        phi,
        bins=bins,
        range=bin_range,
        density=False,
        weights=pt,
    )

    return image, xedges, yedges


# ripped from test_invariant_coordinates.py
def rotation_matrix(phi):
    """Return a matrix M such that p @ M = rot_by_theta(p)."""
    c = numpy.cos(phi)
    s = numpy.sin(phi)
    return numpy.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ]
    ).T


if __name__ == "__main__":
    main()
