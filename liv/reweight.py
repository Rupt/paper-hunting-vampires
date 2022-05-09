"""Reweight standard model events to other other models.

Usage:

python reweight.py FILENAME


e.g.

python reweight.py /home/tombs/Cambridge/lester_flips/sm_atlas_200k.lhe


"""
import sys
from xml.etree import ElementTree

import lhe
import numpy
import standalone.liv_random_jjj as liv_random
import standalone.standard_jjj as standard
from element import Process

# import standalone.liv_1u10_1u30_jjj as liv_1u10_1u30


def main():
    """Take a filename from argv, write reweighted versions."""
    assert len(sys.argv) >= 2
    filename_in = sys.argv[1]
    assert filename_in[-4:] == ".lhe"
    filename_out = filename_in[:-4] + "_reweight.lhe"

    sm = Process(standard)
    liv = Process(liv_random)

    reweight(filename_in, sm, liv, filename_out, 1234)


def reweight(filename_in, from_, to, filename_out, seed):
    """Read events in, reweight by matrix element ratios, and write out.

    The maximum value used for unweighting is written in output metadata.

    Arguments:
        filename_in: file path to read
        seed: integer random seed for unweighting
        to: previous Process to divide out
        from_: Process to weight by
        filename_out: file path to write results to
    """
    with open(filename_in, "r") as file_:
        lhetree = ElementTree.parse(file_)

    newtree = without_events(lhetree)
    events = lhetree.findall(".//event")

    weights = numpy.array([event_weight(event, to, from_) for event in events])
    max_ = weights.max()

    rng = numpy.random.Generator(numpy.random.Philox(seed))
    keep = rng.uniform(0, max_, size=len(weights)) < weights

    root = newtree.getroot()
    for kept, event in zip(keep, events):
        if not kept:
            continue
        root.append(event)

    note_max(newtree, max_)

    # print info
    nkept = keep.sum()
    print(f"{max_ = }, {nkept = }")

    newtree.write(filename_out)
    print("Wrote %r" % filename_out)


def without_events(tree):
    """Return a new ElementTree which reproduces tree with no events."""
    root = tree.getroot()
    new_root = ElementTree.Element(root.tag, root.attrib, text=root.text)
    new = ElementTree.ElementTree(new_root)

    for element in root.findall("./"):
        if element.tag == "event":
            break
        new_root.append(element)

    return new


def note_max(tree, maximum):
    """Add this maximum info to MGGenerationInfo."""
    info = tree.find(".//MGGenerationInfo")
    info.text += "#  Reweight maximum        :       %r\n" % maximum


def get_max(tree):
    """Return the reweight maximum stored in tree."""
    info = tree.find(".//MGGenerationInfo")
    return float(info.text.split(" " * 7)[-1])


def event_weight(event_element, to_process, from_process):
    """Return a module ratio weight for the given event element."""
    event = lhe.parse_event(event_element)
    npart = len(event.particles)
    pdg_id = numpy.empty(npart)
    momentum = numpy.empty((npart, 4))
    helicity = numpy.empty(npart)

    for i, p in enumerate(event.particles):
        pdg_id[i] = p.pdg_id
        momentum[i] = [p.e, p.px, p.py, p.pz]
        helicity[i] = p.helicity

    # bot: helicity sum
    # top: sample a new helicity (useful to have ihel to array function)
    top = to_process(pdg_id, momentum, -1)
    bot = from_process(pdg_id, momentum, -1)

    # hacked to get some samples out
    r = top / bot

    # selecting to get a decent # of samples out
    if not (2 < r < 10):
        r = 0.0
    return r


if __name__ == "__main__":
    main()
