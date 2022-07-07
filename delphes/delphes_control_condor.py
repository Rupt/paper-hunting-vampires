"""Control Delphes and pythia detector reconstruction

This file should be run in the Delphes directory

-use env_atlas for this

"""
import argparse
import subprocess
import os
import tempfile


DIR_DELPHES = (
    "/usera/dnoel/Documents/parity/Delphes-3.5.0/"  # use . if current dir
)
# calling
def main():
    parser = argparse.ArgumentParser(
        description=("Delphes and Pythia8 detector reconstruction.")
    )
    parser.add_argument(
        "-d",
        "--delphes_card",
        type=str,
        help="Delphes card to use",
        default="delphes_card_ATLAS_R04.tcl",
    )
    parser.add_argument(
        "-f", "--file_name", type=str, help="lhe file to run over", default="sm_200k"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        help="lhe file to run over",
        default="/r10/atlas/symmetries/data/madgraph",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="output directory",
        default="/r10/atlas/symmetries/data/delphes",
    )
    parser.add_argument(
        "--nevents", type=str, help="number of events to run over", default=200000
    )  # can this be done automatically?

    parser.add_argument("--pileup", dest="pileup", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=9001)

    args = parser.parse_args()
    print("running over ", args.in_dir, args.file_name)

    if args.pileup:
        delphes_do_pileup(
            args.delphes_card,
            args.file_name,
            args.in_dir,
            args.out_dir,
            args.nevents,
            args.seed,
        )

    else:
        delphes_do(
            args.delphes_card,
            args.file_name,
            args.in_dir,
            args.out_dir,
            args.nevents,
            args.seed,
        )


def get_pythia_card(filepath, nevents, seed):
    """Makes pythia card for running Delphes

    Note - seed * 1000 used such that we don't have same seed for pile-up
    and for the pythia in the delphes simulation"""

    card_string = """
    ! 1) Settings used in the main program.

    Main:numberOfEvents = {nevents}         ! number of events to generate
    Main:timesAllowErrors = 3          ! how many aborts before run stops

    ! 2) Settings related to output in init(), next() and stat().

    Init:showChangedSettings = on      ! list changed settings
    Init:showChangedParticleData = off ! list changed particle data
    Next:numberCount = 1000             ! print message every n events
    Next:numberShowInfo = 1            ! print event information n times
    Next:numberShowProcess = 1         ! print process record n times
    Next:numberShowEvent = 0           ! print event record n times

    ! 3) Set the input LHE file

    Beams:frameType = 4
    Beams:LHEF = {filepath}

    Random:setSeed = on
    Random:setSeed = {seed}


    ! 4) Insert information about jet matching manually
    JetMatching:merge = 0 # switch off MLM matching
    JetMatching:setMad = off
    JetMatching:qCut = 0.0
    JetMatching:nQmatch = 4
    JetMatching:clFact = 1.0

    Merging:doKTMerging on # kT scale for merging shower products into jets
    Merging:nJetMax 1 # Maximal number of additional jets in the matrix element
    Merging:Process pp>jjj # The string specifying the hard core process
    Merging:TMS 200 # The value of the merging scale TMS = ktdurham
    Merging:Dparameter 0.4 # Definition of longitudinally invariant kT separation

    """.format(
        nevents=nevents, filepath=filepath, seed=seed * 1000
    )

    return card_string


def add_random_seed_tcl(delphes_card, out_dir="", seed=9001):
    """add a random seed to delphes card so we can exaclty reproduce our results
    and change the minBias.pileup filepath - allows for multiple delphes processes
    to be running over different minBias files simultaneously


    """
    with open(os.path.join(out_dir, delphes_card), "r") as f:
        # read a list of lines into file_lines
        file_lines = f.readlines()

    # add trailing / here to allow runnng with default out_dir too
    if out_dir != "":
        out_dir = out_dir + "/"

    changed_pileup_filepath = False
    changed_seed = False
    for i, line in enumerate(file_lines):
        # set random seed
        if line == "# set RandomSeed <insert_number_here>\n":
            file_lines[i] = "set RandomSeed {}\n".format(seed)
            changed_seed = True

        # set MinBias.pileup filepath - this is made on the fly when producing the delphes pileup file
        if line == "  set PileUpFile MinBias.pileup\n":
            file_lines[i] = "  set PileUpFile {}MinBias.pileup\n".format(out_dir)
            changed_pileup_filepath = True

    if not changed_seed:
        print(" WARNING: Couldn't change random seed - it's not exactly reproducible!")
        raise  # can drop this in the future

    if not changed_pileup_filepath:
        print(
            " WARNING: Couldn't change pileup file path - will call standard one in delphes directory if this is a pileup file!"
        )

    # write the lines with edited seed to a temp file
    tcl_file_ = tempfile.NamedTemporaryFile("w+")
    tcl_file_.writelines(file_lines)
    tcl_file_.flush()

    return tcl_file_


def delphes_do(delphes_card, file_name, in_dir, out_dir, nevents, seed=9001):
    """Run delphes..."""
    # write the pyhia card

    if not file_name.endswith(".lhe"):
        file_name += ".lhe"

    filepath_lhe = os.path.join(out_dir, file_name)  # unzip to out_dir
    pythia_card_string = get_pythia_card(filepath_lhe, nevents, seed)
    print(pythia_card_string)

    # write this to a temp file...
    pythia_file_ = tempfile.NamedTemporaryFile("w+")
    pythia_file_.write(pythia_card_string)
    pythia_file_.flush()

    # add random seed and MinBias.pileup directory
    tcl_file_ = add_random_seed_tcl(delphes_card, out_dir, seed)

    # run delphes
    commands = [
        DIR_DELPHES + "DelphesPythia8",
        tcl_file_.name,
        pythia_file_.name,
        "{}/{}_{}_{}events.root".format(
            out_dir,
            file_name.replace(".lhe", ""),
            delphes_card.replace(".tcl", ""),
            nevents,
        ),
    ]

    subprocess.run(commands)

    tcl_file_.close()
    pythia_file_.close()


# -------------------------------------------------------------------------------
# pileup


def get_pythia_pileup_card(nevents, seed=9001):
    """Makes pythia card for generating pileup events"""
    card_string = """
    ! File: generatePileUp.cmnd
    ! This file contains commands to be read in for a Pythia8 run.
    ! Lines not beginning with a letter or digit are comments.
    ! Names are case-insensitive  -  but spellings-sensitive!
    ! The changes here are illustrative, not always physics-motivated.

    ! 1) Settings that will be used in a main program.
    Main:numberOfEvents = {nevents}          ! number of events to generate
    Main:timesAllowErrors = 3          ! abort run after this many flawed events

    ! 2) Settings related to output in init(), next() and stat().
    Init:showChangedSettings = on      ! list changed settings
    Init:showAllSettings = off         ! list all settings
    Init:showChangedParticleData = on  ! list changed particle data
    Init:showAllParticleData = off     ! list all particle data
    Next:numberCount = 1000            ! print message every n events
    Next:numberShowLHA = 1             ! print LHA information n times
    Next:numberShowInfo = 1            ! print event information n times
    Next:numberShowProcess = 1         ! print process record n times
    Next:numberShowEvent = 1           ! print event record n times
    Stat:showPartonLevel = on          ! additional statistics on MPI
    Random:setSeed = on
    Random:setSeed = {seed}

    ! 3) Beam parameter settings. Values below agree with default ones.
    Beams:idA = 2212                   ! first beam, p = 2212, pbar = -2212
    Beams:idB = 2212                   ! second beam, p = 2212, pbar = -2212
    Beams:eCM = 13000.                 ! CM energy of collision

    ! 4a) Pick processes and kinematics cuts.
    SoftQCD:all = on                   ! Allow total sigma = elastic/SD/DD/ND

    ! 4b) Other settings. Can be expanded as desired.
    Tune:pp = 5                         ! use Tune 5

    """.format(
        nevents=nevents, seed=seed
    )

    return card_string


def generate_pile_up(nevents, out_dir, seed=9001):
    """generate pile up sample and save to MinBias.root
    since this is in the .tcl file
    """

    # write the pythia card for writing pileup
    pythia_card_string = get_pythia_pileup_card(nevents, seed)
    print(pythia_card_string)

    # write this to a temp file...
    pythia_file_ = tempfile.NamedTemporaryFile("w+")
    pythia_file_.write(pythia_card_string)
    pythia_file_.flush()

    # ---------------------
    # generate the pileup event sample
    gen_commands = [
        DIR_DELPHES + "DelphesPythia8",
        os.path.join(out_dir, "converter_card.tcl"),
        pythia_file_.name,
        os.path.join(out_dir, "MinBias.root"),
    ]

    subprocess.run(gen_commands)
    pythia_file_.close()

    # convert .root -> .pileup ready for
    conv_commands = [
        DIR_DELPHES + "root2pileup",
        os.path.join(out_dir, "MinBias.pileup"),
        os.path.join(out_dir, "MinBias.root"),
    ]
    subprocess.run(conv_commands)


def delphes_do_pileup(delphes_card, file_name, in_dir, out_dir, nevents, seed=9001):
    """Run delphes..."""

    print("***************generating pileup******************")
    generate_pile_up(nevents, out_dir, seed)

    print("***************Running delphes ", file_name, in_dir, " ***************")
    delphes_do(delphes_card, file_name, in_dir, out_dir, nevents, seed)


if __name__ == "__main__":
    main()
