import os
import subprocess
import tempfile
import time
import argparse

DIR_DELPHES = '/usera/dnoel/Documents/parity/Delphes-3.5.0/'
CWD = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser(description="bin eta phi")
    parser.add_argument(
        "--condor", help="run on condor?", action="store_true", default=False
    )
    parser.add_argument(
        "-d",
        "--delphes_card",
        type=str,
        help="Delphes card to use",
        default="delphes_card_ATLAS_PileUp_R04.tcl",
    )
    args = parser.parse_args()


    madgraph_dir = "data"
    delphes_dir = "output"

    in_dir = madgraph_dir
    out_dir = delphes_dir
    os.makedirs(out_dir, exist_ok=True)


    filename = "sm.lhe.gz"
    seed = '0'
    model = 'sm'
    num = '0'
    set_name = 'train'
    run_delphes(
                    filename,
                    in_dir,
                    out_dir,
                    seed,
                    model,
                    set_name,
                    num,
                    args.delphes_card,
                    run_condor=args.condor,
                    do_processing = True
                )



# save pile up and no-pile up
def get_delphes_sh_string(
    filename, in_dir, out_dir, seed, tcl_file="delphes_card_ATLAS_PileUp_R04.tcl", nevents = 200000
):

    if "PileUp" in tcl_file:
        pileup_str = "--pileup"
    else:
        pileup_str = ""

    sh_string = """#!/bin/bash
    export CURRENTDIR=`pwd`
    echo Running the bash file in $CURRENTDIR
    gunzip -c {filename}.gz > {filename}

    cp {CWD}/{tcl_file} .
    cp {CWD}/converter_card.tcl .

    # run in Delphes install directory
    cd {DIR_DELPHES}
    cp {CWD}/delphes_control_condor.py .
    python delphes_control_condor.py -f {filename} --in_dir {in_dir} --out_dir $CURRENTDIR  -d {tcl_file} {pileup_str} --seed {seed} --nevents {nevents}
    rm delphes_control_condor.py

    # return to initial dir
    cd $CURRENTDIR
    """.format(
        filename=filename,
        in_dir=in_dir,
        out_dir=out_dir,
        tcl_file=tcl_file,
        seed=seed,
        pileup_str=pileup_str,
        nevents=nevents,
        DIR_DELPHES=DIR_DELPHES,
        CWD=CWD
    )

    return sh_string


def run_delphes(
    filename, in_dir, out_dir, seed, model, set_name, num, tcl_file, run_condor=False, do_processing=False
):
    """unzip lhe.gz, run delphes with get_delphes_sh_string
    and then clean up

    Can reduce the number of arguments?"""

    nevents = 20

    lhe_filename = filename.replace(".gz", "")
    lhe_filepath = os.path.join(out_dir, lhe_filename)

    sh_string = get_delphes_sh_string(lhe_filename, in_dir, out_dir, seed, tcl_file, nevents)

    if do_processing :
        sh_string = add_processing_string(sh_string, filename, tcl_file, nevents)

    condor_run_filename = os.path.join(out_dir, "condor_run.sh")
    sh_file_ = open(condor_run_filename, "w")
    sh_file_.write(sh_string)
    sh_file_.flush()
    sh_file_.close()
    time.sleep(0.5)

    if run_condor:
        print("submitting: condor ")
        os.system(
            "condor_submit condor_multiple.job PROCESS_NAME={process_name} SET_NAME={set_name} NUM={num} ".format(
                process_name=model, set_name=set_name, num=num
            )
        )

    else:
        # for command line copy the condor logic - copy file to out_dir and change to it
        os.system(
            "cp {gz_file} {out_dir}".format(
                gz_file=os.path.join(in_dir, filename), out_dir=out_dir
            )
        )
        cwd = os.getcwd()
        os.chdir(out_dir)
        os.system("chmod a+x {}".format("condor_run.sh"))
        os.system("./condor_run.sh")
        os.chdir(cwd)



def print_seeds(models, sets, madgraph_dir, delphes_dir):
    """print all the seeds used to file
    - follows the same logic as the main loop"""

    f = open("seeds_used.txt", "w")

    for i, model in enumerate(models, 1):
        for j, set_name in enumerate(sets, 1):

            in_dir = os.path.join(madgraph_dir, model, set_name)

            for filename in os.listdir(in_dir):

                # get the number for the file e.g. sm_3j_4j_200k_10.lhe.gz -> 10
                num = int(filename.split("_")[-1].replace(".lhe.gz", ""))

                seed = "{}{}{}".format(i, j, num)
                f.write("{}/{} seed = {}\n".format(in_dir, filename, seed))
    f.close()

#---------------------------------

def add_processing_string(sh_string, filename, delphes_card, nevents):
    """add to the delphes running job also the code for processing the delphes file"""

    filename_delphes = "{}_{}_{}events.root".format(
                filename.replace(".lhe.gz", ""),
                delphes_card.replace(".tcl", ""),
                nevents)

    process_string = """
        # pwd
        # export CURRENTDIR=`pwd`

        #copy to the output directory
        cp {CWD}/condor_process_delphes.py .
        cp {CWD}/process_delphes_all.py .

        python condor_process_delphes.py --filename {filename_delphes}
        echo {filename_delphes}

        #remove the .root file
        rm condor_process_delphes.py
        rm process_delphes_all.py
    """.format(filename_delphes=filename_delphes,
        CWD=CWD)

    return sh_string + process_string
if __name__ == "__main__":
    main()
