"""Control madgraph jobs of preparing matrix elements and simulating events.

Usage examples in README.md.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import use_lab_frame

DEFINITIONS = [
    "define i = g u d u~ d~",
]

PP_JJJ = "p p > j j j"


# calling
def main():
    options = ("output", "output_standalone", "launch")
    option = sys.argv[1]

    if option not in options:
        options_str = " | ".join(options)
        print(f"Usage: python {__file__} {options_str} *args")
        return

    if option == "output":
        parse_output()
    elif option == "output_standalone":
        parse_output_standalone()
    else:
        assert option == "launch"
        parse_launch()

    return
    title = "standard_jjj"

    output(title, PP_JJJ)
    output_standalone(title + "_standalone", PP_JJJ)
    launch(title, 1)


def parse_output():
    parser = argparse.ArgumentParser(
        description=("Create a MadGraph process output directory.")
    )
    parser.add_argument("output")
    parser.add_argument("title", type=str, help="directory name")
    parser.add_argument(
        "--process",
        type=str,
        nargs="+",
        help="particle scattering process(es)",
        default=PP_JJJ,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="optional model to import",
        default=None,
    )
    parser.add_argument(
        "--lab",
        action="store_true",
        help="modify to evaluate matrix elements in the lab frame",
        default=None,
    )
    args = parser.parse_args()

    output(args.title, args.process, model=args.model, lab=args.lab)


def parse_output_standalone():
    parser = argparse.ArgumentParser(
        description=("Create and build a MadGraph matrix element module.")
    )
    parser.add_argument("output_standalone")
    parser.add_argument("title", type=str, help="directory name")
    parser.add_argument(
        "--process",
        type=str,
        help="particle scattering process",
        default=PP_JJJ,
    )
    parser.add_argument(
        "--model", type=str, help="optional model to import", default=None
    )
    args = parser.parse_args()

    output_standalone(args.title, args.process, model=args.model)


def parse_launch():
    parser = argparse.ArgumentParser(
        description=("Generate samples with madgraph.")
    )
    parser.add_argument("output_standalone")
    parser.add_argument("title", type=str, help="directory name")
    parser.add_argument("--seed", type=int, help="random seed", required=True)
    parser.add_argument("--name", type=str, help="output name", default=None)
    parser.add_argument(
        "--nevents",
        type=int,
        help="target maximum number of events",
        default=10000,
    )
    parser.add_argument(
        "--ncores",
        type=int,
        help="use multicore mode with this many cores",
        default=None,
    )
    parser.add_argument(
        "--kwargs",
        type=str,
        help="json dict of other parameters to set in launching",
        default=None,
    )
    args = parser.parse_args()

    if args.kwargs is None:
        kwargs = {}
    else:
        kwargs = json.loads(args.kwargs)

    launch(
        args.title,
        args.seed,
        name=args.name,
        nevents=args.nevents,
        ncores=args.ncores,
        set_kwargs=kwargs,
    )


# madgraph
def output(title, process, *, model=None, lab=False):
    """Output a MadGraph process."""
    if isinstance(process, str):
        process = [process]

    if model is None:
        imports = []
    else:
        imports = [f"import model {model}"]

    add_processes = []
    for proc in process[1:]:
        add_processes.append(f"add process {proc}")

    madgraph_do(
        [
            *DEFINITIONS,
            *imports,
            f"generate {process[0]}",
            *add_processes,
            f"output {title} -nojpeg --noeps=True",
        ]
    )

    os.makedirs(os.path.join(title, "Events"), exist_ok=True)

    if lab:
        print("converting to use lab frame...")
        use_lab_frame.modify_process(title)


def output_standalone(title, process, *, model=None):
    """Output and build a MadGraph matrix element python module."""
    if model is None:
        imports = []
    else:
        imports = [f"import model {model}"]

    madgraph_do(
        [
            *DEFINITIONS,
            *imports,
            f"generate {process}",
            f"output standalone {title} --prefix=int -nojpeg --noeps=True",
        ]
    )

    os.makedirs(os.path.join(title, "Events"), exist_ok=True)

    # build
    subprocess.run(
        ["make", "allmatrix2py.so"],
        shell=True,
        cwd=os.path.join(title, "SubProcesses"),
    )

    # make the title importable
    subprocess.run(
        [
            "ln",
            "-f",
            os.path.join(
                title,
                "SubProcesses",
                "all_matrix2py.cpython-38-x86_64-linux-gnu.so",
            ),
            title,
        ]
    )

    with open(os.path.join(title, "__init__.py"), "w") as file_:
        file_.write("from .all_matrix2py import *")


def launch(
    title, seed, *, name=None, nevents=10000, ncores=None, set_kwargs=None
):
    """Generate events for the given process."""
    assert seed != 0, "found seed == 0, which would be replaced by MadGraph"
    sets = {
        "iseed": seed,
        "nevents": nevents,
        "use_syst": False,
    }
    if set_kwargs is not None:
        sets.update(set_kwargs)

    if ncores is None:
        multicore = ""
        mncores = []
    else:
        multicore = "--multicore"
        mncores = [f"{ncores}"]

    if name is None:
        name = ""
    else:
        name = f"--name {name}"

    madgraph_do(
        [
            f"launch {title} {multicore} {name}",
            *mncores,
            "0",
            *(f"set {key} {value}" for key, value in sets.items()),
            "0",
        ]
    )


def madgraph_do(commands):
    """Pass the given commands to madgraph as if in interactive mode."""
    file_ = tempfile.NamedTemporaryFile("w+")
    for command in commands:
        file_.write(command + "\n")
    file_.flush()

    subprocess.run(["./mg5_aMC", "-f", file_.name])

    file_.close()


if __name__ == "__main__":
    main()
