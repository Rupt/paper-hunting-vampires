"""Modify MadGraph processes to evaluate matrix elements in the lab frame.

Warning: apply this before using the process.

Differing effects have been observed if using the process before modifying it,
presumably related to compilation.


Usage:

python use_lab_frame.py PROCESS_PATH

e.g.

python use_lab_frame.py process/liv_one_jjj

"""
import glob
import sys

BEGIN = "      ! LIV (use_lab_frame.py)\n"
END = "      ! ENDLIV\n"

DECLARATIONS = [
    "      DOUBLE PRECISION LIV_R,LIV_Y,LIV_BY\n",
    "      INTEGER LIV_I\n",
]

BOOSTER = [
    "      LIV_R = EBEAM(IB(1)) * XBK(IB(1)) / PP(0, 1)\n",
    "      LIV_Y = (LIV_R*LIV_R + 1) / (2*LIV_R)\n",
    "      LIV_BY = (LIV_R*LIV_R - 1) / (2*LIV_R)\n",
    "      IF (PP(3, 1) .LT. 0) THEN\n",
    "        LIV_BY = -LIV_BY\n",
    "      ENDIF\n",
    "      DO LIV_I=1,NEXTERNAL\n",
    "        P1(0, LIV_I) = LIV_Y*PP(0, LIV_I) + LIV_BY*PP(3, LIV_I)\n",
    "        P1(1, LIV_I) = PP(1, LIV_I)\n",
    "        P1(2, LIV_I) = PP(2, LIV_I)\n",
    "        P1(3, LIV_I) = LIV_BY*PP(0, LIV_I) + LIV_Y*PP(3, LIV_I)\n",
    "      ENDDO\n",
]

END_OF_LOCAL_VARIABLES = "      SAVE NFACT\n"
START_OF_BOOST = "      IF(FRAME_ID.NE.6)THEN\n"


def main():
    assert len(sys.argv) == 2, "usage: python use_lab_frame.py PROCESS_PATH"
    path = sys.argv[1]
    modify_process(path)


def modify_process(path):
    """Modify process at path to evaluate matrix elements in lab frame."""
    for target in glob.glob(path + "/SubProcesses/*/auto_dsig?.f"):
        modify_dsig(target)


def modify_dsig(target):
    """Rewrite fortran file `target' to use the lab frame."""
    with open(target, "r") as file_:
        lines = file_.readlines()
    out = []
    itlines = iter(lines)
    # scan until end of local variables
    while True:
        try:
            line = next(itlines)
        except StopIteration:
            raise ValueError("No line %r in file %r" % (END_OF_LOCAL_VARIABLES, target))
        out.append(line)
        if line == END_OF_LOCAL_VARIABLES:
            break
    # add liv local variables
    out.append(BEGIN)
    out += DECLARATIONS
    out.append(END)
    # scan until boost part; remove:
    # 0 IF(FRAME_ID.NE.6)THEN
    # 1   CALL BOOST_TO_FRAME(PP, FRAME_ID, P1)
    # 2 ELSE
    # 3   P1 = PP
    # 4 ENDIF
    while True:
        try:
            line = next(itlines)
        except StopIteration:
            raise ValueError("No line %r in file %r" % (START_OF_BOOST, target))
        if line == START_OF_BOOST:
            break
        out.append(line)
    try:
        for _ in range(4):
            next(itlines)
    except StopIteration:
        raise ValueError(
            "Not enough lines after %r in file %r" % (START_OF_BOOST, target)
        )
    # add liv boost part
    out.append(BEGIN)
    out += BOOSTER
    out.append(END)
    # take the rest
    out += itlines
    with open(target, "w") as file_:
        file_.writelines(out)
    print("wrote %r" % target)


if __name__ == "__main__":
    main()
