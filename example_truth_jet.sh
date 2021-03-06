# Usage:
# source example_truth_jet.sh ${LHE_GZ}
# e.g.
# source example_truth_jet.sh pv_msme_0p5.lhe.gz
LHE_GZ=$1

if [ ! ${LHE_GZ} ]
then
    echo "Usage: source example_truth_jet.sh \${LHE_GZ}"
    return
fi

OUTNAME=$(basename ${LHE_GZ})

make env_basic/bin/activate
source env_basic/bin/activate


# Read the xml-based lhe file into an array format
python gen_data/process_lhe_to_h5.py \
${LHE_GZ} \
${OUTNAME/.lhe.gz/_truth.h5}


# Apply selection cuts (3jet > 220 GeV |eta| < 2.8 )
python gen_data/process_jet_cuts.py \
${OUTNAME/.lhe.gz/_truth.h5} \
${OUTNAME/.lhe.gz/_truth_cut.h5}


# Create image representations of the events
python gen_data/process_truth_to_images.py \
--infile ${OUTNAME/.lhe.gz/_truth_cut.h5} \
--outfile ${OUTNAME/.lhe.gz/_truth_cut_images.h5}


deactivate

echo ls -lh ${OUTNAME/.lhe.gz/_truth.h5} ${OUTNAME/.lhe.gz/_truth_cut.h5} ${OUTNAME/.lhe.gz/_truth_cut_images.h5}
ls -lh ${OUTNAME/.lhe.gz/_truth.h5} ${OUTNAME/.lhe.gz/_truth_cut.h5} ${OUTNAME/.lhe.gz/_truth_cut_images.h5}
