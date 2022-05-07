# Usage:
# source example_truth_jet.sh ${LAMBDAPV}
# e.g.
# source example_truth_jet.sh 0.5
LAMBDAPV=$1

if [ ! ${LAMBDAPV} ]
then
    echo "Usage: source example_truth_jet.sh \${LAMBDAPV}"
    return
fi

make env_basic/bin/activate
source env_basic/bin/activate


# Read the xml-based lhe file into an array format
python gen_data/process_lhe_to_h5.py \
pv_msme_${LAMBDAPV/\./p}.lhe.gz \
pv_msme_${LAMBDAPV/\./p}_truth.h5


# Apply selection cuts (3jet > 220 GeV |eta| < 2.8 )
python gen_data/process_jet_cuts.py \
pv_msme_${LAMBDAPV/\./p}_truth.h5 \
pv_msme_${LAMBDAPV/\./p}_truth_cut.h5


# Create image representations of the events
python gen_data/process_truth_to_images.py \
--infile pv_msme_${LAMBDAPV/\./p}_truth_cut.h5 \
--outfile pv_msme_${LAMBDAPV/\./p}_truth_cut_images.h5


deactivate
