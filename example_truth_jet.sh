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

python gen_data/process_lhe_to_h5.py \
pv_msme_${LAMBDAPV/\./p}.lhe.gz \
pv_msme_${LAMBDAPV/\./p}_truth.h5




deactivate
