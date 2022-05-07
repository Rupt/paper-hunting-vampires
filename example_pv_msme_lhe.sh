# Generate a sample of events from the PV-mSME with lambdaPV set to
# the argument:
# Usage:
# source example_pv_msme_lhe.sh ${LAMBDAPV}
# e.g.
# source example_pv_msme_lhe.sh 0.5
LAMBDAPV=$1

if [ ! ${LAMBDAPV} ]
then
    echo "Usage: source example_pv_msme_lhe.sh \${LAMBDAPV}"
    return
fi


make env_basic/bin/activate
source env_basic/bin/activate


# Make the PV-mSME coupling matrices.
sed s/_/${LAMBDAPV}/g liv/parameter/pv_msme_template.json \
> liv/parameter/pv_msme_${LAMBDAPV/\./p}.json


# Make the PV-mSME MadGraph model.
python liv/configure.py \
liv/parameter/pv_msme_${LAMBDAPV/\./p}.json \
liv/MG5_aMC_v3_3_0/models/pv_msme_${LAMBDAPV/\./p}


# Make the PV-mSME MadGraph process (slow),
# with matrix elements evaluated in the lab frame.
# We using only 3 jets to make this example run faster.
# To add the fourth, replace the line beginning --process with:
# --process "p p > j j j" "p p > j j j j"
( \
cd liv; \
python madcontrol.py output process/pv_msme_${LAMBDAPV/\./p} \
--lab \
--process "p p > j j j" \
--model pv_msme_${LAMBDAPV/\./p} \
)


# Simulate some events (slower).
( \
cd liv; \
python madcontrol.py launch process/pv_msme_${LAMBDAPV/\./p} \
--name example \
--seed 1 \
--nevents 10_000 \
--kwargs '{"ptj": 200, "etaj": 3.2, "ickkw": 0, "xqcut": 0, "ktdurham": 200}' \
--ncores $((($(nproc --all) + 1) / 2)) \
)


# Grab the zipped lhe.
mv \
liv/process/pv_msme_${LAMBDAPV/\./p}/Events/example/unweighted_events.lhe.gz \
pv_msme_${LAMBDAPV/\./p}.lhe.gz


echo pv_msme_${LAMBDAPV/\./p}.lhe.gz
ls -lh pv_msme_${LAMBDAPV/\./p}.lhe.gz
