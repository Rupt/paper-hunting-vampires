# Generate a sample of events from the PV-mSME with lambdaPV set to:
LAMBDAPV=0.5


SCATTERING_PROCESS='#p p > j j j#'
# for 3jet and 4jet combined, use:
# SCATTERING_PROCESS='"p p > j j j" "p p > j j j j"


# make the PV-mSME coupling matrices
sed s/_/${LAMBDAPV}/g liv/parameter/pv_msme_template.json \
> liv/parameter/pv_msme_${LAMBDAPV/\./p}.json

# make the PV-mSME MadGraph model
python liv/configure.py \
liv/parameter/pv_msme_${LAMBDAPV/\./p}.json \
liv/MG5_aMC_v3_3_0/models/pv_msme_${LAMBDAPV/\./p}

# make the PV-mSME MadGraph process (slow)
# with matrix elements evaluated in the lab frame
# we using only 3 jets to make this example run faster
# tp add the fourth, replace the line beginning --process with:
# --process "p p > j j j" "p p > j j j j"
( \
cd liv; \
python madcontrol.py output process/pv_msme_${LAMBDAPV/\./p} \
--lab \
--process "p p > j j j" \
--model pv_msme_${LAMBDAPV/\./p} \
)

# simulate some events
( \
cd liv; \
python madcontrol.py launch process/pv_msme_${LAMBDAPV/\./p} \
--name example \
--seed 5678 \
--nevents 10_000 \
--kwargs '{"ptj3min": 200, "etaj": 3.2, "ickkw": 0, "xqcut": 0, "ktdurham": 30, "ptj": 20, "mmjj": 0}' \
--ncores $((($(nproc --all) + 1) / 2)) \
)

# grab the zipped lhe
mv \
liv/process/pv_msme_${LAMBDAPV/\./p}/Events/example/unweighted_events.lhe.gz \
pv_msme_${LAMBDAPV/\./p}.lhe.gz

ls -lh pv_msme_${LAMBDAPV/\./p}.lhe.gz
