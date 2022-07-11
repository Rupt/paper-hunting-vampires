# run a bdt testing example
# usage: source example_bdt.sh

make env_bdt/bin/activate
source env_bdt/bin/activate

python example_bdt.py \
--datapath pv_msme_3j_4j_1_seed_80_truth_cut.h5 \
--modelpath results/models/jet_bdt_truth/liv_3j_4j_1/

deactivate
