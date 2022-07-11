# run an nn testing example
# usage: source example_bdt.sh

make env_nn/bin/activate
source env_nn/bin/activate

python example_nn.py \
--datapath pv_msme_3j_4j_1_seed_80_truth_cut.h5 \
--modelpath results/models/jet_net_truth/liv_3j_4j_1/

deactivate
