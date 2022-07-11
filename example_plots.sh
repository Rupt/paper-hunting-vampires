make env_basic/bin/activate
source env_basic/bin/activate

mkdir -p plots

python plot/plot_ht.py
python plot/plot_pt.py
python plot/plot_quality_sub.py
python plot/plot_parity.py
python plot/plot_parity_transformed.py
python plot/plot_two_bin_stats.py
python plot/plot_rings.py
python plot/plot_kinematics.py
PYTHONPATH=$PYTHONPATH:parity_tests python plot/plot_images.py
# TODO currently broken due to missing a input file
# python plot/plot_cnn.py

deactivate

echo ls -lh plots
ls -lh plots
