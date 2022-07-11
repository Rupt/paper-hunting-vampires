SHELL := /bin/bash

env_basic/bin/activate:
	python3 -m venv env_basic
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements_basic.txt \
	)

env_bdt/bin/activate:
	python3 -m venv env_bdt
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements_bdt.txt \
	)

env_nn/bin/activate:
	python3 -m venv env_nn
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements_nn.txt \
	)

pv_msme_0p5.lhe.gz:
	source example_pv_msme_lhe.sh 0.5

pv_msme_3j_4j_1_seed_80_truth_cut.h5:
	source example_truth_jet.sh example_results/pv_msme_3j_4j_1_seed_80.lhe.gz
