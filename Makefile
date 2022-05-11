SHELL := /bin/bash

env_basic/bin/activate:
	python3 -m venv env_basic
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install \
	scipy==1.8.0 \
	numpy==1.22.3 \
	matplotlib==3.5.2 \
	h5py==3.6.0 \
	torch==1.11.0 \
	)
