SHELL := /bin/bash

env_basic/bin/activate:
	python3 -m venv env_basic
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements_basic.txt \
	)

env_bdt/bin/activate:
	python3 -m venv env_basic
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements_bdt.txt \
	)

env_nn/bin/activate:
	python3 -m venv env_basic
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements_nn.txt \
	)
