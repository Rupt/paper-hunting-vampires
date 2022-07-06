SHELL := /bin/bash

env_basic/bin/activate:
	python3 -m venv env_basic
	( \
	source $@; \
	pip install --upgrade pip; \
	pip install -r requirements.txt \
	)
