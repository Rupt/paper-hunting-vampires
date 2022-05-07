SHELL := /bin/bash

env_basic/bin/activate:
	python3 -m venv env_basic
	( \
	source env_basic/bin/activate; \
	pip install --upgrade pip; \
	pip install scipy==1.8.0 numpy==1.22.3 matplotlib==3.5.2; \
	)
