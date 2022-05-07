SHELL := /bin/bash

.PHONY: help
help:
	@echo "TODO"


# an lhe file MadGraph parton-level simulation
pv_msme_%.lhe: liv/process/pv_msme_%/README
	echo $* > $@

# a PV-mSME MadGraph process
.PRECIOUS: liv/process/pv_msme_%/README
liv/process/pv_msme_%/README: liv/MG5_aMC_v3_3_0/models/pv_msme_%/__init__.py
	( \
	cd liv; \
	python madcontrol.py output process/pv_msme_$* --model pv_msme_$* \
	)

# a PV-mSME MadGraph model
.PRECIOUS: liv/MG5_aMC_v3_3_0/models/pv_msme_%/__init__.py
liv/MG5_aMC_v3_3_0/models/pv_msme_%/__init__.py:
	( \
	cd liv; \
	python configure.py parameter/pv_msme_$*.json MG5_aMC_v3_3_0/models/pv_msme_$*; \
	)


foo_%.txt: bar_%.txt
	touch foo_$*.txt

bar_%.txt:
	touch bar_$*.txt

nada: bar_*.txt
