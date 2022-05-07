SHELL := /bin/bash

.PHONY: help
help:
	@echo "TODO"


.PHONY: clean
clean:
	@echo "TODO"
	rm -f *.lhe


# MadGraph parton-level simulation
pv_msme_lambda%.lhe:
	echo $* > $@
