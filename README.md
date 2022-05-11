# paper-hunting-vampires
Supporting material for an upcoming paper.

Use a Linux environment with a recent version of **python 3**.

For compatibility, it may be helpful to start from a clean conda environment.
The bundled version of MadGraph appears to not work with python 3.10.
```bash
conda create -n blank python==3.9.12 pytorch==1.11.0
```
then to set up
```bash
conda activate blank
export PYTHONPATH=$PYTHONPATH:parity_tests
```


## Generate paper plots

These all use data serialized in this git repository.

```bash
source example_plots.sh
```


## Simulate a PV-mSME .lhe file with MadGraph

Choosing lambdaPV = 0.5 as an example.

```bash
source example_pv_msme_lhe.sh 0.5
```

## Extract truth-jet data

```bash
source example_truth_jet.sh pv_msme_0p5.lhe.gz
```
