# Software for:<br>Hunting for vampires and other unlikely forms of parity violation at the Large Hadron Collider
https://arxiv.org/abs/2205.09876

# Usage
Use a Linux environment with a recent version of **python 3**.

For compatibility, it may be helpful to start from a clean conda environment. \
The bundled version of MadGraph does **not** work with python 3.10.
```bash
conda create -n blank python==3.9.12
```
then to set up
```bash
conda activate blank
```


# Generate paper plots
All plots used in the paper (and some others) are produced from serialized data
which are included in this repository. \
To jump into the plotting environment and reproduce those plots, execute:
```bash
source example_plots.sh
```

# Simulate a PV-mSME .lhe file
Choosing $\lambda_\textrm{PV} = 0.5$ as an example. \
Again, this hops into an environment. \
It then:
* generates the mSME model for Madgraph,
* runs Madgraph to simulate **3-jet** events under our kinematic selections (4-jet is very slow), and
* moves the lhe to the local directory as `pv_msme_0p5.lhe.gz`.

We also leave lots of mess behind in the liv/ directory; you can use `git diff' to see what's there. \
Execute:
```bash
source example_pv_msme_lhe.sh 0.5
```

# Extract truth-jet data
```bash
source example_truth_jet.sh pv_msme_0p5.lhe.gz
```


# Run Delphes reconstruction
Follow environment setup in Delphes/README.md then
```python run_delphes.py```
