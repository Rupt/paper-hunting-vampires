# Supporting data and code for:<br>Hunting for vampires and other unlikely forms of parity violation at the Large Hadron Collider
https://arxiv.org/abs/2205.09876

# Usage
Use a Linux environment with a recent version of **python 3**.

For compatibility, it may be helpful to start from a clean conda environment. \
The bundled version of MadGraph does **not** work with python 3.10. \
We also require gfortran for MadGraph and texlive for plotting.
```bash
conda create -n hunting-vampires -c conda-forge python==3.9.12 gfortran==12.1.0
```
then to set up
```bash
conda activate hunting-vampires
```

Some LaTeX context is also required for matplotlib, but I have failed to
install it through conda (texlive-core doesn't work).
From a CERN-linked environment, for example, you can link texlive with `export PATH=/cvmfs/sft.cern.ch/lcg/external/texlive/2020/bin/x86_64-linux:$PATH`.

# Generate paper plots
All plots used in the paper (and some others) are produced from serialized data
which are included in this repository. \
To jump into the plotting environment and reproduce those plots, execute:
```bash
source example_plots.sh
```

# Simulate a PV-mSME lhe file
Choosing $\lambda_\textrm{PV} = 0.5$ as an example. \
Again, this hops into an environment. \
It then:
* generates the mSME model for Madgraph,
* modifies the Madgraph code to evaludate matrix elements in the lab frame(*),
* runs Madgraph to simulate 3-partonic-jet events under our kinematic selections (4-jet is very slow), and
* moves the lhe to the local directory as `pv_msme_0p5.lhe.gz`.

(*) Madgraph is modified by the `--lab` arguments to `lib/madcontrol.py`.
The modification is implemented in `liv/use_lab_frame.py`,
and implements a Lorentz boost in each `liv/process/${PROCESS}/SubProcesses/*/auto_dsig?.f` file.

We also leave lots of mess behind in the liv/ directory; you can use `git diff' to see what's there.

Execute:
```bash
source example_pv_msme_lhe.sh 0.5
```

# Extract truth-jet lhe data to h5
For later use, we convert parts of the lhe file
([XML](http://harmful.cat-v.org/software/xml/)) to an efficient format.

This script first converts it to padded four-momenta in an
[h5](http://www.h5py.org/) file `pv_msme_0p5_truth.h5`. \
It then applies kinematic cuts to produce `pv_msme_0p5_truth_cut.h5`. \
It then converts that to the image representation in `pv_msme_0p5_truth_cut_images.h5`.

Execute:
```bash
source example_truth_jet.sh pv_msme_0p5.lhe.gz
```
(Note that the `0.5` above has become `0p5` here.)

An example lhe (from the test set) with $\lambda_\textrm{PV} = 1$ and up to four partonic jets
is included as `example_results/pv_msme_3j_4j_1_seed_80.lhe.gz`. \
Extract it by executing:
```bash
source example_truth_jet.sh example_results/pv_msme_3j_4j_1_seed_80.lhe.gz
```

# Test a BDT model
Lead a serialized BDT model and test it against the $\lambda_\textrm{PV} = 1$
sample that we (could have) generated above.
```bash
source example_bdt.sh
```
This prints out a json-formatted report of its results in which:
* `ntest` is the number of testing data,
* `log_r_test` is the model-versus-symmetry $\log$-likelihood ratio, which equals $nQ$, and
* `quality` is $Q$ with its standard mean and standard deviation estiamtes.

Many other models are saved in `results/models/`. \
Modify the paths given in `example_bdt.sh` as arguments to `example_bdt.py` to test them too!

# Test an NN model
Lead a serialized NN model and test it against the $\lambda_\textrm{PV} = 1$
sample that we (could have) generated above.
```bash
source example_nn.sh
```
Just as for the BDT, this prints out a json-formatted report of its results in which:
* `ntest` is the number of testing data,
* `log_r_test` is the model-versus-symmetry $\log$-likelihood ratio, which equals $nQ$, and
* `quality` is $Q$ with its standard mean and standard deviation estiamtes.

Many other models are saved in `results/models/`. \
Modify the paths given in `example_nn.sh` as arguments to `example_nn.py` to test them too!

We don't attempt to set up a GPU; you can ignore the warning
`WARNING:absl:No GPU/TPU found, falling back to CPU...`.


# Run Delphes reconstruction
Follow instructions in `delphes/README.md` for environment setup and execution.
(`cd delphes`)
