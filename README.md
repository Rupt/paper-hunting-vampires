# Supporting data and code for:<br>Hunting for vampires and other unlikely forms of parity violation at the Large Hadron Collider
arxiv: https://arxiv.org/abs/2205.09876
zenodo: https://doi.org/10.5281/zenodo.6827724


# Download datasets
You do not need to download the complete datasets to use this repository. But you may wish to, for example, to train and test your own models.

This repository includes:
- all data to reproduce all plots in the paper (serialized results, not raw input data),
- serialized versions of our trained BDT and NN models,
- some small example datasets.
Larger datasets are shared on [Zenodo](https://zenodo.org/). Thanks for hosting them, Zenodo!

Each run of MadGraph produces an lhe file which includes partonic truth and various metadata. For each model (meaning each lambdaPV or rotated coupling matrix), we are one example lhe file in the
- [lhe files dataset](https://doi.org/10.5281/zenodo.6527112).

We share complete train, validation, and test datasets for
- [truth-jet, reco-jet, and rotated truth-jet data](https://doi.org/10.5281/zenodo.6822267).

The calo-image datasets are very large. We share two examples:
- [calo-image datasets for lambdaPV=1](https://doi.org/10.5281/zenodo.6823457), and
- [calo-image datasets for the standard model](https://doi.org/10.5281/zenodo.6826628).


# Use this repository
Use a Linux environment with a recent version of **python 3**.

For compatibility, it may be helpful to start from a clean conda environment. \
The bundled version of MadGraph does **not** work with python 3.10. \
We also require gfortran for MadGraph and texlive for plotting.
```bash
conda create -n hunting-vampires -c conda-forge python==3.9.12 gfortran==12.1.0
```
Then to set up
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
* generates the mSME model for MadGraph,
* modifies the MadGraph code to evaluate matrix elements in the lab frame(*),
* runs MadGraph to simulate 3-partonic-jet events under our kinematic selections (4-jet is very slow), and
* moves the lhe to the local directory as `pv_msme_0p5.lhe.gz`.

(*) MadGraph is modified by the `--lab` arguments to `lib/madcontrol.py`.
The modification is implemented in `liv/use_lab_frame.py`,
and implements a Lorentz boost in each `liv/process/${PROCESS}/SubProcesses/*/auto_dsig?.f` file.

We also leave lots of mess behind in the liv/ directory; you can use `git diff` to see what's there.

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
This prints out a json-formatted report of its results, in which:
* `ntest` is the number of testing data,
* `log_r_test` is the model-versus-symmetry $\log$-likelihood ratio, which equals $nQ$, and
* `quality` is $Q$ with its standard mean and standard deviation estiamtes.

Many other models are saved in `results/models/`. \
Modify the paths given in `example_bdt.sh` as arguments to `example_bdt.py` to test them, too!


# Test an NN model
Lead a serialized NN model and test it against the $\lambda_\textrm{PV} = 1$
sample that we (could have) generated above.
```bash
source example_nn.sh
```
Just as for the BDT, this prints out a json-formatted report of its results, in which:
* `ntest` is the number of testing data,
* `log_r_test` is the model-versus-symmetry $\log$-likelihood ratio, which equals $nQ$, and
* `quality` is $Q$ with its standard mean and standard deviation estimates.

Many other models are saved in `results/models/`. \
Modify the paths given in `example_nn.sh` as arguments to `example_nn.py` to test them, too!

We don't attempt to set up a GPU; you can ignore the warning
`WARNING:absl:No GPU/TPU found, falling back to CPU...`.


# Rotated PV-mSME extras
We include plots of cross-sections and Q split by hour in a notebook.

The split-Q plot uses data shared on Zenodo. \
To run it yourself, download `truth-jet-rot*` from https://zenodo.org/record/6822267 and unzip them at a common path. \
In the notebook, update the line `DATAPATH = "..."` in cell 8 to point to these unzipped data.

To launch it:
```bash
make env_nn/bin/activate
source env_nn/bin/activate
jupyter notebook
# in the notebook web interface, open: example_rotated_pv_msme.ipynb
```

# Run Delphes reconstruction
Follow instructions in `delphes/README.md` for environment setup and execution.
(`cd delphes`)
