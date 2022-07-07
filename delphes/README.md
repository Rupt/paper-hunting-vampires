This code runs in a ROOT 6.14 environment for example set up with
```
setupATLAS
lsetup "root 6.14.08-x86_64-centos7-gcc8-opt"
python3 -m venv env
	source env/bin/activate
	python3 -m pip install \
        uproot==4.2.0 \
		pandas==1.3.5 \
		h5py==3.6.0 \
		matplotlib==3.5.1 \
		awkward==1.7.0
```

For setup of delphes with pythia follow  https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/Pythia8
```
#download delphes
wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.5.0.tar.gz
tar -zxf Delphes-3.5.0.tar.gz

#Download  pythia8235.tgz from https://pythia.org/releases/
tar xzvf pythia8235.tgz
cd pythia8235
./configure --prefix=/usera/dnoel/Documents/parity/generation/pythia8235
make install

#recompile delphes
cd../Delphes-3.5.0
export PYTHIA8=/usera/dnoel/Documents/parity/generation/pythia8235
make HAS_PYTHIA8=true
```


For running delphes reconstruction and processing to images and arrays of jets:
```python run_delphes.py```
This is set up to be able to run on a HTcondor batch system, so involves copying of files


