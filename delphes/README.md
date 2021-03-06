This code runs in a ROOT environment for example set up with
```bash
setupATLAS
lsetup "root 6.22.00-python3-x86_64-centos7-gcc8-opt"
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For setup of delphes with pythia follow  https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/Pythia8
```bash
# download delphes
wget http://cp3.irmp.ucl.ac.be/downloads/Delphes-3.5.0.tar.gz
tar -zxf Delphes-3.5.0.tar.gz

# download pythia8235.tgz from https://pythia.org/releases/
wget https://pythia.org/download/pythia82/pythia8235.tgz
tar -zxf pythia8235.tgz

# compile pythia
cd pythia8235
./configure --prefix=$(pwd)
make -j $(nproc --all) install

# compile delphes
cd ../Delphes-3.5.0
export PYTHIA8=../pythia8235
make -j $(nproc --all) HAS_PYTHIA8=true

cd ..
```

For running delphes reconstruction and processing to images and arrays of jets:
```bash
python run_delphes.py
```
This is set up to be able to run on a HTcondor batch system, so involves copying of files


(These steps are combined in `example_delphes.sh`.)
