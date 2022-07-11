# run all as described in delphes/README.md
# usage: source example_delphes.sh
setupATLAS
lsetup "root 6.22.00-python3-x86_64-centos7-gcc8-opt"
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

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

python run_delphes.py


