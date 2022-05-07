# Simulate lorentz symmetry violating data


### Environment from an ATLAS system.
```
source setup.sh

```


### Run MadGraph interactive
```
./mg5_aMC

```


### Make standard model jjj
```
# make the output for simulating events
python madcontrol.py output process/standard_jjj

# make the python module for evaluating matrix elements
python madcontrol.py output_standalone standalone/standard_jjj

```


### Make liv model jjj with zero couplings
```
# prepare json parameters
python -c "import configure; configure.dump_zero(\"parameter/example_zero.json\")"

# prepare the model with those parameters
python configure.py parameter/example_zero.json MG5_aMC_v3_3_0/models/liv_zero

# make the output for simulating events
python madcontrol.py output process/liv_zero_jjj --model liv_zero

# make the python module for evaluating matrix elements
python madcontrol.py output_standalone standalone/liv_zero_jjj --model liv_zero

```


### Make liv model jjj with random couplings
```
python -c "import configure; configure.dump_random(\"parameter/example_random.json\")"

python configure.py parameter/example_random.json MG5_aMC_v3_3_0/models/liv_random

python madcontrol.py output process/liv_random_jjj --model liv_random

python madcontrol.py output_standalone standalone/liv_random_jjj --model liv_random

```


### Demonstrate matrix element calculations
```
python demo_matrix_element.py

python demo_violation.py

python rings.py

```


### Generate MadGraph events

Selected for Atlas-like trigger.

```
python madcontrol.py launch process/standard_jjj \
    --name standard_jjj_20k \
    --seed 1 \
    --nevents 20_000 \
    --kwargs '{"ptj3min": 200, "etaj": 3.2}' \
    --ncores 16


python madcontrol.py launch process/liv_random_jjj \
    --name random_jjj_20k \
    --seed 1 \
    --nevents 20_000 \
    --kwargs '{"ptj3min": 200, "etaj": 3.2}' \
    --ncores 16

```

### Searching for strong parity violation

All axial, "rotation", in `parameter/axial1.json`:

```
{
	"q0300": "1",
	"q1200": "1",
	"q2100": "-1",
	"q3000": "-1",
	"u0300": "-1",
	"u1200": "-1",
	"u2100": "1",
	"u3000": "1",
	"d0300": "-1",
	"d1200": "-1",
	"d2100": "1",
	"d3000": "1"
}
```

```
python -c "import configure; configure.dump_random(\"parameter/example_random.json\")"

python configure.py parameter/axial1.json MG5_aMC_v3_3_0/models/liv_axial1

python madcontrol.py output process/liv_axial1_jjj --model liv_axial1

python madcontrol.py output_standalone standalone/liv_axial1_jjj --model liv_axial1

```
