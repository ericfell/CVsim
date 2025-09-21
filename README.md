[![Documentation Status](https://readthedocs.org/projects/cvsim/badge/?version=latest)](https://cvsim.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/ericfell/CVsim/graph/badge.svg?token=90223DIBVS)](https://codecov.io/gh/ericfell/CVsim)  [![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)  [![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

CVsim
--------

`cvsim` is a Python package for cyclic voltammogram (CV) simulation via a semi-analytical method developed by Oldham and Myland<sup>1</sup>. It is valid for CV experiments performed on disk macroelectrodes, and uses a semi-integration algorithm. In the limit of infinitely small potential steps, this algorithm is an exact solution. 
Due to the precision of standard potentiostats, simulations that use a potential step of 1-5 mV typically provide a reasonable accuracy-computing time trade-off, where accuracy sanity checks (e.g. Randles-Sevcik relationship for E<sub>r</sub> and E<sub>q</sub> mechanisms) have been performed.


Currently available mechanisms: 
- One-electron process: E<sub>r</sub> , E<sub>q</sub> , and E<sub>q</sub>C
- Two-electron process: E<sub>q</sub>E<sub>q</sub> and Square scheme


## Installation

`cvsim` can be installed from PyPI using pip:

```bash
pip install cvsim
```

See [Getting started with `cvsim.py`](https://cvsim.readthedocs.io/en/latest/getting-started.html) for instructions on simulating CVs.

## Dependencies

`cvsim` requires:

- Python (>=3.10)
- SciPy


## Package Structure

- `mechanisms.py`: Specifies one-/two-electron process mechanisms to be simulated.
- `fit_curve.py`: Performs fitting of experimental CV data according to a desired mechanism. 


## Example

To simulate a reversible (Nernstian) CV:

```python
from cvsim.mechanisms import E_rev

potential, current = E_rev(
    start_potential=0.3,       # V vs. reference
    switch_potential=-0.5,     # V vs. reference
    reduction_potential=-0.1,  # V vs. reference
    scan_rate=0.1,             # V/s
    c_bulk=1.0,                # mM (mol/m^3)
    diffusion_reactant=1e-6,   # cm^2/s
    diffusion_product=2e-6,    # cm^2/s
).simulate()
```



**Input Parameters**
- E<sub>start</sub>/E<sub>switch</sub>/E<sup>o'</sup> = V vs ref.
- Scanrate = V/s
- Active species concentration = mM (mol/m<sup>3</sup>)
- Diffusion coefficients = cm<sup>2</sup>/s

**Optional Parameters**
- Potential step size = mV
- Disk radius = mm
- Temperature = K

**and if needed**
- Standard rate constant, k<sub>o</sub> = cm/s
- 1<sup>st</sup> order chemical rate constants (k<sub>forward</sub>, k<sub>backward</sub>) = s<sup>-1</sup>




[1] [Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, *Electrochimica Acta*, **56**, 2011, 10612-10625.](https://www.sciencedirect.com/science/article/abs/pii/S0013468611007651)


*The schemes for CE<sub>r</sub> , catalytic C'E<sub>q</sub> , and E<sub>r</sub>CE<sub>r</sub> need development, PRs welcome!


## License
[MIT](https://choosealicense.com/licenses/mit/) 