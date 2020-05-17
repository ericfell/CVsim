# CVsim
Semi-analytical method[1] for simulating CVs on a disk macroelectrode, using a semiintegration algorithm.

**Files**
- one_electron_CV.py provides the *OneElectronCV* class for the E<sub>r</sub>, E<sub>q</sub>, and E<sub>q</sub>C schemes

**Input Parameter Units**
- E<sub>start</sub>/E<sub>switch</sub>/E<sup>o</sup> = V
- Scanrate = V/s
- Potential Step = mV
- Active species concentration = mM (mol/m<sup>3</sup>)
- Diffusion coefficients = cm<sup>2</sup>/s
- Disk radius = mm
- Temperature = K

**(If needed)**
- k<sub>o</sub> = cm/s
- 1<sup>st</sup> order chemical rate constants = 1/s


[1] Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, *Electrochimica Acta*, **56**, 2011, 10612-10625.
