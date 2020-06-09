# CVsim
Semi-analytical method<sup>1</sup> for simulating cyclic voltammograms on a disk macroelectrode, using a semiintegration algorithm. In the limit of infinitely small potential steps, this algorithm is an exact solution. Due to the precision of standard potentiostats simulations using potential steps of 1-5 mV typically provide a reasonable accuracy-computing time trade-off, where accuracy sanity checks (e.g. Randles-Sevcik relationship for E<sub>r</sub> and E<sub>q</sub> mechanisms) have been performed.

### Files
1. `one_electron_CV.py` provides the *OneElectronCV* class for the E<sub>r</sub> , E<sub>q</sub> , and E<sub>q</sub>C schemes
2. `two_electron_CV.py` provides the *TwoElectronCV* class for the E<sub>q</sub>E<sub>q</sub> , and square schemes
3. `test_plots_fits.py` provides quick examples of a) calling the *OneElectronCV* or *TwoElectronCV* class to simulate and plot mechanistic schemes, and b) fitting real/simulated data (likely a few ways to do this, some more forgiving than others)
4. `one_electron_multiscan.py` provides the *OneElectronCV_multi* class which enables multiple-scan simulation (pseudo steady state) of schemes contained in *OneElectronCV*

**Input Parameter Units**
- E<sub>start</sub>/E<sub>switch</sub>/E<sup>o</sup> = V
- Scanrate = V/s
- Potential Step = mV
- Active species concentration = mM (mol/m<sup>3</sup>)
- Diffusion coefficients = cm<sup>2</sup>/s
- Disk radius = mm
- Temperature = K

**and if needed**
- Standard rate constant, k<sub>o</sub> = cm/s
- 1<sup>st</sup> order chemical rate constants (k<sub>forward</sub>, k<sub>backward</sub>) = s<sup>-1</sup>




[1] Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, *Electrochimica Acta*, **56**, 2011, 10612-10625.

*The schemes for CE<sub>r</sub> , catalytic C'E<sub>q</sub> , and E<sub>r</sub>CE<sub>r</sub> are currently in development
