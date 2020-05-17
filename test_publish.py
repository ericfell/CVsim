# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:51:46 2020

@author: erico
"""
###should provide example code of: 1) plotting a few things, 2) fitting data
import numpy as np
import matplotlib.pyplot as plt
from one_electron_CV import OneElectronCV
from two_electron_CV import TwoElectronCV
import scipy.constants as spc
from scipy.optimize import curve_fit

F = spc.physical_constants['Faraday constant'][0]
R = spc.R

# Part A: how to simulate schemes

# example 1 electron schemes
setup = OneElectronCV(-0.4, 0.4, 0, 1, 1, 1, 1e-6, 1e-6, 5, 298)
potential, current = setup.reversible()
potential2, current2 = setup.quasireversible(0.4, 1e-3)
# example 2 electron scheme
setup2 = TwoElectronCV(0.5, -0.5, 0.1, -0.05, 1, 1, 1, 1e-6, 1e-6, 1e-6, 5, 298)
potential3, current3 = setup2.quasireversible(0.5, 0.5, 1e-3, 1e-4)

plt.figure(1)
plt.plot(potential, current, label='$E_{r}$')
plt.plot(potential2, current2, label='$E_{q}$')
plt.plot(potential3, current3, label='$E_{q}E_{q}$')
plt.xlabel('Potential (V)')
plt.ylabel('Current (mA)')
plt.legend()
plt.show()
##############################################################################
##############################################################################
# Part B: how to fit experimental data

#for simplicity an 'experiment' dataset is simulated and then fit. Ideally you
#would input real data
################## 1) 'experiment' data ######################################
test1 =  OneElectronCV(0.3, -0.3, 0, 1, 1, 1, 1.5e-6, 1.1e-6, 5, 298)
potential4, current4 = test1.quasireversible(0.5, 1e-4) 
plt.figure(2)
plt.plot(potential4, current4, label='experiment')
################## 2) test data into real function############################
##or turn into function of time???
pot2, pot3 = np.split(potential4, 2)
n_pot = pot2 - (abs(potential4[0]) + abs(potential4[-1]) + 0.001)#make function, this must be total range scanned, first pass
new_p = np.concatenate((pot2, n_pot))   
plt.figure(3)
plt.plot(new_p, current4, label='experiment')
################## 3) fit curve to data of real function########################
def func(x, a, b, c):
    Esrt = x[-1]
    Erev = np.split(x, 2)[0][-1]
    setup1 =  OneElectronCV(Esrt, Erev, 0, 1, 1, 1, a, b, 5, 298)
    potential_fit, current_fit = setup1.quasireversible(0.5, c) 
    return current_fit

popt, pcov = curve_fit(func, potential4, current4, bounds=(0, [1e-5, 1e-5, 1e-3]))#p0=(1e-9, 2e-9, 0.09)
fitted_current = func(potential4, *popt)
perr = np.sqrt(np.diag(pcov)) # calculates one standard deviation of the parameters

plt.figure(3)
plt.plot(new_p, fitted_current, label='Fit')
plt.legend()
plt.show()
##############4) turn fit back into CV form
plt.figure(2)
plt.plot(potential4, fitted_current, '--', label='Fit')

plt.figure(4)
plt.plot(new_p, (current4 - fitted_current))# / current4)*100)
plt.ylabel('% Error (exp - fit)')

#popt gives fitted vals