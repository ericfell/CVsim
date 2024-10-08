
import numpy as np
import matplotlib.pyplot as plt
from mechanisms import E_rev, E_q, E_qC, EE, SquareScheme
import scipy.constants as spc
from scipy.optimize import curve_fit

F = spc.physical_constants['Faraday constant'][0]
R = spc.R

# Part A: how to simulate schemes

# example 1 electron schemes
potential1, current1 = E_q(0.3, -0.3, 0.0, 1.0, 1.0, 1e-5, 1e-5, 0.3, 1e-3).simulate()
potential2, current2 = E_qC(0.3, -0.3, 0.0, 1.0, 1.0, 1e-5, 1e-5, 0.3, 1e-3, 0.1, 1).simulate()

# example 2 electron scheme
potential3, current3 = EE(-0.5, 0.4, -0.1, 0.1, 1.0, 1, 1e-5, 1e-5, 1e-5, 0.5, 0.5, 1e-2, 5e-3).simulate()

fig, ax = plt.subplots()
ax.plot(potential1, [x*1000 for x in current1], label='$E_{q}$')
ax.plot(potential2, [x*1000 for x in current2], label='$E_{q}C$')
ax.plot(potential3, [x*1000 for x in current3], label='$E_{q}E_{q}$')
ax.xlabel('Potential (V)')
ax.ylabel('Current (mA)')
ax.tight_layout()
ax.legend()
plt.show()
##############################################################################
##############################################################################
# Part B: how to fit experimental data

#for simplicity an 'experiment' dataset is simulated and then fit. Ideally you
#would input real data to fit

################## 1) 'experiment' data 

fig, ax = plt.subplots()
ax.plot(potential1, [x*1000 for x in current1], label='experiment')
plt.show()
################## 2) turn test data into real function
new_p = np.linspace(1, 1200, 1200)
fig, ax = plt.subplots()
ax.plot(new_p, [x*1000 for x in current1], label='experiment')
plt.show()
################## 3) fit curve to data of real function
def func(x, a, b, c): #modify variables as needed
    E_start = x[-1]
    E_reverse = np.round(np.split(x, 2)[0][-1], 4)
    print(E_reverse)
    #here is where you choose what is known and unknown
    setup1 =  OneElectronCV(E_start, E_reverse, 0, 1, 1, 1, a, b, 5, 298)
    potential_fit, current_fit = setup1.quasireversible(0.5, c) 
    return current_fit

popt, pcov = curve_fit(func, potential, current, bounds=(0, [1e-5, 1e-5, 1e-3]))#p0=(1e-9, 2e-9, 0.09)
fitted_current = func(potential, *popt)
sigma = np.sqrt(np.diag(pcov)) # one standard deviation of the parameters
# print fits and std dev. for a,b,c etc.
print([f"{x:.2E} +/- {y:.0E}" for x,y in zip(popt, sigma)])

plt.figure(3)
plt.plot(new_p, [x*1000 for x in fitted_current], '--', label='Fit')
plt.xlabel('Timestep')
plt.ylabel('Current (mA)')
plt.tight_layout()
plt.legend()
plt.show()

############## 4) turn fit back into CV form and overlay experiment data
plt.figure(2)
plt.plot(potential, [x*1000 for x in fitted_current], '--', label='Fit')
plt.xlabel('Potential (V)')
plt.ylabel('Current (mA)')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(4)
plt.plot(new_p, [((x - y)*1e6) for x,y in zip(current, fitted_current)], '--')
plt.xlabel('Timestep')
plt.ylabel("Residuals ($\mu$A) ")
plt.tight_layout()
plt.show()

