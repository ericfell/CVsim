# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:26:13 2020

@author: erico
"""
import numpy as np
import scipy.constants as spc

F = spc.physical_constants['Faraday constant'][0]
R = spc.R

class OneElectronCV:
    """
    This is a class to simulate cyclic voltammograms for a disk macroelectrode
    for one electron processes.
    
    Algorithm Reference:
    [1] Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, Electrochimica Acta, 56, 2011, 10612-10625. 
    """  
   
    def __init__(self, E_start, E_switch, E_not, scanrate, mV_step, c_bulk, 
                 diff_r, diff_p, disk_radius, temperature):
        """
        Inputs that define the CV setup, and are shared by all reaction
        mechanism functions available for simulation.
        """
        self.E_start = E_start    # starting potential (V)
        self.E_switch = E_switch  # switching potential (V)
        self.E_not = E_not        # standard reduction potential (V)
        self.scanrate = scanrate  # scanrate (V/s) 
        self.potential_step = (mV_step / 1000)  # potential step (V)   
        self.delta_t = (self.potential_step / self.scanrate) # time step (s)
        self.c_bulk = c_bulk   # bulk [reactant] (mM or mol/m^3) 
        self.diff_r = (diff_r / 1e4)  # D coefficient of reactant (m^2/s) 
        self.diff_p = (diff_p / 1e4)  # D coefficient of product (m^2/s) 
        self.D_ratio = np.sqrt(self.diff_r / self.diff_p)
        self.D_const = np.sqrt(self.diff_r / self.delta_t)
        self.area = np.pi*((disk_radius / 1000)**2)  # Electrode area (m^2)         
        self.temperature = temperature  # Kelvin
        self.N_max = int(np.abs(E_switch - E_start)*2 / self.potential_step) #number of points 
    ##########################################################################    
    def voltage_profile(self):
        """
        Return potential steps for voltage profile and for exponential 
        Nernstian/Butler-Volmer function.
        """
        potential = np.array([])
        E_func = np.zeros(self.N_max)
        const = -F / (R*self.temperature)
        if self.E_start < self.E_switch: #defines reduction or oxidation first
            self.direction = -1
        else:
            self.direction = 1     
        delta_theta = self.direction*self.potential_step
        Theta = (self.E_start - delta_theta)          
        for k in range(1, self.N_max + 1): 
            potential = np.append(potential, Theta)               
            #exponential potential function
            E_func[k-1] = (np.exp(const*self.direction*(self.E_switch 
                           + self.direction*abs((k*self.potential_step) 
                           + self.direction*(self.E_switch - self.E_start)) 
                           - self.E_not)))
            if k < (int(self.N_max/2)):
                Theta -= delta_theta
            else:
                Theta += delta_theta    
        return potential, E_func    
    ########################################################################## 
    def sum_function(self): 
        """Return weighting factors for semi-integration method."""
        W_n = np.ones(self.N_max)
        for i in range(1, self.N_max):
            W_n[i] = (2*i - 1)*( W_n[i-1] / (2*i))      
        return W_n
    ##########################################################################
    ##########################################################################
    def reversible(self):
        """
        Return current-potential profile for reversible (Nernstian), 
        one electron transfer (E_r).
        """
        W_n = self.sum_function()
        potential, E_func = self.voltage_profile()
        current = np.zeros(self.N_max)
        constant = (-F*self.direction*self.area*self.c_bulk*self.D_const)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N-1] = (constant / (1 + (self.D_ratio / E_func[N-1]))) 
            else:
                summ = sum(W_n[k]*current[N-k-1] for k in range(1,N))                
                current[N-1] = ((constant / (1 + (self.D_ratio / E_func[N-1])))
                                - summ)  
        return potential, current 
    ##########################################################################
    ##########################################################################
    def quasireversible(self, alpha, k_not):
        """
        Return current-potential profile for quasi-reversible, one electron
        transfer (E_q). Requires input of alpha and k_not (cm/s).
        """
        k_not = k_not / 100
        W_n = self.sum_function()
        potential, E_func = self.voltage_profile()
        current = np.zeros(self.N_max) 
        constant = (-F*self.direction*self.area*self.c_bulk*self.D_const)
        for N in range(1, self.N_max + 1): 
            if N == 1:
                current[N-1] = (constant / (1 + (self.D_ratio / E_func[N-1])
                                + (self.D_const / (np.power(E_func[N-1],alpha)
                                * k_not))))
            else:
                summ = sum(W_n[k]*current[N-k-1] for k in range(1,N)) 
                current[N-1] = ((constant - (1 + (self.D_ratio / E_func[N-1]))
                                * summ) / (1 + (self.D_ratio / E_func[N-1]) 
                                + (self.D_const / (np.power(E_func[N-1],alpha)
                                * k_not))))
        return potential, current
    ##########################################################################
    ##########################################################################
    def quasireversible_chemical(self, alpha, k_not, k_forward, k_backward):
        """
        Return current-potential profile for quasi-reversible, one electron
        transfer followed by homogeneous chemical kinetics (E_q C).
        Requires input of alpha, k_not (cm/s), k_for (1/s), k_back (1/s).
        """
        k_not = k_not / 100
        k_sum = k_forward + k_backward
        big_K = k_forward / k_backward
        W_n = self.sum_function()
        potential, E_func = self.voltage_profile()
        current = np.zeros(self.N_max)    
        constant = (-F*self.direction*self.area*self.c_bulk*self.D_const)
        for N in range(1, self.N_max + 1): 
            if N == 1:
                current[N-1] = (constant / (1 + (self.D_const 
                                / (np.power(E_func[N-1], alpha)*k_not))
                                + (self.D_ratio / E_func[N-1])))
            else:
                summ = sum(W_n[k]*current[N-k-1] for k in range(1,N)) 
                summ_exp = sum((W_n[k]*current[N-k-1]*np.exp((-k-1)*k_sum)) 
                               for k in range(1,N))               
                current[N-1] = ((constant - ((1 + (self.D_ratio / ((1 + big_K)
                                * E_func[N-1])))*summ) - (((big_K*self.D_ratio) 
                                / (E_func[N-1]*(1 + big_K)))*summ_exp)) / (1 
                                + (self.D_const / (np.power(E_func[N-1], alpha)
                                * k_not)) + (self.D_ratio / E_func[N-1])))
        return potential, current       
##############################################################################
if __name__ == '__main__':
    print('testing')