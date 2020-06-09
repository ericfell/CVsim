
import numpy as np
import scipy.constants as spc

F = spc.physical_constants['Faraday constant'][0]
R = spc.R

class OneElectronCV_multi:
    """
    This is a class to simulate cyclic voltammograms for a disk macroelectrode
    for one electron processes.
    
    Algorithm Reference:
    [1] Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, Electrochimica Acta, 56, 2011, 10612-10625.
    
    """  
   
    def __init__(self, E_start, E_switch, E_not, scanrate, mV_step, c_bulk, 
                 diff_r, diff_p, disk_radius, temperature, scans):
        """
        Inputs that define the CV setup, and are shared by all reaction
        mechanism functions available for simulation.
        
        Parameters
        ----------
        E_start : float
            Starting potential of scan (volts)
        E_switch : float
            Switching potential of scan (volts)
        E_not : float
            Standard reduction potential (volts)
        scanrate : float
            Potential sweep rate (volts / second)
        mV_step : float
            Potential increment of scan (millivolts)
        c_bulk  : float
            Bulk concentration of redox species (mM or mol/m^3) 
        diff_r : float
            Diffusion coefficient of reactant (cm^2/s) 
        diff_p : float
            Diffusion coefficient of product (cm^2/s)
        disk_radius : float
            Radius of disk macroelectrode (mm)
        temperature : float
            Temperature (kelvin)
        scans : integer
            Number of full CV scans desired
            
        """
        self.E_start = E_start    
        self.E_switch = E_switch  
        self.E_not = E_not        
        self.scanrate = scanrate  
        self.potential_step = (mV_step / 1000)     
        self.delta_t = (self.potential_step / self.scanrate) 
        self.c_bulk = c_bulk   
        self.diff_r = (diff_r / 1e4)   
        self.diff_p = (diff_p / 1e4)   
        self.D_ratio = np.sqrt(self.diff_r / self.diff_p)
        self.D_const = np.sqrt(self.diff_r / self.delta_t)
        self.area = np.pi*((disk_radius / 1000)**2)           
        self.temperature = temperature  
        self.scans = scans
        self.N_max = int(np.abs(E_switch - E_start)*2*self.scans / self.potential_step) 
    ##########################################################################    
    def voltage_profile(self):
        """
        Return potential steps for voltage profile and for exponential 
        Nernstian/Butler-Volmer function of arbitrary number of scans.
        
        Parameters
        ----------
        self
        
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        E_func : np.array 
            Array of values from exponential potential equation
            
        """
        potential = np.array([])
        E_func = np.zeros(self.N_max)
        const = -F / (R*self.temperature)
        #define reduction or oxidation first
        if self.E_start < self.E_switch: 
            self.direction = -1
        else:
            self.direction = 1     
        delta_theta = self.direction*self.potential_step
        Theta = (self.E_start - delta_theta)   
        switch = True
        for k in range(1, self.N_max + 1): 
            potential = np.append(potential, Theta)               
            #exponential potential function for multi-scan
            E_func[k-1] = (np.exp(const*self.direction*(self.E_switch 
                               + self.direction*abs((k*self.potential_step) 
                               + self.direction*(self.E_switch - self.E_start) 
                               - (2*abs(self.E_switch - self.E_start)*int((k
                               * self.potential_step) / (2*abs(self.E_switch 
                               - self.E_start))))) - self.E_not)))
                        
            if k%(self.N_max / (2*self.scans)) == 0: 
                switch = not switch
            
            if switch:
                Theta -= delta_theta
            else:
                Theta += delta_theta   
        return potential, E_func    
    ########################################################################## 
    def sum_function(self): 
        """Return weighting factors for semi-integration method.
        
        Parameters
        ----------
        self
        
        Returns
        -------
        W_n: np.array
            Array of weighting factor
        
        """
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
        
        Parameters
        ----------
        self
        
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        current: np.array
            Array of current values in full CV sweep
            
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
        transfer (E_q). 
        
        Parameters
        ----------
        self
        alpha : float
            Charge transfer coefficient
        k_not : float
            Standard rate constant (cm/s)
        
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        current: np.array
            Array of current values in full CV sweep
            
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
        
        Parameters
        ----------
        self
        alpha : float
            Charge transfer coefficient
        k_not : float
            Standard rate constant (cm/s)
        k_forward : float
            First order chemical rate constant (1/s)
        k_backward : float
            First order chemical rate constant (1/s)
            
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        current: np.array
            Array of current values in full CV sweep
            
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