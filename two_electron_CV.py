
import numpy as np
import scipy.constants as spc

F = spc.physical_constants['Faraday constant'][0]
R = spc.R

class TwoElectronCV:
    """
    This is a class to simulate cyclic voltammograms for a disk macroelectrode
    for two electron processes.
    
    Algorithm Reference:
    [1] Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, Electrochimica Acta, 56, 2011, 10612-10625. 
    
    """  
     
    def __init__(self, E_start, E_switch, E_not1, E_not2, scanrate, mV_step, 
                 c_bulk, diff_r, diff_i, diff_p, disk_radius, temperature): 
        """
        Parameters to define the CV setup that are shared by all 
        reaction mechanism functions available for simulation.
        
        Parameters
        ----------
        E_start : float
            Starting potential of scan (volts)
        E_switch : float
            Switching potential of scan (volts)
        E_not1 : float
            Standard reduction potential of first electron (volts)
        E_not2 : float
            Standard reduction potential of second electron (volts)
        scanrate : float
            Potential sweep rate (volts / second)
        mV_step : float
            Potential increment of scan (millivolts)
        c_bulk  : float
            Bulk concentration of redox species (mM or mol/m^3) 
        diff_r : float
            Diffusion coefficient of reactant (cm^2/s) 
        diff_i : float
            Diffusion coefficient of intermediate (cm^2/s) 
        diff_p : float
            Diffusion coefficient of product (cm^2/s)
        disk_radius : float
            Radius of disk macroelectrode (mm)
        temperature : float
            Temperature (kelvin)
            
        """
        self.E_start = E_start   
        self.E_switch = E_switch   
        self.E_not1 = E_not1   
        self.E_not2 = E_not2   
        self.scanrate = scanrate    
        self.potential_step = (mV_step / 1000)  
        self.delta_t = (self.potential_step / self.scanrate) 
        self.c_bulk = c_bulk    
        self.diff_r = (diff_r / 1e4) 
        self.diff_i = (diff_i / 1e4)  
        self.diff_p = (diff_p / 1e4)  
        self.area = np.pi*((disk_radius / 1000)**2)         
        self.temperature = temperature  
        self.N_max = int(np.abs(E_switch - E_start)*2 / self.potential_step) 
    ########################################################################## 
    def voltage_profile(self):
        """
        Return potential steps for voltage profile and for exponential 
        Nernstian/Butler-Volmer function.
        
         Parameters
        ----------
        self
        
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        E_func1 : np.array 
            Array of values from first electron exponential potential equation
        E_func2 : np.array 
            Array of values from second electron exponential potential equation
            
        """
        potential = np.array([])
        E_func1 = np.zeros(self.N_max)
        E_func2 = np.zeros(self.N_max)
        const = -F / (R*self.temperature)
        #define reduction or oxidation first
        if self.E_start < self.E_switch: 
            self.direction = -1.
        else:
            self.direction = 1.
        delta_theta = self.direction*self.potential_step   
        Theta = (self.E_start - delta_theta)      
        for k in range(1, self.N_max + 1): 
            potential = np.append(potential, Theta)               
            #exponential potential function
            E_func1[k-1] = np.exp(const*self.direction*(self.E_switch 
                                  + self.direction*abs((k*self.potential_step) 
                                  + self.direction*(self.E_switch 
                                  - self.E_start)) - self.E_not1))
            
            E_func2[k-1] = np.exp(const*self.direction*(self.E_switch 
                                  + self.direction*abs((k*self.potential_step) 
                                  + self.direction*(self.E_switch 
                                  - self.E_start)) - self.E_not2))            
            if k < (int(self.N_max/2)):
                Theta -= delta_theta
            else:
                Theta += delta_theta     
        return potential, E_func1, E_func2
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
    def quasireversible(self, alpha1, alpha2, k_not1, k_not2): 
        """
        Return current-potential profile for reversible/quasi-reversible, 
        two successive one-electron transfers (E_q E_q). 
        
        Parameters
        ----------
        self
        alpha1 : float
            Charge transfer coefficient, first electron
        alpha2 : float
            Charge transfer coefficient, second electron
        k_not1 : float
            Standard rate constant (cm/s), first electron
        k_not2 : float
            Standard rate constant (cm/s), second electron
        
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        current: np.array
            Array of current values in full CV sweep
            
        """
        k_not1 = k_not1 / 100
        k_not2 = k_not2 / 100
        W_n = self.sum_function()
        potential, E_func1, E_func2 = self.voltage_profile()
        D_const = np.sqrt(self.diff_i / self.delta_t)
        Z_func = ((D_const / k_not1)*np.power(E_func1, (1 - alpha1)))
        Y_func = (1 + (E_func1*np.sqrt(self.diff_i / self.diff_r)))
        W_func = ((D_const / k_not2) / np.power(E_func2, alpha2))
        V_func = (1 + (np.sqrt(self.diff_i / self.diff_p) / E_func2))
        current1 = np.zeros(self.N_max)
        current2 = np.zeros(self.N_max)
        constant = (-F*self.direction*self.area*self.c_bulk*D_const)
        for N in range(1, self.N_max +1):
            if N == 1:
                current1[N-1] = (((W_func[N-1] + V_func[N-1])*E_func1[N-1]
                                 * constant) / ((Z_func[N-1] + Y_func[N-1])
                                 * (W_func[N-1] + V_func[N-1]) - 1)) 
                     
                current2[N-1] = ((E_func1[N-1]*constant) / ((Z_func[N-1] 
                                 + Y_func[N-1])*(W_func[N-1] + V_func[N-1])-1))    
            else:
                summ1 = sum(W_n[k]*current1[N-k-1] for k in range(1,N))
                summ2 = sum(W_n[k]*current2[N-k-1] for k in range(1,N))
                current1[N-1] = ((((W_func[N-1] + V_func[N-1])*E_func1[N-1]
                                 * constant) - ((Y_func[N-1]*(W_func[N-1] 
                                 + V_func[N-1]) - 1)*summ1) + W_func[N-1]
                                 * summ2) / ((Z_func[N-1] + Y_func[N-1])
                                 * (W_func[N-1] + V_func[N-1]) -1))
                    
                current2[N-1] = (((E_func1[N-1]*constant) + Z_func[N-1]*summ1 
                                 - (V_func[N-1]*(Z_func[N-1] + Y_func[N-1]) 
                                 - 1)*summ2) / ((Z_func[N-1] + Y_func[N-1])
                                 * (W_func[N-1] + V_func[N-1]) -1))  
        current = [current1[i] + current2[i] for i in range(len(current1))]
        return potential, current
    ##########################################################################
    ##########################################################################
    def square_scheme(self, alpha1, alpha2, k_not1, k_not2, k_forward1, 
                      k_backward1, k_forward2, k_backward2): 
        """
        Return current-potential profile for two quasi-reversible, 
        one-electron transfers of homogeneously interconverting reactants
        (square scheme). 
        
        Parameters
        ----------
        self
        alpha1 : float
            Charge transfer coefficient, first electron
        alpha2 : float
            Charge transfer coefficient, second electron
        k_not1 : float
            Standard rate constant (cm/s), first electron
        k_not2 : float
            Standard rate constant (cm/s), second electron
        k_forward1 : float
            First order chemical rate constant (1/s), first reactant
        k_backward1 : float
            First order chemical rate constant (1/s), first reactant
        k_forward2 : float
            First order chemical rate constant (1/s), second reactant
        k_backward2 : float
            First order chemical rate constant (1/s), second reactant
        
        Returns
        -------
        potential: np.array
            Array of potential values in full CV sweep
        current: np.array
            Array of current values in full CV sweep
            
        """
        k_not1 = k_not1 / 100
        k_not2 = k_not2 / 100
        W_n = self.sum_function()
        potential, E_func1, E_func2 = self.voltage_profile() 
        k_sum1 = k_forward1 + k_backward1
        big_K1 = k_forward1 / k_backward1
        k_sum2 = k_forward2 + k_backward2
        big_K2 = k_forward2 / k_backward2
        D_const = np.sqrt(self.diff_r / self.delta_t)
        D_ratio = np.sqrt(self.diff_r / self.diff_p)
        current1 = np.zeros(self.N_max)
        current2 = np.zeros(self.N_max)
        constant = (-F*self.direction*self.area*self.c_bulk*D_const)
        for N in range(1, self.N_max +1):
            if N == 1:
                current1[N-1] = ((constant / (1 + big_K1)) / ((D_const 
                                 / (np.power(E_func1[N-1], alpha1)*k_not1)) 
                                 + 1 + (D_ratio / E_func1[N-1])))
                
                current2[N-1] = (((constant*big_K1) / (1 + big_K1)) / ((D_const 
                                 / (np.power(E_func2[N-1], alpha2)*k_not2)) 
                                 + 1 + (D_ratio / E_func2[N-1])))    
            else:
                summ1 = sum(W_n[k]*current1[N-k-1] for k in range(1,N))
                summ2 = sum(W_n[k]*current2[N-k-1] for k in range(1,N))
                summ1_exp1 = sum((W_n[k]*current1[N-k-1]*np.exp((-k-1)*k_sum1)) 
                               for k in range(1,N))
                summ1_exp2 = sum((W_n[k]*current1[N-k-1]*np.exp((-k-1)*k_sum2)) 
                               for k in range(1,N))
                summ2_exp1 = sum((W_n[k]*current2[N-k-1]*np.exp((-k-1)*k_sum1)) 
                               for k in range(1,N))
                summ2_exp2 = sum((W_n[k]*current2[N-k-1]*np.exp((-k-1)*k_sum2)) 
                               for k in range(1,N))
                
                current1[N-1] = (((constant / (1 + big_K1)) - ((summ1 + summ2 
                                 + (big_K1*summ1_exp1) - summ2_exp1) / (1 
                                 + big_K1)) - ((summ1 + summ2 + (big_K2
                                 * summ1_exp2) - summ2_exp2) / ((E_func1[N-1]
                                 * (1 + big_K2)) / D_ratio))) / ((D_const 
                                 / (np.power(E_func1[N-1], alpha1)*k_not1)) 
                                 + 1 + (D_ratio / E_func1[N-1])))
                
                current2[N-1] = ((((constant*big_K1) / (1 + big_K1)) 
                                 - (((big_K1*(summ1 + summ2)) - (big_K1
                                 * summ1_exp1) + summ2_exp1) / (1 + big_K1)) 
                                 - (((big_K2*(summ1 + summ2)) - (big_K2
                                 * summ1_exp2) + summ2_exp2) / ((E_func2[N-1]
                                 * (1 + big_K2)) / D_ratio))) / ((D_const 
                                 / (np.power(E_func2[N-1], alpha2)*k_not2)) 
                                 + 1 + (D_ratio / E_func2[N-1])))
                                 
        current = [current1[i] + current2[i] for i in range(len(current1))]                       
        return potential, current
##############################################################################                                          
if __name__ == '__main__':
    print('testing')        
