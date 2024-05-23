"""
Class for cyclic voltammetry setup and simulation of one-electron processes.
"""

import numpy as np
import scipy.constants as spc

# Faraday constant (C/mol)
F = spc.value('Faraday constant')

# Molar gas constant (J/K/mol)
R = spc.R


class OneElectronCV:
    """
    Simulate cyclic voltammograms for a disk macro-electrode
    for one electron processes.

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (volts).
    switch_potential : float
        Switching potential of scan (volts).
    formal_potential : float
        Standard reduction potential (volts).
    scan_rate : float
        Potential sweep rate (volts / second).
    step_size : float
        Voltage increment size during CV scan (milli-volts).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    disk_radius : float
        Radius of disk macroelectrode (mm).
    temperature : float
        Temperature (kelvin).

    Notes
    -----
    
    Algorithm Reference:
    [1] Oldham, K. B.; Myland, J. C. Modelling cyclic voltammetry without 
    digital simulation, Electrochimica Acta, 56, 2011, 10612-10625.
    
    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            formal_potential: float,
            scan_rate: float,
            step_size: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            disk_radius: float,
            temperature: float = 298.0,
    ) -> None:
        self.start_potential = start_potential
        self.switch_potential = switch_potential
        self.formal_potential = formal_potential
        self.scan_rate = scan_rate
        self.step_size = step_size / 1000
        self.delta_t = self.step_size / self.scan_rate
        self.c_bulk = c_bulk
        self.diffusion_ratio = (diffusion_reactant / diffusion_product) ** 0.5
        self.velocity_constant = (diffusion_reactant / self.delta_t) ** 0.5
        self.area = np.pi * (disk_radius / 1000)**2
        self.temperature = temperature
        self.N_max = int((abs(switch_potential - start_potential) * 2) / self.step_size)
        self.scan_direction = -1 if self.start_potential < self.switch_potential else 1
        self.nernst_constant = -F / (R * self.temperature)
        self.cv_constant = -F * self.scan_direction * self.area * self.c_bulk * self.velocity_constant

    def voltage_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return potential steps for voltage profile and for exponential 
        Nernstian/Butler-Volmer function.

        Returns
        -------
        potential : np.ndarray
            Potential values in a full CV sweep.
        xi_function : np.ndarray
            Array of values from exponential potential equation.
            
        """
        potential = np.array([])
        #potential = np.zeros(self.N_max)
        xi_function = np.zeros(self.N_max)

        delta_theta = self.scan_direction * self.step_size
        theta = self.start_potential - delta_theta
        #potential[0] = theta ##
        for k in range(1, self.N_max + 1):
            potential = np.append(potential, theta)
            #potential[k] = theta
            # exponential potential function
            xi_function[k - 1] = np.exp(self.nernst_constant * self.scan_direction
                                        * (self.switch_potential + self.scan_direction * abs((k * self.step_size)
                                                                                             + self.scan_direction
                                                                                             * (self.switch_potential - self.start_potential))
                                           - self.formal_potential))
            if k < (int(self.N_max / 2)):
                theta -= delta_theta
            else:
                theta += delta_theta
        return potential, xi_function

    def semi_integrate_weights(self) -> np.ndarray:
        """
        Weighting factors for semi-integration method.

        Returns
        -------
        weights : np.ndarray
            Array of weighting factors.
        
        """

        weights = np.ones(self.N_max)
        for i in range(1, self.N_max):
            weights[i] = (2 * i - 1) * (weights[i - 1] / (2 * i))
        return weights

    def reversible(self) -> tuple[np.ndarray, np.ndarray]:  # reversible_mechanism
        """
        Current-potential profile for reversible (Nernstian)
        one electron transfer (E_r).
        
        Returns
        -------
        potential: np.ndarray
            Array of potential values in full CV sweep.
        current: np.ndarray
            Array of current values in full CV sweep.
            
        """
        weights = self.semi_integrate_weights()
        potential, xi_function = self.voltage_profile()
        current = np.zeros(self.N_max)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N - 1] = (self.cv_constant / (1 + (self.diffusion_ratio / xi_function[N - 1])))
            else:
                summ = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                current[N - 1] = ((self.cv_constant / (1 + (self.diffusion_ratio / xi_function[N - 1])))
                                  - summ)
        return potential, current

    def quasireversible(self, alpha: float, k_not: float) -> tuple[np.ndarray, np.ndarray]:  # e_mechanism
        """
        Current-potential profile for quasi-reversible one electron
        transfer (E_q). 
        
        Parameters
        ----------
        alpha : float
            Charge transfer coefficient (unit-less).
        k_not : float
            Standard electrochemical rate constant (cm/s).
        
        Returns
        -------
        potential: np.ndarray
            Array of potential values in full CV sweep.
        current: np.ndarray
            Array of current values in full CV sweep.
            
        """

        k_not = k_not / 100
        weights = self.semi_integrate_weights()
        potential, xi_function = self.voltage_profile()
        current = np.zeros(self.N_max)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N - 1] = (self.cv_constant / (1 + (self.diffusion_ratio / xi_function[N - 1])
                                                      + (self.velocity_constant / (np.power(xi_function[N - 1], alpha)
                                                                                   * k_not))))
            else:
                summ = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                current[N - 1] = ((self.cv_constant - (1 + (self.diffusion_ratio / xi_function[N - 1]))
                                   * summ) / (1 + (self.diffusion_ratio / xi_function[N - 1])
                                              + (self.velocity_constant / (np.power(xi_function[N - 1], alpha)
                                                                           * k_not))))
        return potential, current

    def quasireversible_chemical(
            self,
            alpha: float,
            k_not: float,
            k_forward: float,
            k_backward: float,
    ) -> tuple[np.ndarray, np.ndarray]:  # ec_mechanism
        """
        Current-potential profile for quasi-reversible, one electron
        transfer followed by homogeneous chemical kinetics (E_q C).
        
        Parameters
        ----------
        alpha : float
            Charge transfer coefficient (unit-less).
        k_not : float
            Standard electrochemical rate constant (cm/s).
        k_forward : float
            First order chemical rate constant (1/s).
        k_backward : float
            First order chemical rate constant (1/s).
            
        Returns
        -------
        potential: np.ndarray
            Array of potential values in full CV sweep.
        current: np.ndarray
            Array of current values in full CV sweep.
            
        """
        k_not = k_not / 100
        k_sum = k_forward + k_backward
        eqlbm_const = k_forward / k_backward
        weights = self.semi_integrate_weights()
        potential, xi_function = self.voltage_profile()
        current = np.zeros(self.N_max)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N - 1] = (self.cv_constant / (1 + (self.velocity_constant
                                                           / (np.power(xi_function[N - 1], alpha) * k_not))
                                                      + (self.diffusion_ratio / xi_function[N - 1])))
            else:
                summ = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                summ_exp = sum((weights[k] * current[N - k - 1] * np.exp((-k - 1) * k_sum))
                               for k in range(1, N))
                current[N - 1] = ((self.cv_constant - ((1 + (self.diffusion_ratio / ((1 + eqlbm_const)
                                                                                     * xi_function[N - 1]))) * summ) - (
                                               ((eqlbm_const * self.diffusion_ratio)
                                                / (xi_function[N - 1] * (1 + eqlbm_const))) * summ_exp)) / (1
                                                                                                       + (
                                                                                                                                                 self.velocity_constant / (
                                                                                                                       np.power(
                                                                                                                           xi_function[
                                                                                                                               N - 1],
                                                                                                                           alpha)
                                                                                                                       * k_not)) + (
                                                                                                               self.diffusion_ratio /
                                                                                                               xi_function[
                                                                                                                       N - 1])))
        return potential, current

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('testing')
    test1 = OneElectronCV(0.3, -0.3, 0, 0.1, 1, 5, 1.5e-6, 1.1e-6, 5, 298)
    potential, current = test1.reversible()#.quasireversible(0.5, 5e-2)
    plt.figure()
    plt.plot(potential, [x * 1000 for x in current], label='experiment')
    plt.show()
