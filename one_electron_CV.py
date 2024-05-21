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
    E_start : float
        Starting potential of scan (volts).
    E_switch : float
        Switching potential of scan (volts).
    E_not : float
        Standard reduction potential (volts).
    scan_rate : float
        Potential sweep rate (volts / second).
    mV_step : float
        Potential increment of scan (millivolts).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diff_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diff_product : float
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
            E_start: float,
            E_switch: float,
            E_not: float,
            scan_rate: float,
            mV_step: float,
            c_bulk: float,
            diff_reactant: float,
            diff_product: float,
            disk_radius: float,
            temperature: float = 298.0,
    ) -> None:
        self.E_start = E_start
        self.E_switch = E_switch
        self.E_not = E_not
        self.scan_rate = scan_rate
        self.potential_step = (mV_step / 1000)
        self.delta_t = (self.potential_step / self.scan_rate)
        self.c_bulk = c_bulk
        self.D_ratio = np.sqrt(diff_reactant / diff_product)
        self.D_const = np.sqrt(diff_reactant / self.delta_t)
        self.area = np.pi * ((disk_radius / 1000) ** 2)
        self.temperature = temperature
        self.N_max = int((np.abs(E_switch - E_start) * 2) / self.potential_step)
        self.scan_direction = -1 if self.E_start < self.E_switch else 1
        self.nernst_constant = -F / (R * self.temperature)
        self.cv_constant = -F * self.scan_direction * self.area * self.c_bulk * self.D_const

    def voltage_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return potential steps for voltage profile and for exponential 
        Nernstian/Butler-Volmer function.

        Returns
        -------
        potential : np.ndarray
            Potential values in a full CV sweep.
        E_func : np.ndarray
            Array of values from exponential potential equation.
            
        """
        potential = np.array([])
        E_func = np.zeros(self.N_max)

        delta_theta = self.scan_direction * self.potential_step
        Theta = self.E_start - delta_theta
        for k in range(1, self.N_max + 1):
            potential = np.append(potential, Theta)
            # exponential potential function
            E_func[k - 1] = np.exp(self.nernst_constant * self.scan_direction
                                   * (self.E_switch + self.scan_direction * abs((k * self.potential_step)
                                                                                + self.scan_direction
                                                                                * (self.E_switch - self.E_start))
                                      - self.E_not))
            if k < (int(self.N_max / 2)):
                Theta -= delta_theta
            else:
                Theta += delta_theta
        return potential, E_func

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
        potential, E_func = self.voltage_profile()
        current = np.zeros(self.N_max)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N - 1] = (self.cv_constant / (1 + (self.D_ratio / E_func[N - 1])))
            else:
                summ = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                current[N - 1] = ((self.cv_constant / (1 + (self.D_ratio / E_func[N - 1])))
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
        potential, E_func = self.voltage_profile()
        current = np.zeros(self.N_max)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N - 1] = (self.cv_constant / (1 + (self.D_ratio / E_func[N - 1])
                                                      + (self.D_const / (np.power(E_func[N - 1], alpha)
                                                                         * k_not))))
            else:
                summ = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                current[N - 1] = ((self.cv_constant - (1 + (self.D_ratio / E_func[N - 1]))
                                   * summ) / (1 + (self.D_ratio / E_func[N - 1])
                                              + (self.D_const / (np.power(E_func[N - 1], alpha)
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
        potential, E_func = self.voltage_profile()
        current = np.zeros(self.N_max)
        for N in range(1, self.N_max + 1):
            if N == 1:
                current[N - 1] = (self.cv_constant / (1 + (self.D_const
                                                           / (np.power(E_func[N - 1], alpha) * k_not))
                                                      + (self.D_ratio / E_func[N - 1])))
            else:
                summ = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                summ_exp = sum((weights[k] * current[N - k - 1] * np.exp((-k - 1) * k_sum))
                               for k in range(1, N))
                current[N - 1] = ((self.cv_constant - ((1 + (self.D_ratio / ((1 + eqlbm_const)
                                                                             * E_func[N - 1]))) * summ) - (
                                               ((eqlbm_const * self.D_ratio)
                                                / (E_func[N - 1] * (1 + eqlbm_const))) * summ_exp)) / (1
                                                                                                       + (
                                                                                                                   self.D_const / (
                                                                                                                       np.power(
                                                                                                                           E_func[
                                                                                                                               N - 1],
                                                                                                                           alpha)
                                                                                                                       * k_not)) + (
                                                                                                                   self.D_ratio /
                                                                                                                   E_func[
                                                                                                                       N - 1])))
        return potential, current
