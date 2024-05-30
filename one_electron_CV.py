"""
Class for cyclic voltammetry setup and simulation of one-electron processes.
"""

import numpy as np
import scipy.constants as spc

# Faraday constant (C/mol)
F = spc.value('Faraday constant')

# Molar gas constant (J/K/mol)
R = spc.R

# TO-DO
# do we need numpy arrays or just lists?


class OneElectronCV:
    """
    Simulate cyclic voltammograms for one electron processes
    on a disk macro-electrode.

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    formal_potential : float
        Formal reduction potential (V vs. reference).
    scan_rate : float
        Potential sweep rate (V/s).
    step_size : float
        Voltage increment during CV scan (mV).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (25C).

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
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        self.start_potential = start_potential
        self.switch_potential = switch_potential
        self.formal_potential = formal_potential
        self.scan_rate = scan_rate
        self.step_size = step_size / 1000  # mV to V
        self.delta_t = self.step_size / self.scan_rate
        self.c_bulk = c_bulk
        self.diffusion_ratio = (diffusion_reactant / diffusion_product) ** 0.5
        self.velocity_constant = ((diffusion_reactant / 1e4) / self.delta_t) ** 0.5  # cm^2/s to m^2/s
        self.area = np.pi * (disk_radius / 1000)**2  # mm to m, then m^2
        self.temperature = temperature
        self.n_max = int((abs(switch_potential - start_potential) * 2) / self.step_size)
        self.scan_direction = -1 if self.start_potential < self.switch_potential else 1
        self.nernst_constant = -F / (R * self.temperature)
        self.cv_constant = -F * self.scan_direction * self.area * self.c_bulk * self.velocity_constant
        self.delta_theta = self.scan_direction * self.step_size

    def _voltage_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return potential steps for voltage profile and for exponential Nernstian/Butler-Volmer function.
        This is equation (6:1) from [1].

        Returns
        -------
        potential : np.ndarray
            Potential values in a full CV sweep.
        xi_function : np.ndarray
            Values from exponential potential equation.
            
        """

        theta = self.start_potential - self.delta_theta

        potential = np.array([])
        xi_function = np.zeros(self.n_max)
        for k in range(1, self.n_max + 1):
            potential = np.append(potential, theta)
            #potential[k] = theta
            # exponential potential function
            xi_function[k - 1] = np.exp(self.nernst_constant * self.scan_direction
                                        * (self.switch_potential
                                           + self.scan_direction * abs((k * self.step_size) + self.scan_direction
                                                                       * (self.switch_potential - self.start_potential))
                                           - self.formal_potential))
            if k < (int(self.n_max / 2)):
                theta -= self.delta_theta
            else:
                theta += self.delta_theta

        # testing lists only (limited numpy)
        #
        # potential = [0.0] * self.n_max
        # potential[0] = theta

        return potential, xi_function

    def _semi_integrate_weights(self) -> np.ndarray:
        """
        Weighting factors for semi-integration method.
        This is equation (5:5) from [1].

        Returns
        -------
        weights : np.ndarray
            Array of weighting factors.
        
        """

        weights = np.ones(self.n_max)
        for n in range(1, self.n_max):
            weights[n] = (2 * n - 1) * (weights[n - 1] / (2 * n))
        return weights

    def reversible(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Current-potential profile for reversible (Nernstian) one electron transfer (E_r) mechanism.
        This is equation (7:7) from [1].
        
        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.
            
        """
        weights = self._semi_integrate_weights()
        potential, xi_function = self._voltage_profile()
        current = np.zeros(self.n_max)
        for n in range(1, self.n_max + 1):  # TO-DO refactor
            if n == 1:
                current[n - 1] = self.cv_constant / (1 + (self.diffusion_ratio / xi_function[n - 1]))
            else:
                sum_weights = sum(weights[k] * current[n - k - 1] for k in range(1, n))
                current[n - 1] = ((self.cv_constant / (1 + (self.diffusion_ratio / xi_function[n - 1])))
                                  - sum_weights)
        return potential, current

    def quasireversible(self, alpha: float, k_0: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Current-potential profile for quasi-reversible one electron transfer (E_q) mechanism.
        This is equation (8:3) from [1].
        
        Parameters
        ----------
        alpha : float
            Charge transfer coefficient (no units).
        k_0 : float
            Standard electrochemical rate constant (cm/s).
        
        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.
            
        """

        k_0 = k_0 / 100  # cm/s to m/s
        weights = self._semi_integrate_weights()
        potential, xi_function = self._voltage_profile()
        current = np.zeros(self.n_max)
        for N in range(1, self.n_max + 1):  # TO-DO refactor
            if N == 1:
                current[N - 1] = self.cv_constant / (1 + (self.diffusion_ratio / xi_function[N - 1])
                                                     + (self.velocity_constant / (np.power(xi_function[N - 1], alpha)
                                                                                  * k_0)))
            else:
                sum_weights = sum(weights[k] * current[N - k - 1] for k in range(1, N))
                current[N - 1] = ((self.cv_constant - (1 + (self.diffusion_ratio / xi_function[N - 1]))
                                   * sum_weights) / (1 + (self.diffusion_ratio / xi_function[N - 1])
                                              + (self.velocity_constant / (np.power(xi_function[N - 1], alpha)
                                                                           * k_0))))
        return potential, current

    def quasireversible_chemical(
            self,
            alpha: float,
            k_0: float,
            k_forward: float,
            k_backward: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Current-potential profile for quasi-reversible one electron transfer followed by homogeneous chemical
        kinetics (E_q C) mechanism.
        This is equation (10:4) from [1].
        
        Parameters
        ----------
        alpha : float
            Charge transfer coefficient (no units).
        k_0 : float
            Standard electrochemical rate constant (cm/s).
        k_forward : float
            First order chemical rate constant (1/s).
        k_backward : float
            First order chemical rate constant (1/s).
            
        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.
            
        """

        k_0 = k_0 / 100  # cm/s to m/s
        k_sum = k_forward + k_backward
        k_const = k_forward / k_backward
        weights = self._semi_integrate_weights()
        potential, xi_function = self._voltage_profile()
        current = np.zeros(self.n_max)
        for n in range(1, self.n_max + 1):  # TO-DO refactor
            if n == 1:
                current[n - 1] = (self.cv_constant / (1 + (self.velocity_constant
                                                           / (np.power(xi_function[n - 1], alpha) * k_0))
                                                      + (self.diffusion_ratio / xi_function[n - 1])))
            else:
                sum_weights = sum(weights[k] * current[n - k - 1] for k in range(1, n))
                sum_exp_weights = sum((weights[k] * current[n - k - 1] * np.exp((-k - 1) * k_sum)) for k in range(1, n))
                current[n - 1] = (self.cv_constant
                                  - ((1 + (self.diffusion_ratio / ((1 + k_const) * xi_function[n - 1]))) * sum_weights)
                                  - (((k_const * self.diffusion_ratio) / (xi_function[n - 1] * (1 + k_const)))
                                     * sum_exp_weights)
                                  ) / (1 + (self.velocity_constant / (np.power(xi_function[n - 1], alpha) * k_0))
                                       + (self.diffusion_ratio / xi_function[n - 1]))
        return potential, current


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('testing')
    test1 = OneElectronCV(-0.3, 0.3, 0, 1.0, 0.5, 1, 1e-5, 1e-5, 5.642, 298)
    potential, current = test1.reversible()#.quasireversible(0.5, 5e-2)
    peak_idx = np.argmax(current)
    print(f"peak potential: {potential[peak_idx]:.4f} V vs SHE , peak current {current[peak_idx]*1000:.6f} mA")
    plt.figure()
    plt.plot(potential, current, label='experiment')
    plt.show()
