"""
Class for semi-integration simulation of cyclic voltammetry (CV)
of one- and two-electron processes.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as spc

# Faraday constant (C/mol)
F = spc.value('Faraday constant')

# Molar gas constant (J/K/mol)
R = spc.R


class CyclicVoltammetryProtocol(ABC):
    """
    Abstract class representing a simulated cyclic voltammogram on a disk macro-electrode.

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs reference).
    switch_potential : float
        Switching potential of scan (V vs reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (24.85C).

    Notes
    -----

    Algorithm Reference:
    [1] Oldham, K. B.; Myland, J. C. "Modelling cyclic voltammetry without
    digital simulation." Electrochimica Acta, 56, 2011, 10612-10625.

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        self.start_potential = start_potential
        self.switch_potential = switch_potential
        self.reduction_potential = reduction_potential
        self.step_size = step_size / 1000  # mV to V
        self.delta_t = self.step_size / scan_rate
        self.diffusion_reactant = diffusion_reactant / 1e4  # cm^2/s to m^2/s
        self.diffusion_product = diffusion_product / 1e4  # cm^2/s to m^2/s
        self.diffusion_ratio = (self.diffusion_reactant / self.diffusion_product) ** 0.5
        self.velocity_constant = (self.diffusion_reactant / self.delta_t) ** 0.5
        self.electrode_area = np.pi * (disk_radius / 1000) ** 2  # mm to m, then m^2
        self.n_max = int((abs(switch_potential - start_potential) * 2) / self.step_size)
        self.scan_direction = -1 if self.start_potential < self.switch_potential else 1
        self.nernst_constant = -F / (R * temperature)
        self.cv_constant = -F * self.scan_direction * self.electrode_area * c_bulk * self.velocity_constant
        self.delta_theta = self.scan_direction * self.step_size

    def voltage_profile_setup(self, second_reduction_potential: float | None = None) -> tuple[list[float], list]:
        """
        Return potential steps for voltage profile and for exponential Nernstian/Butler-Volmer function.
        This is equation (6:1) from [1].

        Parameters
        ----------
        second_reduction_potential : float | None
            Reduction potential of the second electron transfer process (V vs reference).
            Only used if a 2 electron process class is called.
            Default is None.

        Returns
        -------
        potential : list
            Potential values in a full CV sweep.
        xi_function : np.ndarray
            Values from exponential potential equation.

        """

        electron_transfers = [self.reduction_potential]
        if second_reduction_potential is not None:
            electron_transfers.append(second_reduction_potential)

        theta = self.start_potential - self.delta_theta
        all_xi_functions = []
        potential = []
        for step in range(1, self.n_max + 1):
            potential.append(theta)
            if step < int(self.n_max / 2):
                theta -= self.delta_theta
            else:
                theta += self.delta_theta

        for potential_value in electron_transfers:
            xi_function = np.zeros(self.n_max)
            for k in range(1, self.n_max + 1):
                xi_function[k-1] = np.exp(self.nernst_constant * self.scan_direction
                                          * (self.switch_potential
                                             + self.scan_direction
                                             * abs((k * self.step_size) + self.scan_direction
                                                   * (self.switch_potential - self.start_potential)) - potential_value))

            all_xi_functions.append(xi_function)

        # testing lists only (limited numpy)
        #
        # potential = [0.0] * self.n_max
        # potential[0] = theta

        return potential, all_xi_functions

    def _semi_integrate_weights(self) -> list:
        """
        Weighting factors for semi-integration method.
        This is equation (5:5) from [1].

        Returns
        -------
        weights : np.ndarray
            Array of weighting factors.

        """

        weights = [1.0] * self.n_max
        for n in range(1, self.n_max):
            weights[n] = (2 * n - 1) * (weights[n-1] / (2 * n))
        return weights

    @abstractmethod
    def mechanism(self) -> tuple[list, np.ndarray]:
        """Simulates current-potential profile for desired mechanism"""
        raise NotImplementedError


class E_rev(CyclicVoltammetryProtocol):
    """
    Provides a current-potential profile for a reversible (Nernstian) one electron transfer mechanism.
    This is equation (7:7) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs reference).
    switch_potential : float
        Switching potential of scan (V vs reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (25C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        super().__init__(
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            step_size,
            disk_radius,
            temperature)

    def mechanism(self) -> tuple[list, np.ndarray]:
        """
        Simulates the CV for a reversible one electron transfer mechanism.

        Returns
        -------
        potential : list
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """
        weights = self._semi_integrate_weights()
        potential, xi_function = self.voltage_profile_setup()
        # better way to handle this?
        xi_function = xi_function[0]

        current = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum_weights = sum(weights[k] * current[n-k] for k in range(n))
            current[n] = (self.cv_constant / (1 + (self.diffusion_ratio / xi_function[n]))) - sum_weights
        return potential, current


class E_quasirev(CyclicVoltammetryProtocol):
    """
    Provides a current-potential profile for a quasi-reversible one electron transfer mechanism.
    This is equation (8:3) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs reference).
    switch_potential : float
        Switching potential of scan (V vs reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    alpha : float
        Charge transfer coefficient (no units).
    k_0 : float
        Standard electrochemical rate constant (cm/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (25C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            alpha: float,
            k_0: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        super().__init__(
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            step_size,
            disk_radius,
            temperature)
        self.alpha = alpha
        self.k_0 = k_0 / 100  # cm/s to m/s

    def mechanism(self) -> tuple[list, np.ndarray]:
        """
        Simulates the CV for a quasi-reversible one electron transfer mechanism.

        Returns
        -------
        potential : list
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """

        weights = self._semi_integrate_weights()
        potential, xi_function = self.voltage_profile_setup()
        xi_function = xi_function[0]

        current = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum_weights = sum(weights[k] * current[n-k] for k in range(n))
            current[n] = ((self.cv_constant - (1 + (self.diffusion_ratio / xi_function[n])) * sum_weights)
                          / (1 + (self.diffusion_ratio / xi_function[n])
                             + (self.velocity_constant / (np.power(xi_function[n], self.alpha) * self.k_0))))
        return potential, current


class E_quasirevC(CyclicVoltammetryProtocol):
    """
    Provides a current-potential profile for a quasi-reversible one electron transfer, followed by a reversible first
    order homogeneous chemical transformation mechanism.
    This is equation (10:4) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs reference).
    switch_potential : float
        Switching potential of scan (V vs reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    alpha : float
        Charge transfer coefficient (no units).
    k_0 : float
        Standard electrochemical rate constant (cm/s).
    k_forward : float
        First order forward chemical rate constant (1/s).
    k_backward : float
        First order backward chemical rate constant (1/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (25C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            alpha: float,
            k_0: float,
            k_forward: float,
            k_backward: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        super().__init__(
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            step_size,
            disk_radius,
            temperature)
        self.alpha = alpha
        self.k_0 = k_0 / 100  # cm/s to m/s
        self.k_forward = k_forward
        self.k_backward = k_backward

    def mechanism(self) -> tuple[list, np.ndarray]:
        """
        Simulates the CV for a quasi-reversible one electron transfer followed by a reversible first order homogeneous
        chemical transformation mechanism.

        Returns
        -------
        potential : list
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """

        k_sum = self.k_forward + self.k_backward
        k_const = self.k_forward / self.k_backward
        weights = self._semi_integrate_weights()
        potential, xi_function = self.voltage_profile_setup()
        xi_function = xi_function[0]

        current = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum_weights = sum(weights[k] * current[n-k] for k in range(n))
            sum_exp_weights = sum((weights[k] * current[n-k] * np.exp((-k-1) * k_sum)) for k in range(n))

            current[n] = ((self.cv_constant
                          - ((1 + (self.diffusion_ratio / ((1 + k_const) * xi_function[n]))) * sum_weights)
                          - (((k_const * self.diffusion_ratio) / (xi_function[n] * (1 + k_const))) * sum_exp_weights))
                          / (1 + (self.velocity_constant / (np.power(xi_function[n], self.alpha) * self.k_0))
                             + (self.diffusion_ratio / xi_function[n])))
        return potential, current


class EE(CyclicVoltammetryProtocol):
    """
    Provides a current-potential profile for two successive one-electron quasi-reversible transfers.
    This is equation (12:19) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs reference).
    switch_potential : float
        Switching potential of scan (V vs reference).
    reduction_potential : float
        Reduction potential of the first one-electron transfer process (V vs reference).
    second_reduction_potential : float
        Reduction potential of the second one-electron transfer process (V vs reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_intermediate : float
        Diffusion coefficient of intermediate (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    alpha : float
        Charge transfer coefficient (no units).
    alpha_second_e : float
        Charge transfer coefficient of second redox process (no units).
    k_0 : float
        Standard electrochemical rate constant (cm/s).
    k_0_second_e : float
        Standard electrochemical rate constant of second redox process (cm/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (25C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            second_reduction_potential: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_intermediate: float,
            diffusion_product: float,
            alpha: float,
            alpha_second_e: float,
            k_0: float,
            k_0_second_e: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        super().__init__(
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            step_size,
            disk_radius,
            temperature)
        self.second_reduction_potential = second_reduction_potential
        self.diffusion_intermediate = diffusion_intermediate / 1e4  # cm^2/s to m^2/s
        self.alpha = alpha
        self.alpha_second_e = alpha_second_e
        self.k_0 = k_0 / 100  # cm/s to m/s
        self.k_0_second_e = k_0_second_e / 100  # cm/s to m/s

        # if self.second_reduction_potential is None:
        #    raise ValueError("'second_reduction_potential' must also be declared when using a 2 electron mechanism")

    def mechanism(self) -> tuple[list[float], list]:
        """ TO-DO: w and v functions can combine?
        Simulates the CV for two successive one-electron quasi-reversible transfer (EE) mechanism.

        Returns
        -------
        potential : list
            Potential values in full CV sweep.
        current : list
            Current values in full CV sweep.

        """

        weights = self._semi_integrate_weights()
        potential, (xi_function1, xi_function2) = self.voltage_profile_setup(self.second_reduction_potential)

        intermediate_const = (self.diffusion_intermediate / self.delta_t) ** 0.5

        # Equation (12:15) from [1]
        z_function = (intermediate_const / self.k_0) * np.power(xi_function1, (1 - self.alpha))
        # Equation (12:16) from [1]
        y_function = 1 + (xi_function1 * np.sqrt(self.diffusion_intermediate / self.diffusion_reactant))
        # Equation (12:17) from [1]
        w_function = (intermediate_const / self.k_0_second_e) / np.power(xi_function2, self.alpha_second_e)
        # Equation (12:18) from [1]
        v_function = 1 + (np.sqrt(self.diffusion_intermediate / self.diffusion_product) / xi_function2)

        w_v_sum = w_function + v_function
        z_y_sum = z_function + y_function

        i_constant = (self.cv_constant / self.velocity_constant) * intermediate_const

        current1 = np.zeros(self.n_max)  # make lists?
        current2 = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum1 = sum(weights[k] * current1[n-k] for k in range(n))
            sum2 = sum(weights[k] * current2[n-k] for k in range(n))

            current1[n] = ((w_v_sum[n] * xi_function1[n] * i_constant
                            - ((y_function[n] * w_v_sum[n] - 1) * sum1) + w_function[n] * sum2)
                           / (z_y_sum[n] * w_v_sum[n] - 1))

            current2[n] = ((xi_function1[n] * i_constant + z_function[n] * sum1
                            - (v_function[n] * z_y_sum[n] - 1) * sum2)
                           / ((z_function[n] + y_function[n]) * w_v_sum[n] - 1))

        current = current1 + current2
        return potential, current


class SquareScheme(CyclicVoltammetryProtocol):
    """
    Provides a current-potential profile for two quasi-reversible, one-electron transfers of homogeneously
    interconverting reactants (square scheme).
    This is equations (14:14 and 14:15) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs reference).
    switch_potential : float
        Switching potential of scan (V vs reference).
    reduction_potential : float
        Reduction potential of the first one-electron transfer process (V vs reference).
    second_reduction_potential : float
        Reduction potential of the second one-electron transfer process (V vs reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    alpha : float
        Charge transfer coefficient (no units).
    alpha_second_e : float
        Charge transfer coefficient of second redox process (no units).
    k_0 : float
        Standard electrochemical rate constant (cm/s).
    k_0_second_e : float
        Standard electrochemical rate constant of second redox process (cm/s).
    k_forward_first : float
        First order forward chemical rate constant for first redox species (1/s).
    k_backward_first : float
        First order backward chemical rate constant for first redox species (1/s).
    k_forward_second : float
        First order forward chemical rate constant for second redox species (1/s).
    k_backward_second : float
        First order backward chemical rate constant for second redox species (1/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298 K (25C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            second_reduction_potential: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            alpha: float,
            alpha_second_e: float,
            k_0: float,
            k_0_second_e: float,
            k_forward_first: float,
            k_backward_first: float,
            k_forward_second: float,
            k_backward_second: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
    ) -> None:
        super().__init__(
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            step_size,
            disk_radius,
            temperature)
        self.second_reduction_potential = second_reduction_potential
        self.alpha = alpha
        self.alpha_second_e = alpha_second_e
        self.k_0 = k_0 / 100  # cm/s to m/s
        self.k_0_second_e = k_0_second_e / 100  # cm/s to m/s
        self.k_forward_first = k_forward_first
        self.k_backward_first = k_backward_first
        self.k_forward_second = k_forward_second
        self.k_backward_second = k_backward_second

    def mechanism(self) -> tuple[list[float], np.ndarray]:
        """
        Simulates the CV for two quasi-reversible, one-electron transfers of homogeneously interconverting
        reactants (square scheme).

        Returns
        -------
        potential : list
            Potential values in full CV sweep.
        current : list
            Current values in full CV sweep.
        """

        weights = self._semi_integrate_weights()
        potential, (xi_function1, xi_function2) = self.voltage_profile_setup(self.second_reduction_potential)

        k_sum1 = self.k_forward_first + self.k_backward_first
        big_k1 = self.k_forward_first / self.k_backward_first
        k_sum2 = self.k_forward_second + self.k_backward_second
        big_k2 = self.k_forward_second / self.k_backward_second

        current1 = np.zeros(self.n_max)
        current2 = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum1 = sum(weights[k] * current1[n-k] for k in range(n))
            sum2 = sum(weights[k] * current2[n-k] for k in range(n))
            sum_sums = sum1 + sum2

            sum1_exp1 = sum((weights[k] * current1[n-k] * np.exp((-k-1) * k_sum1)) for k in range(n))
            sum1_exp2 = sum((weights[k] * current1[n-k] * np.exp((-k-1) * k_sum2)) for k in range(n))
            sum2_exp1 = sum((weights[k] * current2[n-k] * np.exp((-k-1) * k_sum1)) for k in range(n))
            sum2_exp2 = sum((weights[k] * current2[n-k] * np.exp((-k-1) * k_sum2)) for k in range(n))

            current1[n] = (((self.cv_constant / (1 + big_k1))
                            - ((sum_sums + (big_k1 * sum1_exp1) - sum2_exp1) / (1 + big_k1))
                            - ((sum_sums + (big_k2 * sum1_exp2) - sum2_exp2)
                               / ((xi_function1[n] * (1 + big_k2)) / self.diffusion_ratio)))
                           / ((self.velocity_constant / (np.power(xi_function1[n], self.alpha) * self.k_0))
                              + (self.diffusion_ratio / xi_function1[n]) + 1))

            current2[n] = ((((self.cv_constant * big_k1) / (1 + big_k1))
                            - (((big_k1 * sum_sums) - (big_k1 * sum1_exp1) + sum2_exp1) / (1 + big_k1))
                            - (((big_k2 * sum_sums) - (big_k2 * sum1_exp2) + sum2_exp2)
                               / ((xi_function2[n] * (1 + big_k2)) / self.diffusion_ratio)))
                           / ((self.velocity_constant / (np.power(xi_function2[n], self.alpha_second_e)
                                                         * self.k_0_second_e))
                              + (self.diffusion_ratio / xi_function2[n]) + 1))

        current = current1 + current2
        return potential, current
