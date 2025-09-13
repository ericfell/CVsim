"""
Module for semi-integration simulation of cyclic voltammetry (CV)
of one- and two-electron processes.

Algorithm Reference:
    [1] Oldham, K. B.; Myland, J. C. "Modelling cyclic voltammetry without
    digital simulation." Electrochimica Acta, 56, 2011, 10612-10625.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as spc

# Faraday constant (C/mol)
F = spc.value('Faraday constant')

# Molar gas constant (J/K/mol)
R = spc.R


class CyclicVoltammetryScheme(ABC):
    """
    A scheme for a simulated cyclic voltammogram on a disk macro-electrode.

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs. reference).
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
        Default is 298.0 K (24.85C).

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
        self._ensure_positive('step_size', step_size)
        self._ensure_positive('scan_rate', scan_rate)
        self._ensure_positive('diffusion_reactant', diffusion_reactant)
        self._ensure_positive('diffusion_product', diffusion_product)
        self._ensure_positive('disk_radius', disk_radius)
        self._ensure_positive('c_bulk', c_bulk)
        self._ensure_positive('temperature', temperature)

        if start_potential == switch_potential:
            raise ValueError("'start_potential' and 'switch_potential' must be different")

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
        self.n_max = round((abs(switch_potential - start_potential) * 2) / self.step_size)
        self.scan_direction = -1 if self.start_potential < self.switch_potential else 1

        self.nernst_constant = -F / (R * temperature)
        self.cv_constant = -F * self.scan_direction * self.electrode_area * c_bulk * self.velocity_constant
        self.delta_theta = self.scan_direction * self.step_size

        # Weighting factors for semi-integration method.
        # This is equation (5:5) from [1].
        weights = [1.0] * self.n_max
        for n in range(1, self.n_max):
            weights[n] = (2 * n - 1) * (weights[n - 1] / (2 * n))
        self.semi_integration_weights = weights

    @staticmethod
    def _ensure_positive(param: str, value: float):
        if value <= 0.0:
            raise ValueError(f"'{param}' must be > 0.0")

    @staticmethod
    def _ensure_nonnegative(param: str, value: float):
        if value <= 0.0:
            raise ValueError(f"'{param}' must be > 0.0")

    @staticmethod
    def _ensure_open_unit_interval(param: str, value: float):
        if not 0.0 < value < 1.0:
            raise ValueError(f"'{param}' must be between 0.0 and 1.0")

    def _voltage_profile_setup(self, reduction_potential2: float | None = None) -> tuple[np.ndarray, list]:
        """
        Compute the potential steps for voltage profile and for exponential Nernstian/Butler-Volmer function.
        This is equation (6:1) from [1].

        Parameters
        ----------
        reduction_potential2 : float, optional
            Reduction potential of the second electron transfer process (V vs. reference).
            Only used if a 2 electron process class is called.
            Default is None.

        Returns
        -------
        potential : np.ndarray
            Potential values in a full CV sweep.
        xi_functions : list
            Values from exponential potential equation.

        """

        electron_transfers = [self.reduction_potential]
        if reduction_potential2 is not None:
            electron_transfers.append(reduction_potential2)

        thetas = [round((i - self.delta_theta) * 1000) for i in [self.start_potential, self.switch_potential]]
        forward_scan = np.arange(thetas[0], thetas[1], step=self.delta_theta * -1000)
        reverse_scan = np.append(forward_scan[-2::-1], round(self.start_potential * 1000))
        potential = np.concatenate([forward_scan, reverse_scan]) / 1000

        potential_diff = self.scan_direction * (self.switch_potential - self.start_potential)
        xi_functions = []
        for potential_value in electron_transfers:
            xi_function = np.zeros(self.n_max)
            for k in range(1, self.n_max + 1):
                potential_excursion = self.scan_direction * abs(k * self.step_size + potential_diff)
                xi_function[k - 1] = np.exp(self.scan_direction * self.nernst_constant
                                            * (self.switch_potential + potential_excursion - potential_value))

            xi_functions.append(xi_function)
        return potential, xi_functions

    @abstractmethod
    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates current-potential profile for desired mechanism.

        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """
        raise NotImplementedError


class E_rev(CyclicVoltammetryScheme):
    """
    Provides a current-potential profile for a reversible (Nernstian) one-electron transfer mechanism.
    This is equation (7:7) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs. reference).
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
        Default is 298.0 K (24.85C).

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
            temperature,
        )

    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates the CV for a reversible one-electron transfer mechanism.

        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """

        weights = self.semi_integration_weights
        potential, [xi_function] = self._voltage_profile_setup()

        current = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum_weights = sum(weights[k] * current[n - k] for k in range(n))
            xi_ratio = 1 + (self.diffusion_ratio / xi_function[n])
            current[n] = (self.cv_constant / xi_ratio) - sum_weights
        return potential, current


class E_q(CyclicVoltammetryScheme):
    """
    Provides a current-potential profile for a quasi-reversible one-electron transfer mechanism.
    This is equation (8:3) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs. reference).
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
    k0 : float
        Standard electrochemical rate constant (cm/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).

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
            k0: float,
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
            temperature,
        )
        self._ensure_open_unit_interval('alpha', alpha)
        self._ensure_nonnegative('k0', k0)

        self.alpha = alpha
        self.k0 = k0 / 100  # cm/s to m/s

    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates the CV for a quasi-reversible one-electron transfer mechanism.

        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """

        weights = self.semi_integration_weights
        potential, [xi_function] = self._voltage_profile_setup()

        current = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum_weights = sum(weights[k] * current[n - k] for k in range(n))
            xi_ratio = 1 + (self.diffusion_ratio / xi_function[n])
            xi_alpha = self.k0 * (xi_function[n] ** self.alpha)
            numerator = self.cv_constant - xi_ratio * sum_weights
            denominator = xi_ratio + (self.velocity_constant / xi_alpha)
            current[n] = numerator / denominator

        return potential, current


class E_qC(CyclicVoltammetryScheme):
    """
    Provides a current-potential profile for a quasi-reversible one-electron transfer, followed by a reversible first
    order homogeneous chemical transformation mechanism.
    This is equation (10:4) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    reduction_potential : float
        Reduction potential of the one-electron transfer process (V vs. reference).
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
    k0 : float
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
        Default is 298.0 K (24.85C).

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
            k0: float,
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
            temperature,
        )
        self._ensure_open_unit_interval('alpha', alpha)
        self._ensure_nonnegative('k0', k0)
        self._ensure_nonnegative('k_backward', k_backward)
        self._ensure_positive('k_forward', k_forward)

        self.alpha = alpha
        self.k0 = k0 / 100  # cm/s to m/s
        self.k_forward = k_forward
        self.k_backward = k_backward

    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates the CV for a quasi-reversible one-electron transfer followed by a reversible first order homogeneous
        chemical transformation mechanism.

        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """

        k_sum = self.k_forward + self.k_backward
        k_const = self.k_forward / self.k_backward
        weights = self.semi_integration_weights
        potential, [xi_function] = self._voltage_profile_setup()

        current = np.zeros(self.n_max)
        exp_factors = np.zeros(self.n_max)

        exp_k_sum = np.exp(-k_sum)
        exp_factors[0] = exp_k_sum
        for n in range(1, self.n_max):
            exp_factors[n] = exp_factors[n - 1] * exp_k_sum

        for n in range(self.n_max):
            sum_weights = 0
            sum_exp_weights = 0

            for k in range(n):
                weighted_current = weights[k] * current[n - k]
                sum_weights += weighted_current
                sum_exp_weights += weighted_current * exp_factors[k]

            xi_diffusion = self.diffusion_ratio / ((1 + k_const) * xi_function[n])
            numerator = (self.cv_constant - ((1 + xi_diffusion) * sum_weights)
                         - (k_const * xi_diffusion * sum_exp_weights))

            xi_alpha = self.k0 * (xi_function[n] ** self.alpha)
            denominator = 1 + (self.velocity_constant / xi_alpha) + (self.diffusion_ratio / xi_function[n])

            current[n] = numerator / denominator

        return potential, current


class EE(CyclicVoltammetryScheme):
    """
    Provides a current-potential profile for two successive one-electron quasi-reversible transfers.
    This is equation (12:19) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    reduction_potential : float
        Reduction potential of the first one-electron transfer process (V vs. reference).
    reduction_potential2 : float
        Reduction potential of the second one-electron transfer process (V vs. reference).
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
        Charge transfer coefficient of first redox process (no units).
    alpha2 : float
        Charge transfer coefficient of second redox process (no units).
    k0 : float
        Standard electrochemical rate constant of first redox process (cm/s).
    k0_2 : float
        Standard electrochemical rate constant of second redox process (cm/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            reduction_potential2: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_intermediate: float,
            diffusion_product: float,
            alpha: float,
            alpha2: float,
            k0: float,
            k0_2: float,
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
            temperature,
        )
        self._ensure_open_unit_interval('alpha', alpha)
        self._ensure_open_unit_interval('alpha2', alpha2)
        self._ensure_positive('diffusion_intermediate', diffusion_intermediate)
        self._ensure_positive('k0', k0)
        self._ensure_positive('k0_2', k0_2)

        self.reduction_potential2 = reduction_potential2
        self.diffusion_intermediate = diffusion_intermediate / 1e4  # cm^2/s to m^2/s
        self.alpha = alpha
        self.alpha2 = alpha2
        self.k0 = k0 / 100  # cm/s to m/s
        self.k0_2 = k0_2 / 100  # cm/s to m/s

    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates the CV for two successive one-electron quasi-reversible transfer (EE) mechanism.

        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.

        """

        weights = self.semi_integration_weights
        potential, (xi_function1, xi_function2) = self._voltage_profile_setup(self.reduction_potential2)

        intermediate_const = (self.diffusion_intermediate / self.delta_t) ** 0.5

        # Equation (12:15) from [1]
        z_function = (intermediate_const / self.k0) * (xi_function1 ** (1 - self.alpha))

        # Equation (12:16) from [1]
        y_function = 1 + (xi_function1 * np.sqrt(self.diffusion_intermediate / self.diffusion_reactant))

        # Equation (12:17) from [1]
        w_function = (intermediate_const / self.k0_2) / (xi_function2 ** self.alpha2)

        # Equation (12:18) from [1]
        v_function = 1 + (np.sqrt(self.diffusion_intermediate / self.diffusion_product) / xi_function2)

        w_v_sum = w_function + v_function
        z_y_sum = z_function + y_function

        i_constant = (self.cv_constant / self.velocity_constant) * intermediate_const

        current1 = np.zeros(self.n_max)
        current2 = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum1 = sum(weights[k] * current1[n - k] for k in range(n))
            sum2 = sum(weights[k] * current2[n - k] for k in range(n))

            current1[n] = ((w_v_sum[n] * xi_function1[n] * i_constant
                            - ((y_function[n] * w_v_sum[n] - 1) * sum1) + w_function[n] * sum2)
                           / (z_y_sum[n] * w_v_sum[n] - 1))

            current2[n] = ((xi_function1[n] * i_constant + z_function[n] * sum1
                            - (v_function[n] * z_y_sum[n] - 1) * sum2)
                           / ((z_function[n] + y_function[n]) * w_v_sum[n] - 1))

        current = current1 + current2
        return potential, current


class SquareScheme(CyclicVoltammetryScheme):
    """
    Provides a current-potential profile for two quasi-reversible, one-electron transfers of homogeneously
    interconverting reactants (Square Scheme).
    This is equations (14:14 and 14:15) from [1].

    Parameters
    ----------
    start_potential : float
        Starting potential of scan (V vs. reference).
    switch_potential : float
        Switching potential of scan (V vs. reference).
    reduction_potential : float
        Reduction potential of the first one-electron transfer process (V vs. reference).
    reduction_potential2 : float
        Reduction potential of the second one-electron transfer process (V vs. reference).
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    diffusion_reactant : float
        Diffusion coefficient of reactant (cm^2/s).
    diffusion_product : float
        Diffusion coefficient of product (cm^2/s).
    alpha : float
        Charge transfer coefficient of first redox process (no units).
    alpha2 : float
        Charge transfer coefficient of second redox process (no units).
    k0 : float
        Standard electrochemical rate constant of first redox process (cm/s).
    k0_2 : float
        Standard electrochemical rate constant of second redox process (cm/s).
    k_forward : float
        First order forward chemical rate constant for first redox species (1/s).
    k_backward : float
        First order backward chemical rate constant for first redox species (1/s).
    k_forward2 : float
        First order forward chemical rate constant for second redox species (1/s).
    k_backward2 : float
        First order backward chemical rate constant for second redox species (1/s).
    step_size : float
        Voltage increment during CV scan (mV).
        Default is 1.0 mV, a typical potentiostat default.
    disk_radius : float
        Radius of disk macro-electrode (mm).
        Default is 1.5 mm, a typical working electrode.
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).

    """

    def __init__(
            self,
            start_potential: float,
            switch_potential: float,
            reduction_potential: float,
            reduction_potential2: float,
            scan_rate: float,
            c_bulk: float,
            diffusion_reactant: float,
            diffusion_product: float,
            alpha: float,
            alpha2: float,
            k0: float,
            k0_2: float,
            k_forward: float,
            k_backward: float,
            k_forward2: float,
            k_backward2: float,
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
            temperature,
        )
        self._ensure_open_unit_interval('alpha', alpha)
        self._ensure_open_unit_interval('alpha2', alpha2)
        self._ensure_positive('k0', k0)
        self._ensure_positive('k0_2', k0_2)
        self._ensure_positive('k_backward', k_backward)
        self._ensure_positive('k_backward2', k_backward2)
        self._ensure_nonnegative('k_forward', k_forward)
        self._ensure_nonnegative('k_forward2', k_forward2)

        self.reduction_potential2 = reduction_potential2
        self.alpha = alpha
        self.alpha2 = alpha2
        self.k0 = k0 / 100  # cm/s to m/s
        self.k0_2 = k0_2 / 100  # cm/s to m/s
        self.k_forward = k_forward
        self.k_backward = k_backward
        self.k_forward2 = k_forward2
        self.k_backward2 = k_backward2

    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates the CV for two quasi-reversible, one-electron transfers of homogeneously interconverting
        reactants (Square Scheme).

        Returns
        -------
        potential : np.ndarray
            Potential values in full CV sweep.
        current : np.ndarray
            Current values in full CV sweep.
        """

        weights = self.semi_integration_weights
        potential, (xi_function1, xi_function2) = self._voltage_profile_setup(self.reduction_potential2)

        k_sum1 = self.k_forward + self.k_backward
        big_k1 = self.k_forward / self.k_backward
        k_sum2 = self.k_forward2 + self.k_backward2
        big_k2 = self.k_forward2 / self.k_backward2

        current1 = np.zeros(self.n_max)
        current2 = np.zeros(self.n_max)

        for n in range(self.n_max):
            sum1 = sum(weights[k] * current1[n - k] for k in range(n))
            sum2 = sum(weights[k] * current2[n - k] for k in range(n))
            sum_sums = sum1 + sum2

            sum1_exp1 = sum((weights[k] * current1[n - k] * np.exp((-k - 1) * k_sum1)) for k in range(n))
            sum1_exp2 = sum((weights[k] * current1[n - k] * np.exp((-k - 1) * k_sum2)) for k in range(n))
            sum2_exp1 = sum((weights[k] * current2[n - k] * np.exp((-k - 1) * k_sum1)) for k in range(n))
            sum2_exp2 = sum((weights[k] * current2[n - k] * np.exp((-k - 1) * k_sum2)) for k in range(n))

            current1[n] = (((self.cv_constant / (1 + big_k1))
                            - ((sum_sums + (big_k1 * sum1_exp1) - sum2_exp1) / (1 + big_k1))
                            - ((sum_sums + (big_k2 * sum1_exp2) - sum2_exp2)
                               / ((xi_function1[n] * (1 + big_k2)) / self.diffusion_ratio)))
                           / ((self.velocity_constant / (np.power(xi_function1[n], self.alpha) * self.k0))
                              + (self.diffusion_ratio / xi_function1[n]) + 1))

            current2[n] = ((((self.cv_constant * big_k1) / (1 + big_k1))
                            - (((big_k1 * sum_sums) - (big_k1 * sum1_exp1) + sum2_exp1) / (1 + big_k1))
                            - (((big_k2 * sum_sums) - (big_k2 * sum1_exp2) + sum2_exp2)
                               / ((xi_function2[n] * (1 + big_k2)) / self.diffusion_ratio)))
                           / ((self.velocity_constant / (np.power(xi_function2[n], self.alpha2)
                                                         * self.k0_2))
                              + (self.diffusion_ratio / xi_function2[n]) + 1))

        current = current1 + current2
        return potential, current
