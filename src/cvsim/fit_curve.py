"""
Module for cyclic voltammogram fitting, using a semi-integration
simulated CV for one- and two-electron processes.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias, Callable
import numpy as np
from scipy.optimize import curve_fit
from .mechanisms import CyclicVoltammetryScheme, E_rev, E_q, E_qC, EE, SquareScheme


_ParamGuess: TypeAlias = None | float | tuple[float, float] | tuple[float, float, float]

# Max allowable position for initial voltage oscillations in CV scan. Oftentimes the
# first few voltage points returned from potentiostat can oscillate without a defined scan direction.
VOLTAGE_OSCILLATION_LIMIT = 10


class FitMechanism(ABC):
    """
    Scheme for fitting cyclic voltammograms.

    Parameters
    ----------
    voltage_to_fit : list[float] | np.ndarray
        Array of voltage data of the CV to fit.
    current_to_fit : list[float] | np.ndarray
        Array of current data of the CV to fit.
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    step_size : float
        Voltage increment during CV scan (mV).
    disk_radius : float
        Radius of disk macro-electrode (mm).
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).
    reduction_potential : float | None
        Reduction potential of the one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_reactant : float | None
        Diffusion coefficient of reactant (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_product : float | None
        Diffusion coefficient of product (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_rate: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_product: float | None = None,
    ) -> None:
        if len(voltage_to_fit) != len(current_to_fit):
            raise ValueError("'voltage_to_fit' and 'current_to_fit' must be equal length")

        self._ensure_positive('step_size', step_size)
        self._ensure_positive('scan_rate', scan_rate)
        self._ensure_positive('disk_radius', disk_radius)
        self._ensure_positive('c_bulk', c_bulk)
        self._ensure_positive('temperature', temperature)
        self._ensure_positive_or_none('diffusion_reactant', diffusion_reactant)
        self._ensure_positive_or_none('diffusion_product', diffusion_product)

        self.current_to_fit = current_to_fit
        self.scan_rate = scan_rate
        self.c_bulk = c_bulk
        self.step_size = step_size
        self.disk_radius = disk_radius
        self.temperature = temperature
        self.reduction_potential = reduction_potential
        self.diffusion_reactant = diffusion_reactant
        self.diffusion_product = diffusion_product

        # rounding the start/reverse potentials from the input experimental voltage data to
        # 2 decimal places helps reduce noise and--based on authors' experience--it is pretty rare
        # to see start/reverse potentials initialized in the lab being declared to the third decimal place.
        self.start_potential = round(voltage_to_fit[0], 2)

        start_potential_mv = round(self.start_potential * 1000)

        if voltage_to_fit[VOLTAGE_OSCILLATION_LIMIT] > self.start_potential:
            # scan starts towards more positive
            self.switch_potential = round(max(voltage_to_fit), 2)
        else:
            # scan starts towards more negative
            self.switch_potential = round(min(voltage_to_fit), 2)

        switch_potential_mv = round(self.switch_potential * 1000)

        # make a cleaner x array
        scan_direction = -1 if self.start_potential < self.switch_potential else 1
        delta_theta = scan_direction * self.step_size

        thetas = [round((i - delta_theta)) for i in [start_potential_mv, switch_potential_mv]]
        forward_scan = np.arange(thetas[0], thetas[1], step=delta_theta * -1)
        reverse_scan = np.append(forward_scan[-2::-1], start_potential_mv)
        self.voltage = np.concatenate([forward_scan, reverse_scan]) / 1000

        # Contains only variables with a user-specified fixed value.
        # These params are shared by all CVsim mechanisms
        self.fixed_vars = {
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        }

        # Values are [initial guess, lower bound, upper bound]
        # These params are shared by all CVsim mechanisms
        self.default_vars = {
            'reduction_potential': [
                round((self.voltage[np.argmax(self.current_to_fit)]
                       + self.voltage[np.argmin(self.current_to_fit)]) / 2, 3),
                min(self.start_potential, self.switch_potential),
                max(self.start_potential, self.switch_potential),
            ],
            'diffusion_reactant': [1e-6, 5e-8, 1e-4],
            'diffusion_product': [1e-6, 5e-8, 1e-4],
        }

    @staticmethod
    def _ensure_positive(param: str, value: float):
        if value <= 0.0:
            raise ValueError(f"'{param}' must be > 0.0")

    @staticmethod
    def _ensure_positive_or_none(param: str, value: float | None):
        if value is not None and value <= 0.0:
            raise ValueError(f"'{param}' must be > 0.0 or None")

    @staticmethod
    def _ensure_open_unit_interval_or_none(param: str, value: float | None):
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError(f"'{param}' must be between 0.0 and 1.0, or None")

    @staticmethod
    def _non_none_dict(mapping: dict):
        return {k: v for k, v in mapping.items() if v is not None}

    @staticmethod
    def _fit_var_checker(fit_vars: dict, fit_default_vars: dict) -> dict:
        # take fit_vars dict, for each in it, replace the initial guess/bounds if specified
        for param, value in fit_vars.items():
            if isinstance(value, float | int):
                # Initial guess
                fit_default_vars[param][0] = value
            elif isinstance(value, tuple) and len(value) == 2:
                # Lower and upper bound
                if value[0] >= value[1]:
                    raise ValueError(f"'{param}' lower bound must be lower than upper bound")
                fit_default_vars[param][1] = value[0]
                fit_default_vars[param][2] = value[1]
            elif isinstance(value, tuple) and len(value) == 3:
                if not value[1] < value[0] < value[2]:
                    raise ValueError(f"'{param}' lower bound must be lower than upper bound and guess between them")
                fit_default_vars[param] = list(value)
            elif not None:
                raise ValueError(f"'{param}' allowed inputs: "
                                 f"None | float | tuple[float, float] | tuple[float, float, float]")
        return fit_default_vars

    @abstractmethod
    def _scheme(self, get_var: Callable[[str], float]) -> CyclicVoltammetryScheme:
        raise NotImplementedError

    def _fit(self, fit_vars: dict[str, _ParamGuess]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        fit_vars = self._non_none_dict(fit_vars)
        fixed_vars = self._non_none_dict(self.fixed_vars)

        # check intersection of fixed_vars / fit_vars dicts. if so raise error
        intersection_errors = fixed_vars.keys() & fit_vars.keys()
        if intersection_errors:
            raise ValueError(f"Cannot input fixed value and guess value for {*intersection_errors,}")

        # get params that will be fit
        fitting_params = [
            param for param in self.default_vars
            if param not in fixed_vars.keys()
        ]

        # create deep copy of default dict, and trim to set of fit variables
        fit_default_vars = {k: list(v) for k, v in self.default_vars.items() if k in fitting_params}
        var_index = {var: index for index, var in enumerate(fit_default_vars.keys())}

        fit_default_vars = self._fit_var_checker(fit_vars, fit_default_vars)

        for param, (initial, lower, upper) in fit_default_vars.items():
            if not lower < initial < upper:
                # check if default initial guess is outside bounds, set guess to avg of bounds
                # not useful if spans many order of magnitudes, could use logarithmic mean
                fit_default_vars[param] = [(lower + upper) / 2, lower, upper]
                # check if user's guess was outside bounds
                if initial != self.default_vars[param][0]:
                    raise ValueError(f"Initial guess for '{param}' is outside user-defined bounds")

        print(f"final fitting vars: {fit_default_vars}")
        initial_guesses, lower_bounds, upper_bounds = zip(*fit_default_vars.values())

        print(f'Initial guesses: {initial_guesses}')
        print(f'Lower/Upper bounds: {lower_bounds}/{upper_bounds}')
        print(f'Fixed params: {list(fixed_vars)}')
        print(f'Fitting for: {list(fitting_params)}')

        def get_var(args: tuple[float, ...], param: str) -> float:
            # Helper function to retrieve value for fixed variable if it exists, or retrieve the
            # guess for the parameter that is passed in via curve_fit.
            if param in fixed_vars:
                return fixed_vars[param]
            return args[var_index[param]]

        def fit_function(
                x: list[float] | np.ndarray,  # pylint: disable=unused-argument
                *args: float,
        ) -> np.ndarray:
            # Inner function used by scipy's curve_fit to fit a CV according to the mechanism.
            # Note that Scipy's `curve_fit` does not allow for the user to pass in a function with various dynamic
            # parameters so `fit_function` and `get_var` are used to pass CV simulations to `curve_fit` with optional
            # inputs of initial guesses/bounds from `fit`.
            print(f"trying values: {args}")

            _, i_fit = self._scheme(lambda param: get_var(args, param)).simulate()
            return i_fit

        # fit raw data but exclude first data point, as semi-analytical method skips time=0
        fit_results = curve_fit(
            f=fit_function,
            xdata=self.voltage,
            ydata=self.current_to_fit[1:],
            p0=initial_guesses,
            bounds=[lower_bounds, upper_bounds],
        )

        popt, pcov = list(fit_results)
        current_fit = fit_function(self.voltage, *popt)
        sigma = np.sqrt(np.diag(pcov))  # one standard deviation of the parameters

        final_fit: dict[str, float] = {}
        for val, error, param in zip(popt, sigma, fitting_params):
            final_fit[param] = val
            print(f"Final fit: '{param}': {val:.2E} +/- {error:.0E}")

        # Semi-analytical method does not compute the first point (i.e. time=0)
        # so the starting voltage data point with a zero current is reinserted
        self.voltage = np.insert(self.voltage, 0, self.start_potential)
        current_fit = np.insert(current_fit, 0, 0)
        return self.voltage, current_fit, final_fit


class FitE_rev(FitMechanism):
    """
    Scheme for fitting a CV for a reversible (Nernstian) one-electron transfer mechanism.

    Parameters
    ----------
    voltage_to_fit : list[float] | np.ndarray
        Array of voltage data of the CV to fit.
    current_to_fit : list[float] | np.ndarray
        Array of current data of the CV to fit.
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    step_size : float
        Voltage increment during CV scan (mV).
    disk_radius : float
        Radius of disk macro-electrode (mm).
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).
    reduction_potential : float | None
        Reduction potential of the one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_reactant : float | None
        Diffusion coefficient of reactant (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_product : float | None
        Diffusion coefficient of product (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_rate: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_product: float | None = None,
    ) -> None:
        super().__init__(
            voltage_to_fit,
            current_to_fit,
            scan_rate,
            c_bulk,
            step_size,
            disk_radius,
            temperature,
            reduction_potential,
            diffusion_reactant,
            diffusion_product,
        )

    def _scheme(self, get_var: Callable[[str], float]) -> CyclicVoltammetryScheme:
        return E_rev(
                start_potential=self.start_potential,
                switch_potential=self.switch_potential,
                reduction_potential=get_var('reduction_potential'),
                scan_rate=self.scan_rate,
                c_bulk=self.c_bulk,
                diffusion_reactant=get_var('diffusion_reactant'),
                diffusion_product=get_var('diffusion_product'),
                step_size=self.step_size,
                disk_radius=self.disk_radius,
                temperature=self.temperature,
        )

    def fit(
            self,
            reduction_potential: _ParamGuess = None,
            diffusion_reactant: _ParamGuess = None,
            diffusion_product: _ParamGuess = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the CV for a reversible (Nernstian) one-electron transfer mechanism.
        If a parameter is given, it must be a: float for initial guess of parameter; tuple[float, float] for
        (lower bound, upper bound) of the initial guess; or tuple[float, float, float] for
        (initial guess, lower bound, upper bound).

        Parameters
        ----------
        reduction_potential : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the one-electron transfer process (V vs. reference).
            Defaults to None.
        diffusion_reactant : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of reactant (cm^2/s).
            Defaults to None.
        diffusion_product : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of product (cm^2/s).
            Defaults to None.

        Returns
        -------
        voltage : np.ndarray
            Array of potential (V) values of the CV fit.
        current_fit : np.ndarray
            Array of current (A) values of the CV fit.
        final_fit : dict[str, float]
            Dictionary of final fitting parameter values of the CV fit.

        """

        return self._fit({
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        })


class FitE_q(FitMechanism):
    """
    Scheme for fitting a CV for a quasi-reversible one-electron transfer mechanism.

    Parameters
    ----------
    voltage_to_fit : list[float] | np.ndarray
        Array of voltage data of the CV to fit.
    current_to_fit : list[float] | np.ndarray
        Array of current data of the CV to fit.
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    step_size : float
        Voltage increment during CV scan (mV).
    disk_radius : float
        Radius of disk macro-electrode (mm).
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).
    reduction_potential : float | None
        Reduction potential of the one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_reactant : float | None
        Diffusion coefficient of reactant (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_product : float | None
        Diffusion coefficient of product (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    alpha : float | None
        Charge transfer coefficient (no units).
        If known, can be fixed value, otherwise defaults to None.
    k0 : float | None
        Standard electrochemical rate constant (cm/s).
        If known, can be fixed value, otherwise defaults to None.

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_rate: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_product: float | None = None,
            alpha: float | None = None,
            k0: float | None = None,
    ) -> None:
        super().__init__(
            voltage_to_fit,
            current_to_fit,
            scan_rate,
            c_bulk,
            step_size,
            disk_radius,
            temperature,
            reduction_potential,
            diffusion_reactant,
            diffusion_product,
        )

        self._ensure_open_unit_interval_or_none('alpha', alpha)
        self._ensure_positive_or_none('k0', k0)

        self.alpha = alpha
        self.k0 = k0

        self.fixed_vars |= {
            'alpha': alpha,
            'k0': k0,
        }

        # default [initial guess, lower bound, upper bound]
        self.default_vars |= {
            'alpha': [0.5, 0.01, 0.99],
            'k0': [1e-5, 1e-8, 1e-3],
        }

    def _scheme(self, get_var: Callable[[str], float]) -> CyclicVoltammetryScheme:
        return E_q(
            start_potential=self.start_potential,
            switch_potential=self.switch_potential,
            reduction_potential=get_var('reduction_potential'),
            scan_rate=self.scan_rate,
            c_bulk=self.c_bulk,
            diffusion_reactant=get_var('diffusion_reactant'),
            diffusion_product=get_var('diffusion_product'),
            alpha=get_var('alpha'),
            k0=get_var('k0'),
            step_size=self.step_size,
            disk_radius=self.disk_radius,
            temperature=self.temperature,
        )

    def fit(
            self,
            reduction_potential: _ParamGuess = None,
            diffusion_reactant: _ParamGuess = None,
            diffusion_product: _ParamGuess = None,
            alpha: _ParamGuess = None,
            k0: _ParamGuess = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the CV for a quasi-reversible one-electron transfer mechanism.
        If a parameter is given, it must be a: float for initial guess of parameter; tuple[float, float] for
        (lower bound, upper bound) of the initial guess; or tuple[float, float, float] for
        (initial guess, lower bound, upper bound).

        Parameters
        ----------
        reduction_potential : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the one-electron transfer process (V vs. reference).
            Defaults to None.
        diffusion_reactant : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of reactant (cm^2/s).
            Defaults to None.
        diffusion_product : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of product (cm^2/s).
            Defaults to None.
        alpha : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the charge transfer coefficient (no units).
            Defaults to None.
        k0 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the standard electrochemical rate constant (cm/s).
            Defaults to None.

        Returns
        -------
        voltage : np.ndarray
            Array of potential (V) values of the CV fit.
        current_fit : np.ndarray
            Array of current (A) values of the CV fit.
        final_fit : dict[str, float]
            Dictionary of final fitting parameter values of the CV fit.

        """
        return self._fit({
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
            'alpha': alpha,
            'k0': k0,
        })


class FitE_qC(FitMechanism):
    """
    Scheme for fitting a CV for a quasi-reversible one-electron transfer, followed by
    a reversible first order homogeneous chemical transformation mechanism.

    Parameters
    ----------
    voltage_to_fit : list[float] | np.ndarray
        Array of voltage data of the CV to fit.
    current_to_fit : list[float] | np.ndarray
        Array of current data of the CV to fit.
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    step_size : float
        Voltage increment during CV scan (mV).
    disk_radius : float
        Radius of disk macro-electrode (mm).
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).
    reduction_potential : float | None
        Reduction potential of the one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_reactant : float | None
        Diffusion coefficient of reactant (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_product : float | None
        Diffusion coefficient of product (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    alpha : float | None
        Charge transfer coefficient (no units).
        If known, can be fixed value, otherwise defaults to None.
    k0 : float | None
        Standard electrochemical rate constant (cm/s).
        If known, can be fixed value, otherwise defaults to None.
    k_forward : float | None
        First order forward chemical rate constant (1/s).
        If known, can be fixed value, otherwise defaults to None.
    k_backward : float | None
        First order backward chemical rate constant (1/s).
        If known, can be fixed value, otherwise defaults to None.

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_rate: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_product: float | None = None,
            alpha: float | None = None,
            k0: float | None = None,
            k_forward: float | None = None,
            k_backward: float | None = None,
    ) -> None:
        super().__init__(
            voltage_to_fit,
            current_to_fit,
            scan_rate,
            c_bulk,
            step_size,
            disk_radius,
            temperature,
            reduction_potential,
            diffusion_reactant,
            diffusion_product,
        )

        self._ensure_open_unit_interval_or_none('alpha', alpha)
        self._ensure_positive_or_none('k0', k0)
        self._ensure_positive_or_none('k_forward', k_forward)
        self._ensure_positive_or_none('k_backward', k_backward)

        self.alpha = alpha
        self.k0 = k0
        self.k_forward = k_forward
        self.k_backward = k_backward

        self.fixed_vars |= {
            'alpha': alpha,
            'k0': k0,
            'k_forward': k_forward,
            'k_backward': k_backward,
        }

        # default [initial guess, lower bound, upper bound]
        self.default_vars |= {
            'alpha': [0.5, 0.01, 0.99],
            'k0': [1e-5, 1e-8, 1e-3],
            'k_forward': [1e-3, 1e-8, 1e3],
            'k_backward': [1e-3, 1e-8, 1e3],
        }

    def _scheme(self, get_var: Callable[[str], float]) -> CyclicVoltammetryScheme:
        return E_qC(
            start_potential=self.start_potential,
            switch_potential=self.switch_potential,
            reduction_potential=get_var('reduction_potential'),
            scan_rate=self.scan_rate,
            c_bulk=self.c_bulk,
            diffusion_reactant=get_var('diffusion_reactant'),
            diffusion_product=get_var('diffusion_product'),
            alpha=get_var('alpha'),
            k0=get_var('k0'),
            k_forward=get_var('k_forward'),
            k_backward=get_var('k_backward'),
            step_size=self.step_size,
            disk_radius=self.disk_radius,
            temperature=self.temperature,
        )

    def fit(
            self,
            reduction_potential: _ParamGuess = None,
            diffusion_reactant: _ParamGuess = None,
            diffusion_product: _ParamGuess = None,
            alpha: _ParamGuess = None,
            k0: _ParamGuess = None,
            k_forward: _ParamGuess = None,
            k_backward: _ParamGuess = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the CV for a quasi-reversible one-electron transfer, followed by a reversible first
        order homogeneous chemical transformation mechanism.
        If a parameter is given, it must be a: float for initial guess of parameter; tuple[float, float] for
        (lower bound, upper bound) of the initial guess; or tuple[float, float, float] for
        (initial guess, lower bound, upper bound).

        Parameters
        ----------
        reduction_potential : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the one-electron transfer process (V vs. reference).
            Defaults to None.
        diffusion_reactant : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of reactant (cm^2/s).
            Defaults to None.
        diffusion_product : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of product (cm^2/s).
            Defaults to None.
        alpha : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the charge transfer coefficient (no units).
            Defaults to None.
        k0 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the standard electrochemical rate constant (cm/s).
            Defaults to None.
        k_forward : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the first order forward chemical rate constant (1/s).
            Defaults to None.
        k_backward : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the first order backward chemical rate constant (1/s).
            Defaults to None.

        Returns
        -------
        voltage : np.ndarray
            Array of potential (V) values of the CV fit.
        current_fit : np.ndarray
            Array of current (A) values of the CV fit.
        final_fit : dict[str, float]
            Dictionary of final fitting parameter values of the CV fit.

        """
        return self._fit({
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
            'alpha': alpha,
            'k0': k0,
            'k_forward': k_forward,
            'k_backward': k_backward,
        })

class FitEE(FitMechanism):
    """
    Scheme for fitting a CV for a two successive one-electron quasi-reversible transfer mechanism.

    Parameters
    ----------
    voltage_to_fit : list[float] | np.ndarray
        Array of voltage data of the CV to fit.
    current_to_fit : list[float] | np.ndarray
        Array of current data of the CV to fit.
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    step_size : float
        Voltage increment during CV scan (mV).
    disk_radius : float
        Radius of disk macro-electrode (mm).
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).
    reduction_potential : float | None
        Reduction potential of the first one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    reduction_potential2 : float | None
        Reduction potential of the second one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_reactant : float | None
        Diffusion coefficient of reactant (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_intermediate : float | None
        Diffusion coefficient of intermediate (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_product : float | None
        Diffusion coefficient of product (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    alpha : float | None
        Charge transfer coefficient of first redox process (no units).
        If known, can be fixed value, otherwise defaults to None.
    alpha2 : float | None
        Charge transfer coefficient of second redox process (no units).
        If known, can be fixed value, otherwise defaults to None.
    k0 : float | None
        Standard electrochemical rate constant of first redox process (cm/s).
        If known, can be fixed value, otherwise defaults to None.
    k0_2 : float | None
        Standard electrochemical rate constant of second redox process (cm/s).
        If known, can be fixed value, otherwise defaults to None.

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_rate: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            reduction_potential2: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_intermediate: float | None = None,
            diffusion_product: float | None = None,
            alpha: float | None = None,
            alpha2: float | None = None,
            k0: float | None = None,
            k0_2: float | None = None,
    ) -> None:
        super().__init__(
            voltage_to_fit,
            current_to_fit,
            scan_rate,
            c_bulk,
            step_size,
            disk_radius,
            temperature,
            reduction_potential,
            diffusion_reactant,
            diffusion_product,
        )

        self._ensure_positive_or_none('diffusion_intermediate', diffusion_intermediate)
        self._ensure_open_unit_interval_or_none('alpha', alpha)
        self._ensure_open_unit_interval_or_none('alpha2', alpha2)
        self._ensure_positive_or_none('k0', k0)
        self._ensure_positive_or_none('k0_2', k0_2)

        self.reduction_potential2 = reduction_potential2
        self.diffusion_intermediate = diffusion_intermediate
        self.alpha = alpha
        self.alpha2 = alpha2
        self.k0 = k0
        self.k0_2 = k0_2

        self.fixed_vars |= {
            'reduction_potential2': reduction_potential2,
            'diffusion_intermediate': diffusion_intermediate,
            'alpha': alpha,
            'alpha2': alpha2,
            'k0': k0,
            'k0_2': k0_2,
        }

        # default [initial guess, lower bound, upper bound]
        self.default_vars |= {
            'reduction_potential2': [
                round((self.voltage[np.argmax(self.current_to_fit)]
                       + self.voltage[np.argmin(self.current_to_fit)]) / 2, 3),
                min(self.start_potential, self.switch_potential),
                max(self.start_potential, self.switch_potential),
            ],
            'diffusion_intermediate': [1e-6, 5e-8, 1e-4],
            'alpha': [0.5, 0.01, 0.99],
            'alpha2': [0.5, 0.01, 0.99],
            'k0': [1e-5, 1e-8, 1e-3],
            'k0_2': [1e-5, 1e-8, 1e-3],
        }

    def _scheme(self, get_var: Callable[[str], float]) -> CyclicVoltammetryScheme:
        return EE(
            start_potential=self.start_potential,
            switch_potential=self.switch_potential,
            reduction_potential=get_var('reduction_potential'),
            reduction_potential2=get_var('reduction_potential2'),
            scan_rate=self.scan_rate,
            c_bulk=self.c_bulk,
            diffusion_reactant=get_var('diffusion_reactant'),
            diffusion_intermediate=get_var('diffusion_intermediate'),
            diffusion_product=get_var('diffusion_product'),
            alpha=get_var('alpha'),
            alpha2=get_var('alpha2'),
            k0=get_var('k0'),
            k0_2=get_var('k0_2'),
            step_size=self.step_size,
            disk_radius=self.disk_radius,
            temperature=self.temperature,
        )

    def fit(
            self,
            reduction_potential: _ParamGuess = None,
            reduction_potential2: _ParamGuess = None,
            diffusion_reactant: _ParamGuess = None,
            diffusion_intermediate: _ParamGuess = None,
            diffusion_product: _ParamGuess = None,
            alpha: _ParamGuess = None,
            alpha2: _ParamGuess = None,
            k0: _ParamGuess = None,
            k0_2: _ParamGuess = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the CV for a two successive one-electron quasi-reversible transfer mechanism.
        If a parameter is given, it must be a: float for initial guess of parameter; tuple[float, float] for
        (lower bound, upper bound) of the initial guess; or tuple[float, float, float] for
        (initial guess, lower bound, upper bound).

        Parameters
        ----------
        reduction_potential : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the first one-electron transfer process (V vs. reference).
            Defaults to None.
        reduction_potential2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the second one-electron transfer process (V vs. reference).
            Defaults to None.
        diffusion_reactant : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of reactant (cm^2/s).
            Defaults to None.
        diffusion_intermediate : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of intermediate (cm^2/s).
            Defaults to None.
        diffusion_product : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of product (cm^2/s).
            Defaults to None.
        alpha : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the charge transfer coefficient of the first redox process (no units).
            Defaults to None.
        alpha2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the charge transfer coefficient of the second redox process (no units).
            Defaults to None.
        k0 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the standard electrochemical rate constant of the first redox process (cm/s).
            Defaults to None.
        k0_2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the standard electrochemical rate constant of the second redox process (cm/s).
            Defaults to None.

        Returns
        -------
        voltage : np.ndarray
            Array of potential (V) values of the CV fit.
        current_fit : np.ndarray
            Array of current (A) values of the CV fit.
        final_fit : dict[str, float]
            Dictionary of final fitting parameter values of the CV fit.

        """

        return self._fit({
            'reduction_potential': reduction_potential,
            'reduction_potential2': reduction_potential2,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_intermediate': diffusion_intermediate,
            'diffusion_product': diffusion_product,
            'alpha': alpha,
            'alpha2': alpha2,
            'k0': k0,
            'k0_2': k0_2,
        })


class FitSquareScheme(FitMechanism):
    """
    Scheme for fitting a CV for two quasi-reversible, one-electron transfers of homogeneously
    interconverting reactants (Square Scheme) mechanism.

    Parameters
    ----------
    voltage_to_fit : list[float] | np.ndarray
        Array of voltage data of the CV to fit.
    current_to_fit : list[float] | np.ndarray
        Array of current data of the CV to fit.
    scan_rate : float
        Potential sweep rate (V/s).
    c_bulk : float
        Bulk concentration of redox species (mM or mol/m^3).
    step_size : float
        Voltage increment during CV scan (mV).
    disk_radius : float
        Radius of disk macro-electrode (mm).
    temperature : float
        Temperature (K).
        Default is 298.0 K (24.85C).
    reduction_potential : float | None
        Reduction potential of the first one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    reduction_potential2 : float | None
        Reduction potential of the second one-electron transfer process (V vs. reference).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_reactant : float | None
        Diffusion coefficient of reactant (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    diffusion_product : float | None
        Diffusion coefficient of product (cm^2/s).
        If known, can be fixed value, otherwise defaults to None.
    alpha : float | None
        Charge transfer coefficient of first redox process (no units).
        If known, can be fixed value, otherwise defaults to None.
    alpha2 : float | None
        Charge transfer coefficient of second redox process (no units).
        If known, can be fixed value, otherwise defaults to None.
    k0 : float | None
        Standard electrochemical rate constant of first redox process (cm/s).
        If known, can be fixed value, otherwise defaults to None.
    k0_2 : float | None
        Standard electrochemical rate constant of second redox process (cm/s).
        If known, can be fixed value, otherwise defaults to None.
    k_forward : float | None
        First order forward chemical rate constant for first redox species (1/s).
        If known, can be fixed value, otherwise defaults to None.
    k_backward : float | None
        First order backward chemical rate constant for first redox species (1/s).
        If known, can be fixed value, otherwise defaults to None.
    k_forward2 : float | None
        First order forward chemical rate constant for second redox species (1/s).
        If known, can be fixed value, otherwise defaults to None.
    k_backward2 : float | None
        First order backward chemical rate constant for second redox species (1/s).
        If known, can be fixed value, otherwise defaults to None.

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_rate: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            reduction_potential2: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_product: float | None = None,
            alpha: float | None = None,
            alpha2: float | None = None,
            k0: float | None = None,
            k0_2: float | None = None,
            k_forward: float | None = None,
            k_backward: float | None = None,
            k_forward2: float | None = None,
            k_backward2: float | None = None,
    ) -> None:
        super().__init__(
            voltage_to_fit,
            current_to_fit,
            scan_rate,
            c_bulk,
            step_size,
            disk_radius,
            temperature,
            reduction_potential,
            diffusion_reactant,
            diffusion_product,
        )

        self._ensure_open_unit_interval_or_none('alpha', alpha)
        self._ensure_open_unit_interval_or_none('alpha2', alpha2)
        self._ensure_positive_or_none('k0', k0)
        self._ensure_positive_or_none('k0_2', k0_2)
        self._ensure_positive_or_none('k_forward', k_forward)
        self._ensure_positive_or_none('k_backward', k_backward)
        self._ensure_positive_or_none('k_forward2', k_forward2)
        self._ensure_positive_or_none('k_backward2', k_backward2)

        self.reduction_potential2 = reduction_potential2
        self.alpha = alpha
        self.alpha2 = alpha2
        self.k0 = k0
        self.k0_2 = k0_2
        self.k_forward = k_forward
        self.k_backward = k_backward
        self.k_forward2 = k_forward2
        self.k_backward2 = k_backward2

        self.fixed_vars |= {
            'reduction_potential2': reduction_potential2,
            'alpha': alpha,
            'alpha2': alpha2,
            'k0': k0,
            'k0_2': k0_2,
            'k_forward': k_forward,
            'k_backward': k_backward,
            'k_forward2': k_forward2,
            'k_backward2': k_backward2,
        }

        # default [initial guess, lower bound, upper bound]
        self.default_vars |= {
            'reduction_potential2': [
                round((self.voltage[np.argmax(self.current_to_fit)]
                       + self.voltage[np.argmin(self.current_to_fit)]) / 2, 3),
                min(self.start_potential, self.switch_potential),
                max(self.start_potential, self.switch_potential),
            ],
            'alpha': [0.5, 0.01, 0.99],
            'alpha2': [0.5, 0.01, 0.99],
            'k0': [1e-5, 1e-8, 1e-3],
            'k0_2': [1e-5, 1e-8, 1e-3],
            'k_forward': [1e-1, 5e-4, 1e3],
            'k_backward': [1e-1, 5e-4, 1e3],
            'k_forward2': [1e-1, 5e-4, 1e3],
            'k_backward2': [1e-1, 5e-4, 1e3],
        }

    def _scheme(self, get_var: Callable[[str], float]) -> CyclicVoltammetryScheme:
        return SquareScheme(
            start_potential=self.start_potential,
            switch_potential=self.switch_potential,
            reduction_potential=get_var('reduction_potential'),
            reduction_potential2=get_var('reduction_potential2'),
            scan_rate=self.scan_rate,
            c_bulk=self.c_bulk,
            diffusion_reactant=get_var('diffusion_reactant'),
            diffusion_product=get_var('diffusion_product'),
            alpha=get_var('alpha'),
            alpha2=get_var('alpha2'),
            k0=get_var('k0'),
            k0_2=get_var('k0_2'),
            k_forward=get_var('k_forward'),
            k_backward=get_var('k_backward'),
            k_forward2=get_var('k_forward2'),
            k_backward2=get_var('k_backward2'),
            step_size=self.step_size,
            disk_radius=self.disk_radius,
            temperature=self.temperature,
        )

    def fit(
            self,
            reduction_potential: _ParamGuess = None,
            reduction_potential2: _ParamGuess = None,
            diffusion_reactant: _ParamGuess = None,
            diffusion_product: _ParamGuess = None,
            alpha: _ParamGuess = None,
            alpha2: _ParamGuess = None,
            k0: _ParamGuess = None,
            k0_2: _ParamGuess = None,
            k_forward: _ParamGuess = None,
            k_backward: _ParamGuess = None,
            k_forward2: _ParamGuess = None,
            k_backward2: _ParamGuess = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """
        Fits the CV for a Square Scheme mechanism.
        If a parameter is given, it must be a: float for initial guess of parameter; tuple[float, float] for
        (lower bound, upper bound) of the initial guess; or tuple[float, float, float] for
        (initial guess, lower bound, upper bound).

        Parameters
        ----------
        reduction_potential : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the first one-electron transfer process (V vs. reference).
            Defaults to None.
        reduction_potential2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential of the second one-electron transfer process (V vs. reference).
            Defaults to None.
        diffusion_reactant : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of reactant (cm^2/s).
            Defaults to None.
        diffusion_product : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of product (cm^2/s).
            Defaults to None.
        alpha : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the charge transfer coefficient of the first redox process (no units).
            Defaults to None.
        alpha2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the charge transfer coefficient of the second redox process (no units).
            Defaults to None.
        k0 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the standard electrochemical rate constant of the first redox process (cm/s).
            Defaults to None.
        k0_2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the standard electrochemical rate constant of the second redox process (cm/s).
            Defaults to None.
        k_forward : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the first order forward chemical rate constant for the first redox species (1/s).
            Defaults to None.
        k_backward : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the first order backward chemical rate constant for the first redox species (1/s).
            Defaults to None.
        k_forward2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the first order forward chemical rate constant for the second redox species (1/s).
            Defaults to None.
        k_backward2 : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the first order backward chemical rate constant for the second redox species (1/s).
            Defaults to None.

        Returns
        -------
        voltage : np.ndarray
            Array of potential (V) values of the CV fit.
        current_fit : np.ndarray
            Array of current (A) values of the CV fit.
        final_fit : dict[str, float]
            Dictionary of final fitting parameter values of the CV fit.

        """

        return self._fit({
            'reduction_potential': reduction_potential,
            'reduction_potential2': reduction_potential2,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
            'alpha': alpha,
            'alpha2': alpha2,
            'k0': k0,
            'k0_2': k0_2,
            'k_forward': k_forward,
            'k_backward': k_backward,
            'k_forward2': k_forward2,
            'k_backward2': k_backward2,
        })
