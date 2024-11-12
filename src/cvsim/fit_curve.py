"""
Module for cyclic voltammogram fitting, using a semi-integration
simulated CV for one- and two-electron processes.
"""

from abc import ABC, abstractmethod
from typing import TypeAlias
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
    ) -> None:
        if len(voltage_to_fit) != len(current_to_fit):
            raise ValueError("'voltage_to_fit' and 'current_to_fit' must be equal length")

        self._ensure_positive('step_size', step_size)
        self._ensure_positive('scan_rate', scan_rate)
        self._ensure_positive('disk_radius', disk_radius)
        self._ensure_positive('c_bulk', c_bulk)
        self._ensure_positive('temperature', temperature)

        self.current_to_fit = current_to_fit
        self.scan_rate = scan_rate
        self.c_bulk = c_bulk
        self.step_size = step_size
        self.disk_radius = disk_radius
        self.temperature = temperature
        self.start_voltage = round(voltage_to_fit[0] * 1000)

        if round(voltage_to_fit[VOLTAGE_OSCILLATION_LIMIT] * 1000) > self.start_voltage:
            # scan starts towards more positive
            self.reverse_voltage = round(max(voltage_to_fit) * 1000)
        else:
            # scan starts towards more negative
            self.reverse_voltage = round(min(voltage_to_fit) * 1000)

        # make a cleaner x array
        scan_direction = -1 if self.start_voltage < self.reverse_voltage else 1
        delta_theta = scan_direction * self.step_size

        thetas = [round((i - delta_theta)) for i in [self.start_voltage, self.reverse_voltage]]
        forward_scan = np.arange(thetas[0], thetas[1], step=delta_theta * -1)
        reverse_scan = np.append(forward_scan[-2::-1], self.start_voltage)
        self.voltage_to_fit = np.concatenate([forward_scan, reverse_scan]) / 1000

    @staticmethod
    def _ensure_positive(param: str, value: float):
        if value <= 0.0:
            raise ValueError(f"'{param}' must be > 0.0")

    @staticmethod
    def _ensure_positive_or_none(param: str, value: float | None):
        if value is not None and value <= 0.0:
            raise ValueError(f"'{param}' must be > 0.0 or None")

    @staticmethod
    def _non_none_dict(mapping: dict):
        return {k: v for k, v in mapping.items() if v is not None}

    @abstractmethod
    def fit(self, *args):
        """Fit designated mechanism"""
        return NotImplementedError


class FitE_rev(FitMechanism):
    """
    Scheme for fitting a CV for a reversible (Nernstian) one electron transfer mechanism.

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
            diffusion_product: float | None = None
    ) -> None:
        super().__init__(voltage_to_fit, current_to_fit, scan_rate, c_bulk, step_size, disk_radius, temperature)
        self._ensure_positive_or_none('diffusion_reactant', diffusion_reactant)
        self._ensure_positive_or_none('diffusion_product', diffusion_product)

        self.reduction_potential = reduction_potential  # TODO move into abstract class?
        self.diffusion_reactant = diffusion_reactant  # TODO move into abstract class?
        self.diffusion_product = diffusion_product

        # Contains only variables with a user-specified fixed value
        self.fixed_vars = self._non_none_dict({
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        })

        # Values are [initial guess, lower bound, upper bound]
        self.default_vars = {
            'reduction_potential': [
                round((self.voltage_to_fit[np.argmax(self.current_to_fit)]
                       + self.voltage_to_fit[np.argmin(self.current_to_fit)]) / 2, 3),
                round(min(self.start_voltage, self.reverse_voltage) / 1000, 3),
                round(max(self.start_voltage, self.reverse_voltage) / 1000, 3),
            ],
            'diffusion_reactant': [1e-6, 5e-8, 1e-4],
            'diffusion_product': [1e-6, 5e-8, 1e-4],
        }
        # TODO incorrect inputs, error handling

    def fit(
            self,
            reduction_potential: _ParamGuess = None,
            diffusion_reactant: _ParamGuess = None,
            diffusion_product: _ParamGuess = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits the CV for a reversible (Nernstian) one electron transfer mechanism.
        If a parameter is given, it must be a: float for initial guess of parameter; tuple[float, float] for
        (lower bound, upper bound) of the initial guess; or tuple[float, float, float] for
        (initial guess, lower bound, upper bound).

        Parameters
        ----------
        reduction_potential : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the reduction potential (V vs. reference).
            Defaults to None.
        diffusion_reactant : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of reactant (cm^2/s).
            Defaults to None.
        diffusion_product : None | float | tuple[float, float] | tuple[float, float, float]
            Optional guess for the diffusion coefficient of product (cm^2/s).
            Defaults to None.

        Returns
        -------
        voltage_to_fit : np.ndarray
            Array of potential (V) values of the CV fit.
        current_fit : np.ndarray
            Array of current (A) values of the CV fit.

        """

        fit_vars = self._non_none_dict({
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        })

        # check intersection of fixed_vars / fit_vars dicts. if so raise error
        intersection_errors = self.fixed_vars.keys() & fit_vars.keys()
        if intersection_errors:
            raise ValueError(f"Cannot input fixed value and guess value for {*intersection_errors,}")

        # get params that will be fit
        fitting_params = [
            param for param in self.default_vars.keys()
            if param not in self.fixed_vars.keys()
        ]

        # trim dict to set of fit variables
        fit_default_vars = {k: v for k, v in self.default_vars.items() if k in fitting_params}
        var_index = {var: index for index, var in enumerate(fit_default_vars.keys())}

        # take fit_vars dict, for each in it, replace the initial guess/bounds if specified
        for param, value in fit_vars.items():
            if isinstance(value, float | int):
                # Initial guess
                fit_default_vars[param][0] = value
            elif isinstance(value, tuple) and len(value) == 2:
                # Lower and upper bound
                if value[0] >= value[1]:
                    raise ValueError("Lower bound must be lower than upper bound")
                fit_default_vars[param][1] = value[0]
                fit_default_vars[param][2] = value[1]
            elif isinstance(value, tuple) and len(value) == 3:
                if value[1] >= value[2]:
                    raise ValueError("Lower bound must be lower than upper bound")
                fit_default_vars[param] = list(value)
            else:
                if not None:
                    raise ValueError("Allowed inputs: None | float | tuple[float, float] | tuple[float, float, float]")

        for param, (initial, lower, upper) in fit_default_vars.items():
            if not lower < initial < upper:
                # check if default initial guess is outside bounds, set guess to avg of bounds
                fit_default_vars[param] = [(lower + upper) / 2, lower, upper]
                # check if user's guess was outside bounds
                if initial != self.default_vars[param][0]:
                    raise ValueError(f"Initial guess for '{param}' is outside user-defined bounds")

        print(f"final fitting vars: {fit_default_vars}")
        initial_guesses, lower_bounds, upper_bounds = zip(*fit_default_vars.values())

        print(f'Initial guesses: {initial_guesses}')
        print(f'Lower/Upper bounds: {lower_bounds}/{upper_bounds}')
        print(f'Fixed params: {list(self.fixed_vars)}')
        print(f'Fitting for: {list(fitting_params)}')

        def fit_function(
                x: list[float] | np.ndarray,  # pylint: disable=unused-argument
                *args: float,
        ) -> np.ndarray:
            """
            Inner function used by scipy's curve_fit to fit a CV according to the
            one-electron reversible mechanism.

            Parameters
            ----------
            x : list[float] | np.ndarray
                Array of voltage data of the CV to fit.
            *args : float
                Value(s) for parameter(s) that curve_fit tries during fitting procedure.

            Returns
            -------
            i_fit : np.ndarray
                Array of current (A) values of the CV fit.

            Notes
            -----
            Scipy's `curve_fit` does not allow for the user to pass in a function with various
            dynamic parameters so `fit_function` and its inner function `fetch` are used to pass
            CV simulations to `curve_fit` with optional inputs of initial guesses/bounds from `fit`.

            """

            print(f"trying values: {args}")

            def fetch(param: str) -> float:
                """
                Helper function to retrieve value for fixed variable if it exists, or retrieve the
                guess for the parameter that is passed in via curve_fit

                Parameters
                ----------
                param : str
                    Name of desired CV simulation's input parameter.

                Returns
                -------
                float: If param exists in fixed_vars then its value is returned, otherwise
                        return the value of the parameter in the args passed in via curve_fit.

                """
                if param in self.fixed_vars:
                    return self.fixed_vars[param]
                return args[var_index[param]]

            _, i_fit = E_rev(
                start_potential=round(self.start_voltage / 1000, 3),
                switch_potential=round(self.reverse_voltage / 1000, 3),
                reduction_potential=fetch('reduction_potential'),
                scan_rate=self.scan_rate,
                c_bulk=self.c_bulk,
                diffusion_reactant=fetch('diffusion_reactant'),
                diffusion_product=fetch('diffusion_product'),
                step_size=self.step_size,
                disk_radius=self.disk_radius,
                temperature=self.temperature,
            ).simulate()
            return i_fit

        # fit raw data but exclude first data point, as semi-analytical method skips time=0
        fit_results = curve_fit(
            f=fit_function,
            xdata=self.voltage_to_fit,
            ydata=self.current_to_fit[1:],
            p0=initial_guesses,
            bounds=[lower_bounds, upper_bounds],
        )
        # TODO: return the optimal parameters, transform popt from an array into a dict keyed by fitting param name?
        popt, pcov = list(fit_results)
        current_fit = fit_function(self.voltage_to_fit, *popt)
        sigma = np.sqrt(np.diag(pcov))  # one standard deviation of the parameters

        for val, error, param in zip(popt, sigma, fitting_params):
            print(f"Final fit: '{param}': {val:.2E} +/- {error:.0E}")
        print(f"Ill-conditioned if large: {np.linalg.cond(pcov)}")  # remove

        # Semi-analytical method does not compute the first point (i.e. time=0)
        # so the starting voltage data point with a zero current is reinserted
        self.voltage_to_fit = np.insert(self.voltage_to_fit, 0, self.start_voltage / 1000)
        current_fit = np.insert(current_fit, 0, 0)
        return self.voltage_to_fit, current_fit
