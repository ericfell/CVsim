"""
Module for cyclic voltammogram fitting, using a semi-integration
simulated CV for one- and two-electron processes.

"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit
from .mechanisms import CyclicVoltammetryScheme, E_rev, E_q, E_qC, EE, SquareScheme


class FitMechanism(ABC):
    """
    Scheme for fitting CVs

    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_r: float,
            c_bulk: float,
            step_size: float,
            disk_radius: float,
            temperature: float,
    ) -> None:
        self.current_to_fit = current_to_fit
        self.scan_r = scan_r
        self.c_bulk = c_bulk
        self.step_size = step_size
        self.disk_radius = disk_radius
        self.temperature = temperature
        self.start_voltage = round(voltage_to_fit[0] * 1000)

        if round(voltage_to_fit[10] * 1000) > self.start_voltage:  # scan starts towards more positive
            self.reverse_voltage = round(max(voltage_to_fit) * 1000)
        else:  # scan starts towards more negative
            self.reverse_voltage = round(min(voltage_to_fit) * 1000)

        # make a cleaner x array
        scan_direction = -1 if self.start_voltage < self.reverse_voltage else 1
        delta_theta = scan_direction * self.step_size

        thetas = [round((i - delta_theta)) for i in [self.start_voltage, self.reverse_voltage]]
        forward_scan = np.arange(thetas[0], thetas[1], step=delta_theta * -1)
        reverse_scan = np.append(forward_scan[-2::-1], self.start_voltage)
        self.voltage_to_fit = np.concatenate([forward_scan, reverse_scan]) / 1000

    @abstractmethod
    def fit(self, *args):
        """Fit designated mechanism"""
        return NotImplementedError


class FitE_rev(FitMechanism):
    """
    TODO
    """

    def __init__(
            self,
            voltage_to_fit: list[float] | np.ndarray,
            current_to_fit: list[float] | np.ndarray,
            scan_r: float,
            c_bulk: float,
            step_size: float = 1.0,
            disk_radius: float = 1.5,
            temperature: float = 298.0,
            reduction_potential: float | None = None,
            diffusion_reactant: float | None = None,
            diffusion_product: float | None = None
    ) -> None:
        super().__init__(voltage_to_fit, current_to_fit, scan_r, c_bulk, step_size, disk_radius, temperature)
        self.reduction_potential = reduction_potential
        self.diffusion_reactant = diffusion_reactant
        self.diffusion_product = diffusion_product

        # print({k: v for k, v in locals().items() if k != 'self'})
        # Contains only variables with a user-specified fixed value
        self.fixed_vars = {
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        }
        self.fixed_vars = {k: v for k, v in self.fixed_vars.items() if v is not None}

        # Values are [initial guess, lower bound, upper bound]
        self.default_vars = {
            'reduction_potential': [round((self.voltage_to_fit[np.argmax(self.current_to_fit)]
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
        reduction_potential: None | float | tuple[float, float] | tuple[float, float, float] = None,
        diffusion_reactant: None | float | tuple[float, float] | tuple[float, float, float] = None,
        diffusion_product: None | float | tuple[float, float] | tuple[float, float, float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """TODO  no guess | initial guess | bounds | guess, bounds"""

        fit_vars = {
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        }
        fit_vars = {k: v for k, v in fit_vars.items() if v is not None}

        for key, val in fit_vars.items():
            if isinstance(val, tuple) and len(val) not in [2, 3]:
                raise ValueError(f"'{key}' allowed types: None, float, tuple[float, float], tuple[float, float, float]")

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
                fit_default_vars[param] = value

        for param, (initial, lower, upper) in fit_default_vars.items():
            if not lower < initial < upper:
                # if default initial guess is outside bounds, set guess to avg of bounds
                fit_default_vars[param] = ((lower + upper) / 2, lower, upper)
                # if user's guess was outside bounds
                if initial != self.default_vars[param][0]:
                    raise ValueError(f"Initial guess for '{param}' is outside user-defined bounds")

        print(f"final fitting vars: {fit_default_vars}")
        initial_guesses, lower_bounds, upper_bounds = zip(*fit_default_vars.values())

        print(f'Initial guesses: {initial_guesses}')
        print(f'Lower/Upper bounds: {lower_bounds}/{upper_bounds}')
        print(f'Fixed params: {list(self.fixed_vars)}')
        print(f'Fitting for: {list(fitting_params)}')

        def fit_function(x: list[float] | np.ndarray, *args) -> np.ndarray:
            """Inner function used by scipy curve_fit to fit a CV"""
            print(f"trying values: {args}")

            def fetch(param: str) -> float:
                if param in self.fixed_vars:
                    return self.fixed_vars[param]
                return args[var_index[param]]

            _, i_fit = E_rev(
                start_potential=round(self.start_voltage / 1000, 3),
                switch_potential=round(self.reverse_voltage / 1000, 3),
                reduction_potential=fetch('reduction_potential'),
                scan_rate=self.scan_r,
                c_bulk=self.c_bulk,
                diffusion_reactant=fetch('diffusion_reactant'),
                diffusion_product=fetch('diffusion_product'),
                step_size=self.step_size,
                disk_radius=self.disk_radius,
                temperature=self.temperature,
            ).simulate()
            return i_fit
        print(f"data len: x {len(self.voltage_to_fit)} y {len(self.current_to_fit)}")
        popt, pcov = curve_fit(f=fit_function,
                               xdata=self.voltage_to_fit,
                               ydata=np.insert(self.current_to_fit, 0, 0),#[1:],
                               p0=initial_guesses,
                               bounds=[lower_bounds, upper_bounds])

        current_fit = fit_function(self.voltage_to_fit, *popt)
        sigma = np.sqrt(np.diag(pcov))  # one standard deviation of the parameters

        for val, error, param in zip(popt, sigma, fitting_params):
            print(f"Final fit: '{param}': {val:.2E} +/- {error:.0E}")
        print(f"Ill-conditioned if large: {np.linalg.cond(pcov)}")
        # TODO: return the optimal parameters, transform popt from an array into a dict keyed by fitting param name?
        return self.voltage_to_fit, current_fit
