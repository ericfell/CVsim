"""
Module for cyclic voltammogram fitting, using a semi-integration
simulated CV for one- and two-electron processes.

"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit
from mechanisms import CyclicVoltammetryScheme, E_rev, E_q, E_qC, EE, SquareScheme


class HelperClass(ABC):
    """
    Scheme for fitting CVs

    """

    def __init__(self,
                 voltage_to_fit: list | np.ndarray,
                 current_to_fit: list | np.ndarray,
                 scan_r: float,
                 c_bulk: float,
                 step_size: float,
                 disk_radius: float,
                 temperature: float,
                 ) -> None:
        # self.voltage_to_fit = voltage_to_fit
        self.current_to_fit = current_to_fit
        self.scan_r = scan_r
        self.c_bulk = c_bulk
        self.step_size = step_size
        self.disk_radius = disk_radius
        self.temperature = temperature
        self.start_voltage = int(round(voltage_to_fit[0] * 1000))

        if int(round(voltage_to_fit[30] * 1000)) > self.start_voltage:  # scan starts towards more positive
            self.reverse_voltage = int(round(max(voltage_to_fit) * 1000))
        else:  # scan starts towards more negative
            self.reverse_voltage = int(round(min(voltage_to_fit) * 1000))

        # make a cleaner x array
        scan_direction = -1 if self.start_voltage < self.reverse_voltage else 1
        delta_theta = scan_direction * self.step_size

        thetas = [int(round((i - delta_theta))) for i in [self.start_voltage, self.reverse_voltage]]
        forward_scan = np.arange(thetas[0], thetas[1], step=delta_theta * -1)
        reverse_scan = np.append(forward_scan[-2::-1], self.start_voltage)
        self.voltage_to_fit = np.concatenate([forward_scan, reverse_scan]) / 1000
        # TODO is below needed for real (not simulated) raw data?
        #self.voltage_to_fit = potential[1:]  # semi-analytical method doesnt use initial potential

    @abstractmethod
    def fit(self, *args):
        """Fit designated mechanism"""
        return NotImplementedError


class FitE_rev(HelperClass):
    """
    TODO
    """

    def __init__(self,
                 voltage_to_fit,
                 current_to_fit,
                 scan_r,
                 c_bulk,
                 step_size: float = 1.0,
                 disk_radius: float = 1.5,
                 temperature: float = 298.0,
                 reduction_potential: float = None,
                 diffusion_reactant: float = None,
                 diffusion_product: float = None
                 ) -> None:
        super().__init__(voltage_to_fit, current_to_fit, scan_r, c_bulk, step_size, disk_radius, temperature)
        self.reduction_potential = reduction_potential
        self.diffusion_reactant = diffusion_reactant
        self.diffusion_product = diffusion_product

        # Contains only variables with a user-specified fixed value
        self.fixed_vars = {
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        }
        self.fixed_vars = {k: v for k, v in self.fixed_vars.items() if v is not None}

        # Values are (initial guess, lower bound, upper bound)
        self.default_vars = {
            'reduction_potential': (round((self.voltage_to_fit[np.argmax(self.current_to_fit)]
                                           + self.voltage_to_fit[np.argmin(self.current_to_fit)]) / 2, 3),
                                    round(min(self.start_voltage, self.reverse_voltage)/1000, 3),
                                    round(max(self.start_voltage, self.reverse_voltage)/1000, 3)),
            'diffusion_reactant': (1e-6, 5e-8, 1e-4),
            'diffusion_product': (1e-6, 5e-8, 1e-4),
        }

    # new: no guess | initial guess | bounds | guess, bounds
    def fit(self,
            reduction_potential: None | float | tuple[float, float] | tuple[float, float, float] = None,
            diffusion_reactant: None | float | tuple[float, float] | tuple[float, float, float] = None,
            diffusion_product: None | float | tuple[float, float] | tuple[float, float, float] = None,
            ) -> tuple[np.ndarray, np.ndarray]:

        fit_vars = {
            'reduction_potential': reduction_potential,
            'diffusion_reactant': diffusion_reactant,
            'diffusion_product': diffusion_product,
        }
        fit_vars = {k: v for k, v in fit_vars.items() if v is not None}

        # check intersection of big_dict / fit dict. if so print error
        intersection_errors = (self.fixed_vars.keys() & fit_vars.keys())
        if intersection_errors:
            raise ValueError("Cannot input fixed value and guess value for same param")

        # compute set of variables to fit
        fitting_vars = [
            param for param in self.default_vars.keys()
            if param not in self.fixed_vars.keys()
        ]

        # trim dict to set of fit variables
        fit_default_vars = {k: v for k, v in self.default_vars.items() if k in fitting_vars}
        var_index = {var: index for index, var in enumerate(fit_default_vars.keys())}

        # take fit_dict, for each in it, replace the initial guess/bounds if specified
        for param, value in fit_vars.items():
            if isinstance(value, float | int):
                # Initial guess
                fit_vars[param][0] = value
            elif isinstance(value, tuple) and len(value) == 2:
                # Lower and upper bound
                fit_vars[param][1] = value[0]
                fit_vars[param][2] = value[1]
            elif isinstance(value, tuple) and len(value) == 3:
                fit_vars[param] = value

        # turn into array
        initial_guesses, lower_bounds, upper_bounds = zip(*fit_vars.values())


        # params = {k: v for k, v in locals().items() if k != 'self'}


        print(f'Initial guesses: {initial_guesses}')
        print(f'Lower bounds: {lower_bounds}')
        print(f'Upper bounds: {upper_bounds}')
        print(f'Fixed params: {list(self.fixed_vars)}')
        print(f'Fitting for: {list(fitting_vars)}')

        def fit_function(x, *args):  # curve_fit calls this
            """TODO"""
            print(f"fitting for {args} args")

            def fetch(param: str):
                if param in self.fixed_vars:
                    return self.fixed_vars[param]
                return args[var_index[param]]

            v_fit, i_fit = E_rev(
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

        popt, pcov = curve_fit(f=fit_function,
                               xdata=self.voltage_to_fit,
                               ydata=self.current_to_fit[1:],
                               p0=initial_guesses,
                               bounds=[lower_bounds, upper_bounds])

        current_fit = fit_function(self.voltage_to_fit, *popt)
        sigma = np.sqrt(np.diag(pcov))  # one standard deviation of the parameters
        # print fits and std dev. for a,b,c etc.
        # TODO print out param name associated with value
        print([f"{val:.2E} +/- {error:.0E}" for val, error in zip(popt, sigma)])
        print(f"Ill-conditioned if large: {np.linalg.cond(pcov)}")
        return self.voltage_to_fit, current_fit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scan = 0.1
    step = 1
    bulk = 1
    disk = 1.5
    temp = 298
    raw_x, raw_y = E_rev(0.3, -0.5, -0.1, scan, 1, 1e-6, 2e-6, step_size=step, disk_radius=disk,
                         temperature=temp).simulate()

    raw_x = np.insert(raw_x, 0, 0.3)
    raw_y = np.insert(raw_y, 0, 0.0)
    fitted_voltage, fitted_current = FitE_rev(raw_x,
                                              raw_y,
                                              scan_r=scan,
                                              c_bulk=bulk,
                                              step_size=step,
                                              disk_radius=disk,
                                              temperature=temp,
                                              reduction_potential=-0.09,
                                              ).fit(
                                                                    diffusion_reactant=(1.1e-6, 9e-7, 2.2e-6),
                                                                    diffusion_product=(2e-6, 1e-6, 2.2e-6))

    fig, ax = plt.subplots()
    ax.plot(raw_x, raw_y, 'r-')
    ax.plot(fitted_voltage, fitted_current, 'k--')
    plt.show()
