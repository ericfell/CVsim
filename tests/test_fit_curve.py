import pytest
import numpy as np

from cvsim.fit_curve import FitMechanism, FitE_rev, FitE_q
from cvsim.mechanisms import E_rev, E_q


dummy_voltages, dummy_currents = E_rev(0.3, -0.5, -0.1, 0.1, 1, 1e-6, 2e-6).simulate()
dummy_voltages = np.insert(dummy_voltages, 0, 0.3)
dummy_currents = np.insert(dummy_currents, 0, 0.0)

dummy_voltages2, dummy_currents2 = E_q(-0.4, 0.6, 0.05, 0.1, 1, 1e-6, 2e-6, 0.5, 1e-5).simulate()
dummy_voltages2 = np.insert(dummy_voltages2, 0, -0.4)
dummy_currents2 = np.insert(dummy_currents2, 0, 0.0)


class TestFitMechanism:

    def test_abstract_class_init(self):
        with pytest.raises(TypeError):
            FitMechanism()


class TestFitE_rev:

    @pytest.mark.parametrize(
        "voltage_to_fit, "
        "current_to_fit, "
        "scan_rate, "
        "c_bulk, "
        "step_size, "
        "disk_radius, "
        "temperature, "
        "reduction_potential, "
        "diffusion_reactant, "
        "diffusion_product, ",
        [
            (dummy_voltages, dummy_currents, -0.1, 1, 1, 1, 300, 0.1, 1e-6, 1e-6),
            (dummy_voltages, dummy_currents, 0.1, 0, 1, 1, 300, 0.1, 1e-6, 1e-6),
            (dummy_voltages, dummy_currents, 0.1, 1, -1, 1, 300, 0.1, 1e-6, 1e-6),
            (dummy_voltages, dummy_currents[:10], 0.1, 1, 1, 1, 300, 0.12, 1e-6, 1e-6),
            (dummy_voltages, dummy_currents, 0.1, 1, 1, 1, 0, 0.1, 1e-6, 1e-6),
            (dummy_voltages, dummy_currents, 0.1, 1, 1, -2, 300, 0.12, 2e-6, 2e-6),
            (dummy_voltages, dummy_currents, 0.1, 1, 1, 2, 300, 0.12, -2e-6, 2e-6),
            (dummy_voltages, dummy_currents, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 0.0)
        ],
    )
    def test_init(
            self,
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
    ):
        with pytest.raises(ValueError):
            FitE_rev(
                voltage_to_fit=voltage_to_fit,
                current_to_fit=current_to_fit,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
                reduction_potential=reduction_potential,
                diffusion_reactant=diffusion_reactant,
                diffusion_product=diffusion_product,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "d_r, "
        "d_p, "
        "fit_red_pot, "
        "fit_d_r, "
        "fit_d_p, ",
        [
            (0.1, None, None, 0.12, (1e-6, 3e-6), 3.3e-6),
            (None, None, None, None, (6e-6, 4e-6), 1.1e-6),
            (-0.02, None, None, None, (1.1e-6, 5e-6), (6e-6, 4e-6)),
            (-0.02, 3e-6, 5e-6, None, (2e-6, 1.1e-6, 5e-6), 1.1e-6),
            (None, None, None, 0.0, (0.0, 1.1e-6, 5e-6), 1.1e-6),
            (None, None, None, None, (2e-6, 4e-6), (2.1e-6, 5e-6, 1e-6)),
        ],
    )
    def test_fit(self, red_pot, d_r, d_p, fit_red_pot, fit_d_r, fit_d_p):
        with pytest.raises(ValueError):
            v, i = FitE_rev(
                voltage_to_fit=dummy_voltages,
                current_to_fit=dummy_currents,
                scan_rate=0.1,
                c_bulk=1,
                step_size=1,
                disk_radius=1.5,
                temperature=298,
                reduction_potential=red_pot,
                diffusion_reactant=d_r,
                diffusion_product=d_p,
            ).fit(
                reduction_potential=fit_red_pot,
                diffusion_reactant=fit_d_r,
                diffusion_product=fit_d_p,
            )


class TestFitE_q:

    @pytest.mark.parametrize(
        "voltage_to_fit, "
        "current_to_fit, "
        "scan_rate, "
        "c_bulk, "
        "step_size, "
        "disk_radius, "
        "temperature, "
        "reduction_potential, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "k_0, ",
        [
            (dummy_voltages2, dummy_currents2, -0.1, 1, 1, 1, 300, 0.1, 1e-6, 1e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 0, 1, 1, 300, 0.1, 1e-6, 1e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, -1, 1, 300, 0.1, 1e-6, 1e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2[:10], 0.1, 1, 1, 1, 300, 0.12, 1e-6, 1e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 1, 0, 0.1, 1e-6, 1e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, -2, 300, 0.12, 2e-6, 2e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 2, 300, 0.12, -2e-6, 2e-6, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 0.0, 0.5, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.0, 1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.5, -1e-4),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.5, 0.0),
            (dummy_voltages2, dummy_currents2, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, -0.5, 1e-4),
        ],
    )
    def test_init(
            self,
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
            alpha,
            k_0,
    ):
        with pytest.raises(ValueError):
            FitE_q(
                voltage_to_fit=voltage_to_fit,
                current_to_fit=current_to_fit,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
                reduction_potential=reduction_potential,
                diffusion_reactant=diffusion_reactant,
                diffusion_product=diffusion_product,
                alpha=alpha,
                k_0=k_0,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "d_r, "
        "d_p, "
        "alph, "
        "k, "
        "fit_red_pot, "
        "fit_d_r, "
        "fit_d_p, "
        "fit_alph, "
        "fit_k, ",
        [
            (0.1, None, None, None, None, 0.12, (1e-6, 3e-6), 3.3e-6, 0.4, 1e-5),
            (None, None, None, None, None, None, (6e-6, 4e-6), 1.1e-6, 0.4, 1e-5),
            (-0.02, None, None, None, None, None, (1.1e-6, 5e-6), (6e-6, 4e-6), 0.4, 1e-5),
            (-0.02, 3e-6, 5e-6, None, None, None, (2e-6, 1.1e-6, 5e-6), 1.1e-6, 0.4, 1e-5),
            (None, None, None, None, None, 0.0, (0.0, 1.1e-6, 5e-6), 1.1e-6, 0.4, 1e-5),
            (None, None, None, None, None, None, (2e-6, 4e-6), (2.1e-6, 5e-6, 1e-6), 0.4, 1e-5),
            (0.1, 1e-6, 1e-6, 0.5, None, None, None, None, 0.4, 1e-5),
            (0.1, 1e-6, 1e-6, None, 1e-4, None, None, None, 0.4, 1e-5),
            (0.1, 1e-6, 1e-6, None, None, None, None, None, 0.4, (0.0, 1.1e-6, 5e-6)),
            (0.1, 1e-6, 1e-6, None, None, None, None, None, (0.6, 0.2), 1e-5),
        ],
    )
    def test_fit(self, red_pot, d_r, d_p, alph, k, fit_red_pot, fit_d_r, fit_d_p, fit_alph, fit_k):
        with pytest.raises(ValueError):
            v, i = FitE_q(
                voltage_to_fit=dummy_voltages2,
                current_to_fit=dummy_currents2,
                scan_rate=0.1,
                c_bulk=1,
                step_size=1,
                disk_radius=1.5,
                temperature=298,
                reduction_potential=red_pot,
                diffusion_reactant=d_r,
                diffusion_product=d_p,
                alpha=alph,
                k_0=k,
            ).fit(
                reduction_potential=fit_red_pot,
                diffusion_reactant=fit_d_r,
                diffusion_product=fit_d_p,
                alpha=fit_alph,
                k_0=fit_k,
            )
