import pytest
import numpy as np

from cvsim.fit_curve import FitMechanism, FitE_rev, FitE_q, FitE_qC, FitEE, FitSquareScheme
from cvsim.mechanisms import E_rev, E_q, E_qC, EE, SquareScheme


dummy_voltages, dummy_currents = E_rev(0.3, -0.5, -0.1, 0.1, 1, 1e-6, 2e-6).simulate()
dummy_voltages = np.insert(dummy_voltages, 0, 0.3)
dummy_currents = np.insert(dummy_currents, 0, 0.0)

dummy_voltages2, dummy_currents2 = E_q(-0.4, 0.6, 0.05, 0.1, 1, 1e-6, 2e-6, 0.5, 1e-5).simulate()
dummy_voltages2 = np.insert(dummy_voltages2, 0, -0.4)
dummy_currents2 = np.insert(dummy_currents2, 0, 0.0)

dummy_voltages3, dummy_currents3 = E_qC(0.4, -0.6, 0.05, 0.1, 1, 1e-6, 2e-6, 0.5, 1e-4, 1e-3, 1e-4).simulate()
dummy_voltages3 = np.insert(dummy_voltages3, 0, 0.4)
dummy_currents3 = np.insert(dummy_currents3, 0, 0.0)

dummy_voltages4, dummy_currents4 = EE(-0.6, 0.6, -0.05, 0.1, 0.1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-5, 1e-4).simulate()
dummy_voltages4 = np.insert(dummy_voltages4, 0, -0.6)
dummy_currents4 = np.insert(dummy_currents4, 0, 0.0)

dummy_voltages5, dummy_currents5 = SquareScheme(-0.5, 0.6, 0.05, 0.15, 0.1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 2e-3, 1e-1, 2e-1, 1e-3, 3e-3).simulate()
dummy_voltages5 = np.insert(dummy_voltages5, 0, -0.5)
dummy_currents5 = np.insert(dummy_currents5, 0, 0.0)


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
            (dummy_voltages, dummy_currents, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 0.0),
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
            (None, None, None, 0.0, (9e-7, 1.1e-6, 5e-6), 1.1e-6),
            (None, None, None, 0.0, (1.1e-6, 5e-6), (1.1e-6, 5e-6, 4e-6)),
            (None, None, None, [0.1, 0.2], (1.1e-6, 5e-6), (1.1e-6, 5e-7, 4e-6)),
            (None, None, None, (0.0, -0.1, 0.2), (1.1e-6, 5e-6), (5e-6, 4e-6)),
            (0.1, None, None, None, 2e-9, 3.3e-6),
        ],
    )
    def test_fit_params(
            self,
            red_pot,
            d_r,
            d_p,
            fit_red_pot,
            fit_d_r,
            fit_d_p,
    ):
        with pytest.raises(ValueError):
            FitE_rev(
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

    @pytest.mark.parametrize(
        "red_pot, "
        "d_r, "
        "d_p, "
        "fit_red_pot, "
        "fit_d_r, "
        "fit_d_p, ",
        [
            (0.1, None, None, None, (1e-6, 3e-6), 3.3e-6),
            (0.1, None, None, None, (1e-6, 3e-6), 3.3e-7),
        ],
    )
    def test_fitting(
            self,
            red_pot,
            d_r,
            d_p,
            fit_red_pot,
            fit_d_r,
            fit_d_p,
    ):
        v, *_ = FitE_rev(
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
        assert v[0] == dummy_voltages[0]


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
        "k0, ",
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
            k0,
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
                k0=k0,
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
            (0.1, 1e-6, None, None, None, None, None, 1e-9, 0.5, 1e-5),
        ],
    )
    def test_fit_params(
            self,
            red_pot,
            d_r,
            d_p,
            alph,
            k,
            fit_red_pot,
            fit_d_r,
            fit_d_p,
            fit_alph,
            fit_k,
    ):
        with pytest.raises(ValueError):
            FitE_q(
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
                k0=k,
            ).fit(
                reduction_potential=fit_red_pot,
                diffusion_reactant=fit_d_r,
                diffusion_product=fit_d_p,
                alpha=fit_alph,
                k0=fit_k,
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
            (0.1, None, None, None, None, None, (1e-6, 3e-6), 3.3e-6, 0.4, 1e-5),
            (0.1, 1e-6, 1e-6, None, None, None, None, None, 0.4, 1e-5),
        ],
    )
    def test_fitting(
            self,
            red_pot,
            d_r,
            d_p,
            alph,
            k,
            fit_red_pot,
            fit_d_r,
            fit_d_p,
            fit_alph,
            fit_k,
    ):
        v, *_ = FitE_q(
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
            k0=k,
        ).fit(
            reduction_potential=fit_red_pot,
            diffusion_reactant=fit_d_r,
            diffusion_product=fit_d_p,
            alpha=fit_alph,
            k0=fit_k,
        )
        assert v[0] == dummy_voltages2[0]


class TestFitE_qC:

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
        "k0, "
        "k_forward, "
        "k_backward, ",
        [
            (dummy_voltages3, dummy_currents3, -0.1, 1, 1, 1, 300, 0.1, 1e-6, 1e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 0, 1, 1, 300, 0.1, 1e-6, 1e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, -1, 1, 300, 0.1, 1e-6, 1e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3[:10], 0.1, 1, 1, 1, 300, 0.12, 1e-6, 1e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 1, 0, 0.1, 1e-6, 1e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, -2, 300, 0.12, 2e-6, 2e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, -2e-6, 2e-6, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 0.0, 0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.0, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.5, -1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.5, 0.0, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, -0.5, 1e-4, 1e-3, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.5, 1e-4, 0.0, 1e-3),
            (dummy_voltages3, dummy_currents3, 0.1, 1, 1, 2, 300, 0.12, 2e-6, 2e-6, 0.5, 1e-4, 1e-3, -1e-3),
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
            k0,
            k_forward,
            k_backward,
    ):
        with pytest.raises(ValueError):
            FitE_qC(
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
                k0=k0,
                k_forward=k_forward,
                k_backward=k_backward,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "d_r, "
        "d_p, "
        "alph, "
        "k, "
        "k_f, "
        "k_b, "
        "fit_red_pot, "
        "fit_d_r, "
        "fit_d_p, "
        "fit_alph, "
        "fit_k, "
        "fit_kf, "
        "fit_kb, ",
        [
            (0.1, None, None, None, None, None, None, 0.12, (1e-6, 3e-6), 3.3e-6, 0.4, 1e-5, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, (6e-6, 4e-6), 1.1e-6, 0.4, 1e-5, 1e-3, 1e-3),
            (-0.02, None, None, None, None, None, None, None, (1.1e-6, 5e-6), (6e-6, 4e-6), 0.4, 1e-5, 1e-3, 1e-3),
            (-0.02, 3e-6, 5e-6, None, None, None, None, None, (2e-6, 1.1e-6, 5e-6), 1.1e-6, 0.4, 1e-5, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, 0.0, (0.0, 1.1e-6, 5e-6), 1.1e-6, 0.4, 1e-5, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, (2e-6, 4e-6), (2.1e-6, 5e-6, 1e-6), 0.4, 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, 1e-6, 0.5, None, None, None, None, None, None, 0.4, 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, 1e-6, None, 1e-4, None, None, None, None, None, 0.4, 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, 1e-6, None, None, None, None, None, None, None, 0.4, (0.0, 1.1e-6, 5e-6), 1e-3, 1e-3),
            (0.1, 1e-6, 1e-6, None, None, None, None, None, None, None, (0.6, 0.2), 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, None, None, None, None, None, None, None, 1e-9, 0.5, 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, None, None, None, 2e-3, None, None, None, 1e-9, 0.5, 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, None, None, None, None, 3e-3, None, None, 1e-9, 0.5, 1e-5, 1e-3, 1e-3),
        ],
    )
    def test_fit_params(
            self,
            red_pot,
            d_r,
            d_p,
            alph,
            k,
            k_f,
            k_b,
            fit_red_pot,
            fit_d_r,
            fit_d_p,
            fit_alph,
            fit_k,
            fit_kf,
            fit_kb,
    ):
        with pytest.raises(ValueError):
            FitE_qC(
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
                k0=k,
                k_forward=k_f,
                k_backward=k_b,
            ).fit(
                reduction_potential=fit_red_pot,
                diffusion_reactant=fit_d_r,
                diffusion_product=fit_d_p,
                alpha=fit_alph,
                k0=fit_k,
                k_forward=fit_kf,
                k_backward=fit_kb,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "d_r, "
        "d_p, "
        "alph, "
        "k, "
        "k_f, "
        "k_b, "
        "fit_red_pot, "
        "fit_d_r, "
        "fit_d_p, "
        "fit_alph, "
        "fit_k, "
        "fit_kf, "
        "fit_kb, ",
        [
            (0.1, None, None, None, None, None, None, None, (1e-6, 3e-6), 3.3e-6, 0.4, 1e-5, 1e-3, 1e-3),
            (0.1, 1e-6, 1e-6, None, None, None, None, None, None, None, 0.4, 1e-5, 1e-3, 1e-3),
        ],
    )
    def test_fitting(
            self,
            red_pot,
            d_r,
            d_p,
            alph,
            k,
            k_f,
            k_b,
            fit_red_pot,
            fit_d_r,
            fit_d_p,
            fit_alph,
            fit_k,
            fit_kf,
            fit_kb,
    ):
        v, *_ = FitE_qC(
            voltage_to_fit=dummy_voltages3,
            current_to_fit=dummy_currents3,
            scan_rate=0.1,
            c_bulk=1,
            step_size=1,
            disk_radius=1.5,
            temperature=298,
            reduction_potential=red_pot,
            diffusion_reactant=d_r,
            diffusion_product=d_p,
            alpha=alph,
            k0=k,
            k_forward=k_f,
            k_backward=k_b,
        ).fit(
            reduction_potential=fit_red_pot,
            diffusion_reactant=fit_d_r,
            diffusion_product=fit_d_p,
            alpha=fit_alph,
            k0=fit_k,
            k_forward=fit_kf,
            k_backward=fit_kb,
        )
        assert v[0] == dummy_voltages3[0]

class TestFitEE:

    @pytest.mark.parametrize(
        "voltage_to_fit, "
        "current_to_fit, "
        "scan_rate, "
        "c_bulk, "
        "step_size, "
        "disk_radius, "
        "temperature, "
        "reduction_potential, "
        "reduction_potential2, "
        "diffusion_reactant, "
        "diffusion_intermediate, "
        "diffusion_product, "
        "alpha, "
        "alpha2, "
        "k0, "
        "k0_2, ",
        [
            (dummy_voltages4, dummy_currents4, -0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4[4:], 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 0, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, -5, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 0.0, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, -12, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, -1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, -8e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.0, 0.5, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 1.2, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0, 0.5, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, -10, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 2, 1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, -1e-4, 1e-5),
            (dummy_voltages4, dummy_currents4, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 0.0),
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
            reduction_potential2,
            diffusion_reactant,
            diffusion_intermediate,
            diffusion_product,
            alpha,
            alpha2,
            k0,
            k0_2,
    ):
        with pytest.raises(ValueError):
            FitEE(
                voltage_to_fit=voltage_to_fit,
                current_to_fit=current_to_fit,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
                reduction_potential=reduction_potential,
                reduction_potential2=reduction_potential2,
                diffusion_reactant=diffusion_reactant,
                diffusion_intermediate=diffusion_intermediate,
                diffusion_product=diffusion_product,
                alpha=alpha,
                alpha2=alpha2,
                k0=k0,
                k0_2=k0_2,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "red_pot2, "
        "d_r, "
        "d_i, "
        "d_p, "
        "alph, "
        "alph2, "
        "k, "
        "k2, "
        "fit_red_pot, "
        "fit_red_pot2, "
        "fit_d_r, "
        "fit_d_i, "
        "fit_d_p, "
        "fit_alph, "
        "fit_alph2, "
        "fit_k, "
        "fit_k2, ",
        [
            (0.3, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, 0.0, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, 3e-6, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, 4e-6, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, None, 2e-6, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, None, None, 0.2, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, None, None, None, 0.7, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, None, None, None, None, 4e-4, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, None, None, None, None, None, 3e-4, 0.1, 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4),
            (None, None, None, None, None, None, None, None, None, (0.3, 0.1), 0.2, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-4,
             1e-4),
            (None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, (2e-7, 1.1e-6, 5e-6), 1e-6, 0.5, 0.5,
             1e-4, 1e-4),
            (None, None, None, None, None, None, None, None, None, -0.3, 0.2, (0.0, 1e-6, 4e-6), 1e-6, 1e-6, 0.5, 0.5,
             1e-4, 1e-4),
            (None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 1e-6, (0.5, 0.6, 0.9), 0.5,
             1e-4, 1e-4),
            (None, None, None, None, None, None, None, None, 3e-4, 0.1, 0.2, 1e-6, 1e-10, 1e-6, 0.5, 0.5, 1e-4, None),
        ],
    )
    def test_fit_params(
            self,
            red_pot,
            red_pot2,
            d_r,
            d_i,
            d_p,
            alph,
            alph2,
            k,
            k2,
            fit_red_pot,
            fit_red_pot2,
            fit_d_r,
            fit_d_i,
            fit_d_p,
            fit_alph,
            fit_alph2,
            fit_k,
            fit_k2,
    ):
        with pytest.raises(ValueError):
            FitEE(
                voltage_to_fit=dummy_voltages4,
                current_to_fit=dummy_currents4,
                scan_rate=0.1,
                c_bulk=1,
                step_size=1,
                disk_radius=1.5,
                temperature=298,
                reduction_potential=red_pot,
                reduction_potential2=red_pot2,
                diffusion_reactant=d_r,
                diffusion_intermediate=d_i,
                diffusion_product=d_p,
                alpha=alph,
                alpha2=alph2,
                k0=k,
                k0_2=k2,
            ).fit(
                reduction_potential=fit_red_pot,
                reduction_potential2=fit_red_pot2,
                diffusion_reactant=fit_d_r,
                diffusion_intermediate=fit_d_i,
                diffusion_product=fit_d_p,
                alpha=fit_alph,
                alpha2=fit_alph2,
                k0=fit_k,
                k0_2=fit_k2,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "red_pot2, "
        "d_r, "
        "d_i, "
        "d_p, "
        "alph, "
        "alph2, "
        "k, "
        "k2, "
        "fit_red_pot, "
        "fit_red_pot2, "
        "fit_d_r, "
        "fit_d_i, "
        "fit_d_p, "
        "fit_alph, "
        "fit_alph2, "
        "fit_k, "
        "fit_k2, ",
        [
            (-0.05, 0.1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, None, None, None, None, None, None, None, None, None, 1e-4, 1e-4),
            (-0.05, 0.1, 1e-6, 1e-6, 1e-6, None, 0.5, None, None, None, None, None, None, None, (0.3, 0.6), None, 1e-4,
             1e-4),
            (-0.05, 0.1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, None, None, None, None, None, None, None, None, None, (1e-4, 5e-4),
             1e-4),
            (-0.05, 0.1, 1e-6, 1e-6, None, 0.5, 0.5, None, None, None, None, None, None, (1e-6, 5e-7, 2e-6), None, None,
             1e-4, 1e-4),
        ],
    )
    def test_fitting(
            self,
            red_pot,
            red_pot2,
            d_r,
            d_i,
            d_p,
            alph,
            alph2,
            k,
            k2,
            fit_red_pot,
            fit_red_pot2,
            fit_d_r,
            fit_d_i,
            fit_d_p,
            fit_alph,
            fit_alph2,
            fit_k,
            fit_k2,
    ):
        v, *_ = FitEE(
            voltage_to_fit=dummy_voltages4,
            current_to_fit=dummy_currents4,
            scan_rate=0.1,
            c_bulk=1,
            step_size=1,
            disk_radius=1.5,
            temperature=298,
            reduction_potential=red_pot,
            reduction_potential2=red_pot2,
            diffusion_reactant=d_r,
            diffusion_intermediate=d_i,
            diffusion_product=d_p,
            alpha=alph,
            alpha2=alph2,
            k0=k,
            k0_2=k2,
        ).fit(
            reduction_potential=fit_red_pot,
            reduction_potential2=fit_red_pot2,
            diffusion_reactant=fit_d_r,
            diffusion_intermediate=fit_d_i,
            diffusion_product=fit_d_p,
            alpha=fit_alph,
            alpha2=fit_alph2,
            k0=fit_k,
            k0_2=fit_k2,
        )
        assert v[0] == dummy_voltages4[0]


class TestFitSquareScheme:
    @pytest.mark.parametrize(
        "voltage_to_fit, "
        "current_to_fit, "
        "scan_rate, "
        "c_bulk, "
        "step_size, "
        "disk_radius, "
        "temperature, "
        "reduction_potential, "
        "reduction_potential2, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "alpha2, "
        "k0, "
        "k0_2, "
        "k_forward, "
        "k_backward, "
        "k_forward2, "
        "k_backward2, ",
        [
            (dummy_voltages5, dummy_currents5, -0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5[4:], 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 0, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, -5, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 0.0, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, -12, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, -1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, -8e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 0.0, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 1.2, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, -10, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 2, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, -1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 0.0, 1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, -1e-3, 1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, -1e-3, 1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, -1e-3, 1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, -1e-3),
            (dummy_voltages5, dummy_currents5, 0.1, 1, 1, 1, 300, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-5, 1e-3, 1e-3, 1e-3, 0.0),
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
            reduction_potential2,
            diffusion_reactant,
            diffusion_product,
            alpha,
            alpha2,
            k0,
            k0_2,
            k_forward,
            k_backward,
            k_forward2,
            k_backward2,
    ):
        with pytest.raises(ValueError):
            FitSquareScheme(
                voltage_to_fit=voltage_to_fit,
                current_to_fit=current_to_fit,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
                reduction_potential=reduction_potential,
                reduction_potential2=reduction_potential2,
                diffusion_reactant=diffusion_reactant,
                diffusion_product=diffusion_product,
                alpha=alpha,
                alpha2=alpha2,
                k0=k0,
                k0_2=k0_2,
                k_forward=k_forward,
                k_backward=k_backward,
                k_forward2=k_forward2,
                k_backward2=k_backward2,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "red_pot2, "
        "d_r, "
        "d_p, "
        "alph, "
        "alph2, "
        "k, "
        "k2, "
        "kf1, "
        "kb1, "
        "kf2, "
        "kb2, "
        "fit_red_pot, "
        "fit_red_pot2, "
        "fit_d_r, "
        "fit_d_p, "
        "fit_alph, "
        "fit_alph2, "
        "fit_k, "
        "fit_k2, "
        "fit_kf1, "
        "fit_kb1, "
        "fit_kf2, "
        "fit_kb2, ",
        [
            (0.1, None, None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, 0.0, None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, 3e-6, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, 4e-6, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, 0.2, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, 0.7, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, 4e-4, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, 3e-4, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, 1e-4, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, 1e-4, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, 1e-4, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, 1e-4, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, None, (0.3, 0.1), 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, (2e-7, 1.1e-6, 5e-6), 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, (0.0, 1e-6, 4e-6), 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, (0.5, 0.6, 0.9), 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, None, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, (1e-3, 3e-2, 5e-2), 1e-3, 1e-3),
            (None, None, None, None, None, None, None, None, None, None, None, 1e-4, 0.1, 0.2, 1e-6, 1e-6, 0.5, 0.5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, (4e-4, 8e-4, 1e-1)),
        ],
    )
    def test_fit_params(
            self,
            red_pot,
            red_pot2,
            d_r,
            d_p,
            alph,
            alph2,
            k,
            k2,
            kf1,
            kb1,
            kf2,
            kb2,
            fit_red_pot,
            fit_red_pot2,
            fit_d_r,
            fit_d_p,
            fit_alph,
            fit_alph2,
            fit_k,
            fit_k2,
            fit_kf1,
            fit_kb1,
            fit_kf2,
            fit_kb2,
    ):
        with pytest.raises(ValueError):
            FitSquareScheme(
                voltage_to_fit=dummy_voltages5,
                current_to_fit=dummy_currents5,
                scan_rate=0.1,
                c_bulk=1,
                step_size=1,
                disk_radius=1.5,
                temperature=298,
                reduction_potential=red_pot,
                reduction_potential2=red_pot2,
                diffusion_reactant=d_r,
                diffusion_product=d_p,
                alpha=alph,
                alpha2=alph2,
                k0=k,
                k0_2=k2,
                k_forward=kf1,
                k_backward=kb1,
                k_forward2=kf2,
                k_backward2=kb2,
            ).fit(
                reduction_potential=fit_red_pot,
                reduction_potential2=fit_red_pot2,
                diffusion_reactant=fit_d_r,
                diffusion_product=fit_d_p,
                alpha=fit_alph,
                alpha2=fit_alph2,
                k0=fit_k,
                k0_2=fit_k2,
                k_forward=fit_kf1,
                k_backward=fit_kb1,
                k_forward2=fit_kf2,
                k_backward2=fit_kb2,
            )

    @pytest.mark.parametrize(
        "red_pot, "
        "red_pot2, "
        "d_r, "
        "d_p, "
        "alph, "
        "alph2, "
        "k, "
        "k2, "
        "kf1, "
        "kb1, "
        "kf2, "
        "kb2, "
        "fit_red_pot, "
        "fit_red_pot2, "
        "fit_d_r, "
        "fit_d_p, "
        "fit_alph, "
        "fit_alph2, "
        "fit_k, "
        "fit_k2, "
        "fit_kf1, "
        "fit_kb1, "
        "fit_kf2, "
        "fit_kb2, ",
        [
            (0.05, 0.15, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 2e-3, 1e-1, 2e-1, None, None, None, None, None, None, None, None, None, None, None, None, 1e-3, 1e-3),
            (0.05, 0.15, 1e-6, 1e-6, 0.5, None, 1e-3, 2e-3, 1e-1, 2e-1, None, None, None, None, None, None, None, (0.3, 0.6), None, None, None, None, 1e-3, 1e-3),
        ],
    )
    def test_fitting(
            self,
            red_pot,
            red_pot2,
            d_r,
            d_p,
            alph,
            alph2,
            k,
            k2,
            kf1,
            kb1,
            kf2,
            kb2,
            fit_red_pot,
            fit_red_pot2,
            fit_d_r,
            fit_d_p,
            fit_alph,
            fit_alph2,
            fit_k,
            fit_k2,
            fit_kf1,
            fit_kb1,
            fit_kf2,
            fit_kb2,
    ):
        v, *_ = FitSquareScheme(
            voltage_to_fit=dummy_voltages5,
            current_to_fit=dummy_currents5,
            scan_rate=0.1,
            c_bulk=1,
            step_size=1,
            disk_radius=1.5,
            temperature=298,
            reduction_potential=red_pot,
            reduction_potential2=red_pot2,
            diffusion_reactant=d_r,
            diffusion_product=d_p,
            alpha=alph,
            alpha2=alph2,
            k0=k,
            k0_2=k2,
            k_forward=kf1,
            k_backward=kb1,
            k_forward2=kf2,
            k_backward2=kb2,
        ).fit(
            reduction_potential=fit_red_pot,
            reduction_potential2=fit_red_pot2,
            diffusion_reactant=fit_d_r,
            diffusion_product=fit_d_p,
            alpha=fit_alph,
            alpha2=fit_alph2,
            k0=fit_k,
            k0_2=fit_k2,
            k_forward=fit_kf1,
            k_backward=fit_kb1,
            k_forward2=fit_kf2,
            k_backward2=fit_kb2,
        )
        assert v[0] == dummy_voltages5[0]

