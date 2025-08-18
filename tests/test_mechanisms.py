import pytest
import numpy as np

from cvsim.mechanisms import CyclicVoltammetryScheme, E_rev, E_q, E_qC, EE, SquareScheme


class TestCyclicVoltammetryScheme:

    def test_abstract_class_init(self):
        with pytest.raises(TypeError):
            CyclicVoltammetryScheme()

    def test_potential_scan(self):
        test_v = E_rev(0.5, -0.5, 0, 1, 1, 1e-6, 1e-6, step_size=500)
        v, i = test_v.simulate()
        assert (v == np.array([0.0, -0.5, 0.0, 0.5])).all()

        test_v2 = E_rev(-0.1, 0.4, 0, 1, 1, 1e-6, 1e-6, step_size=100)
        v2, i2 = test_v2.simulate()
        assert (v2 == np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1])).all()

        test_v3 = E_rev(-0.2, -0.05, -0.1, 1, 1, 1e-6, 1e-6, step_size=50)
        v3, i3 = test_v3.simulate()
        assert (v3 == np.array([-0.15, -0.1, -0.05, -0.1, -0.15, -0.2])).all()


class TestE_rev:

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0, 1, 1e-6, 1e-6, 1, 1, 300),
            (-0.5, 0.4, 0, 1, -0.1, 1e-6, 1e-6, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, -1e-6, 1e-6, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, -1e-6, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, -1.5, 1, 300),
            (-0.5, 0.4, 0, 2, 1, 1e-6, 1e-6, 1, -10, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 1, 1, 0.0),
            (0.4, 0.4, 0, 3, 1, 1e-6, 1e-6, 1, 1, 300),
        ],
    )
    def test_init(
            self,
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
    ):
        with pytest.raises(ValueError):
            _ = E_rev(
                start_potential=start_potential,
                switch_potential=switch_potential,
                reduction_potential=reduction_potential,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                diffusion_reactant=diffusion_reactant,
                diffusion_product=diffusion_product,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 1, 1, 300),
            (0.5, -0.4, 0, 2, 1, 1e-6, 1e-6, 1, 1, 300),
            (-1.0, 0.7, 0, 5, 1, 1e-6, 1e-6, 1, 1, 300),
        ],
    )
    def test_simulate(
            self,
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
    ):
        potential, current = E_rev(
            start_potential=start_potential,
            switch_potential=switch_potential,
            reduction_potential=reduction_potential,
            scan_rate=scan_rate,
            c_bulk=c_bulk,
            diffusion_reactant=diffusion_reactant,
            diffusion_product=diffusion_product,
            step_size=step_size,
            disk_radius=disk_radius,
            temperature=temperature,
        ).simulate()
        assert len(potential) == len(current)


class TestE_q:

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "k0, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, -0.5, 1e-3, 1, 1, 300),
            (-0.5, 0.4, 0, 1, -0.1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, -1e-6, 1e-6, 0.5, 1e-3, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, -1e-6, 0.5, 1e-3, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 1, 1e-3, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 0, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, -1.5, 1, 300),
            (-0.5, 0.4, 0, 2, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, -10, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 0.0),
            (0.4, 0.4, 0, 3, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 300),
        ],
    )
    def test_init(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            alpha,
            k0,
            step_size,
            disk_radius,
            temperature,
    ):
        with pytest.raises(ValueError):
            _ = E_q(
                start_potential=start_potential,
                switch_potential=switch_potential,
                reduction_potential=reduction_potential,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                diffusion_reactant=diffusion_reactant,
                diffusion_product=diffusion_product,
                alpha=alpha,
                k0=k0,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "k0, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 300),
            (0.5, -0.4, 0, 2, 1, 1e-6, 1e-6, 0.5, 2e-3, 1, 1, 300),
            (-1.0, 0.7, 0, 5, 1, 1e-6, 1e-6, 0.6, 1e-3, 1, 1, 300),
        ],
    )
    def test_simulate(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            alpha,
            k0,
            step_size,
            disk_radius,
            temperature,
    ):
        potential, current = E_q(
            start_potential=start_potential,
            switch_potential=switch_potential,
            reduction_potential=reduction_potential,
            scan_rate=scan_rate,
            c_bulk=c_bulk,
            diffusion_reactant=diffusion_reactant,
            diffusion_product=diffusion_product,
            alpha=alpha,
            k0=k0,
            step_size=step_size,
            disk_radius=disk_radius,
            temperature=temperature,
        ).simulate()
        assert len(potential) == len(current)


class TestE_qC:

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "k0, "
        "k_forward, "
        "k_backward, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, -0.5, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, -0.1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, -1e-6, 1e-6, 0.5, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, -1e-6, 0.5, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 1, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 0, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, -1.5, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 2, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, -10, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 1, 1, 0.0),
            (0.4, 0.4, 0, 3, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 0, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, -2, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 0, 1, 1, 300),
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, -10, 1, 1, 300),
        ],
    )
    def test_init(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            alpha,
            k0,
            k_forward,
            k_backward,
            step_size,
            disk_radius,
            temperature,
    ):
        with pytest.raises(ValueError):
            _ = E_qC(
                start_potential=start_potential,
                switch_potential=switch_potential,
                reduction_potential=reduction_potential,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                diffusion_reactant=diffusion_reactant,
                diffusion_product=diffusion_product,
                alpha=alpha,
                k0=k0,
                k_forward=k_forward,
                k_backward=k_backward,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "k0, "
        "k_forward, "
        "k_backward, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, 0.5, 1e-3, 1, 1, 1, 1, 300),
            (0.5, -0.4, 0, 2, 1, 1e-6, 1e-6, 0.5, 2e-3, 1, 1, 1, 1, 300),
            (-1.0, 0.7, 0, 5, 1, 1e-6, 1e-6, 0.6, 1e-3, 1, 1, 1, 1, 300),
        ],
    )
    def test_simulate(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_product,
            alpha,
            k0,
            k_forward,
            k_backward,
            step_size,
            disk_radius,
            temperature,
    ):
        potential, current = E_qC(
            start_potential=start_potential,
            switch_potential=switch_potential,
            reduction_potential=reduction_potential,
            scan_rate=scan_rate,
            c_bulk=c_bulk,
            diffusion_reactant=diffusion_reactant,
            diffusion_product=diffusion_product,
            alpha=alpha,
            k0=k0,
            k_forward=k_forward,
            k_backward=k_backward,
            step_size=step_size,
            disk_radius=disk_radius,
            temperature=temperature,
        ).simulate()
        assert len(potential) == len(current)


class TestEE:

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "reduction_potential2, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_intermediate, "
        "diffusion_product, "
        "alpha, "
        "alpha2, "
        "k0, "
        "k0_2, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0.1, 0, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, -0.1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, -0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, -1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 0, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, -1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 1, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 0, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, -1.5, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 0, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 2, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, -10, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 0.0),
            (0.4, 0.4, 0, 0.1, 3, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, -2, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 0, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, -0.5, 300),
        ],
    )
    def test_init(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            reduction_potential2,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_intermediate,
            diffusion_product,
            alpha,
            alpha2,
            k0,
            k0_2,
            step_size,
            disk_radius,
            temperature,
    ):
        with pytest.raises(ValueError):
            _ = EE(
                start_potential=start_potential,
                switch_potential=switch_potential,
                reduction_potential=reduction_potential,
                reduction_potential2=reduction_potential2,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
                diffusion_reactant=diffusion_reactant,
                diffusion_intermediate=diffusion_intermediate,
                diffusion_product=diffusion_product,
                alpha=alpha,
                alpha2=alpha2,
                k0=k0,
                k0_2=k0_2,
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "reduction_potential2, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_intermediate, "
        "diffusion_product, "
        "alpha, "
        "alpha2, "
        "k0, "
        "k0_2, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1e-3, 1, 1, 300),
            (0.5, -0.4, 0, 0.1, 2, 1, 1e-6, 1e-6, 1e-6, 0.5, 0.5, 2e-3, 1e-3, 1, 1, 300),
            (-1.0, 0.7, 0, 0.1, 5, 1, 1e-6, 1e-6, 1e-6, 0.6, 0.5, 1e-3, 2e-3, 1, 1, 300),
        ],
    )
    def test_simulate(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            reduction_potential2,
            scan_rate,
            c_bulk,
            diffusion_reactant,
            diffusion_intermediate,
            diffusion_product,
            alpha,
            alpha2,
            k0,
            k0_2,
            step_size,
            disk_radius,
            temperature,
    ):
        potential, current = EE(
            start_potential=start_potential,
            switch_potential=switch_potential,
            reduction_potential=reduction_potential,
            reduction_potential2=reduction_potential2,
            scan_rate=scan_rate,
            c_bulk=c_bulk,
            diffusion_reactant=diffusion_reactant,
            diffusion_intermediate=diffusion_intermediate,
            diffusion_product=diffusion_product,
            alpha=alpha,
            alpha2=alpha2,
            k0=k0,
            k0_2=k0_2,
            step_size=step_size,
            disk_radius=disk_radius,
            temperature=temperature,
        ).simulate()
        assert len(potential) == len(current)


class TestSquareScheme:

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "reduction_potential2, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "alpha2, "
        "k0, "
        "k0_2, "
        "k_forward, "
        "k_backward, "
        "k_forward2, "
        "k_backward2, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0.1, 0, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, -0.1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, -0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, -1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 0, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, -1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 1, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 0, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, -1.5, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 0, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 2, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, -10, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 0.0),
            (0.4, 0.4, 0, 0.1, 3, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, -2, 1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 0, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 1, 1, -0.5, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, -1, 1, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, -1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 0, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, 0, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, -2, 1, 1, 1, 1, 300),
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1, 1, 1, 1, -3, 1, 1, 300),
        ],
    )
    def test_init(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            reduction_potential2,
            scan_rate,
            c_bulk,
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
            step_size,
            disk_radius,
            temperature,
    ):
        with pytest.raises(ValueError):
            _ = SquareScheme(
                start_potential=start_potential,
                switch_potential=switch_potential,
                reduction_potential=reduction_potential,
                reduction_potential2=reduction_potential2,
                scan_rate=scan_rate,
                c_bulk=c_bulk,
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
                step_size=step_size,
                disk_radius=disk_radius,
                temperature=temperature,
            )

    @pytest.mark.parametrize(
        "start_potential, "
        "switch_potential, "
        "reduction_potential, "
        "reduction_potential2, "
        "scan_rate, "
        "c_bulk, "
        "diffusion_reactant, "
        "diffusion_product, "
        "alpha, "
        "alpha2, "
        "k0, "
        "k0_2, "
        "k_forward, "
        "k_backward, "
        "k_forward2, "
        "k_backward2, "
        "step_size, "
        "disk_radius, "
        "temperature, ",
        [
            (-0.5, 0.4, 0, 0.1, 1, 1, 1e-6, 1e-6, 0.5, 0.5, 1e-3, 1e-3, 1, 1, 1, 1, 1, 1, 300),
            (0.5, -0.4, 0, 0.1, 2, 1, 1e-6, 1e-6, 0.5, 0.5, 2e-3, 1e-3, 1, 1, 1, 1, 1, 1, 300),
            (-1.0, 0.7, 0, 0.1, 5, 1, 1e-6, 1e-6, 0.6, 0.5, 1e-3, 2e-3, 1, 1, 1, 1, 1, 1, 300),
        ],
    )
    def test_simulate(
            self,
            start_potential,
            switch_potential,
            reduction_potential,
            reduction_potential2,
            scan_rate,
            c_bulk,
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
            step_size,
            disk_radius,
            temperature,
    ):
        potential, current = SquareScheme(
            start_potential=start_potential,
            switch_potential=switch_potential,
            reduction_potential=reduction_potential,
            reduction_potential2=reduction_potential2,
            scan_rate=scan_rate,
            c_bulk=c_bulk,
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
            step_size=step_size,
            disk_radius=disk_radius,
            temperature=temperature,
        ).simulate()
        assert len(potential) == len(current)
