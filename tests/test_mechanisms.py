import pytest
import numpy as np

from cvsim.mechanisms import CyclicVoltammetryScheme, E_rev, E_q, E_qC, EE, SquareScheme


class TestCyclicVoltammetryScheme:

    def test_abstract_class_init(self):
        with pytest.raises(TypeError):
            CyclicVoltammetryScheme()

    def test_potential_scan(self):
        test_v = E_rev(0.5, -0.5, 0, 1, 1, 1e-6, 1e-6, step_size=500)
        v,i = test_v.simulate()
        assert (v == np.array([0.0, -0.5, 0.0, 0.5])).all()

        test_v2 = E_rev(-0.1, 0.4, 0, 1, 1, 1e-6, 1e-6, step_size=100)
        v2, i2 = test_v2.simulate()
        assert (v2 == np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1])).all()

        test_v3 = E_rev(-0.2, -0.05, -0.1, 1, 1, 1e-6, 1e-6, step_size=50)
        v3, i3 = test_v3.simulate()
        assert (v3 == np.array([-0.15, -0.1, -0.05, -0.1, -0.15, -0.2])).all()


class TestE_q:

    def test_init(self):
        with pytest.raises(ValueError):
            test_1 = E_q(-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, alpha=-0.2, k_0=1e-3)

        with pytest.raises(ValueError):
            test_1 = E_q(-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, alpha=1.5, k_0=1e-3)

        with pytest.raises(ValueError):
            test_1 = E_q(-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, alpha=0.2, k_0=-1e-3)

        with pytest.raises(ValueError):
            test_1 = E_q(-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, alpha=0.0, k_0=-1e-3)

        with pytest.raises(ValueError):
            test_1 = E_q(-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, alpha=1.0, k_0=-1e-3)

        with pytest.raises(ValueError):
            test_1 = E_q(-0.5, 0.4, 0, 1, 1, 1e-6, 1e-6, alpha=0.2, k_0=0.0)

