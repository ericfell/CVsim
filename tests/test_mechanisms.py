import pytest
import numpy as np

from cvsim.mechanisms import CyclicVoltammetryScheme, E_rev, E_q, E_qC, EE, SquareScheme


class TestCyclicVoltammetryScheme:

    def test_abstract_class_init(self):
        with pytest.raises(TypeError):
            CyclicVoltammetryScheme()

