import numpy as np

from compass.utils.h5_helpers import lut_axis


def test_lut_axis_nodes():
    np.testing.assert_allclose(lut_axis(800000.0, 120.0, 4),
                               [800000.0, 800120.0, 800240.0, 800360.0])


def test_lut_axis_keeps_spacing():
    axis = lut_axis(172800.0, 0.028, 109)
    np.testing.assert_allclose(np.diff(axis), 0.028)
    np.testing.assert_allclose(axis[-1], 172800.0 + 108 * 0.028)
