import numpy as np

from compass.utils.lut import azimuth_lut_data


def test_azimuth_lut_data_signs():
    bistatic = np.array([[1.0e-6, 2.0e-6]])
    fm = np.array([[3.0e-5, -4.0e-5]])
    out = azimuth_lut_data(bistatic, fm)
    np.testing.assert_allclose(out, [[2.9e-5, -4.2e-5]])


def test_azimuth_lut_data_fm_term_is_added():
    zero = np.zeros((2, 2))
    fm = np.full((2, 2), 5.0e-5)
    np.testing.assert_allclose(azimuth_lut_data(zero, fm), fm)
