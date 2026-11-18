import h5py
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from compass.s1_fm_rate_postcorrection import fm_rate_time_shift
from compass.utils.cslc_postcorrection import (PARAMETERS, TIMING, CarrierPolynomial,
                                               Poly1d, RegularLUT, TopsCarrier,
                                               interpolate_rows, wrap)


@pytest.fixture
def lut_axes():
    t = 172800.0 + np.arange(109) * 0.028
    r = 799778.6 + np.arange(417) * 120.0
    return t, r


def test_regular_lut_matches_scipy(lut_axes):
    t, r = lut_axes
    rng = np.random.default_rng(0)
    values = rng.normal(size=(t.size, r.size))
    ti = rng.uniform(t[0] - 0.05, t[-1] + 0.05, 5000)
    ri = rng.uniform(r[0] - 300, r[-1] + 300, 5000)
    expected = RegularGridInterpolator((t, r), values, bounds_error=False,
                                       fill_value=None)((ti, ri))
    lut = RegularLUT(t[0], 0.028, r[0], 120.0, values)
    np.testing.assert_allclose(lut(ti, ri), expected, atol=1e-8)


def test_regular_lut_several_arrays(lut_axes):
    t, r = lut_axes
    a = np.ones((t.size, r.size))
    first, second = RegularLUT(t[0], 0.028, r[0], 120.0, a, 2 * a)(t[3], r[5])
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(2.0)


def _sierra_carrier():
    doppler = Poly1d([-32.217, -10295.87, 1.351702e7], 800405.23, 149896229.0)
    fm_rate = Poly1d([-2329.59, 450035.94, -7.86031986e7], 799778.58, 149896229.0)
    return TopsCarrier(sensing_start=172800.0, azimuth_time_interval=0.0020555563,
                       n_lines=1498, starting_range=799778.58, doppler=doppler,
                       azimuth_fm_rate=fm_rate, steering_rate=0.0277571716,
                       velocity=7590.0, wavelength=0.05546576)


def test_carrier_polynomial_follows_the_carrier():
    exact = _sierra_carrier()
    poly = CarrierPolynomial(exact, n_lines=1498, n_samples=22084,
                             range_pixel_spacing=2.329562)
    t = 172800.0 + np.linspace(0.1, 2.9, 50)
    r = np.linspace(800000.0, 849000.0, 50)
    tt, rr = np.meshgrid(t, r)
    np.testing.assert_allclose(wrap(poly(tt, rr) - exact(tt, rr)), 0.0, atol=0.05)


def test_interpolate_rows_exact_for_linear_field():
    rows = np.array([0, 24, 48, 50])
    cols = np.array([0, 24, 40])
    yy, xx = np.meshgrid(rows, cols, indexing='ij')
    out = interpolate_rows(rows, cols, 2.0 * yy + 3.0 * xx, 41, np.arange(10, 30))
    y_out, x_out = np.meshgrid(np.arange(10, 30), np.arange(41), indexing='ij')
    np.testing.assert_allclose(out, 2.0 * y_out + 3.0 * x_out)


def test_wrap_range():
    out = wrap(np.array([-7.0, -np.pi, 0.0, np.pi, 7.0]))
    assert np.all(out >= -np.pi) and np.all(out < np.pi)
    np.testing.assert_allclose(np.exp(1j * out), np.exp(1j * np.array([-7.0, -np.pi, 0.0, np.pi, 7.0])))


@pytest.mark.parametrize('applied, factor', [(True, 2.0), (False, 1.0)])
def test_fm_rate_time_shift_factor(tmp_path, applied, factor):
    path = tmp_path / 'product.h5'
    with h5py.File(path, 'w') as h5:
        h5.create_dataset(f'{PARAMETERS}/azimuth_fm_rate_applied', data=applied)
        h5.create_dataset(f'{TIMING}/zero_doppler_time', data=np.array([0.0, 1.0]))
        h5.create_dataset(f'{TIMING}/zero_doppler_time_spacing', data=1.0)
        h5.create_dataset(f'{TIMING}/slant_range', data=np.array([0.0, 10.0]))
        h5.create_dataset(f'{TIMING}/slant_range_spacing', data=10.0)
        h5.create_dataset(f'{TIMING}/azimuth_fm_rate_mismatch', data=np.full((2, 2), 3.0e-5))
    with h5py.File(path, 'r') as h5:
        shift = fm_rate_time_shift(h5, np.array([0.5]), np.array([5.0]))
    np.testing.assert_allclose(shift, factor * 3.0e-5)
