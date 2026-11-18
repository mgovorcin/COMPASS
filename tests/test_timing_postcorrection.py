import datetime

import numpy as np
import pytest

from compass.s1_timing_postcorrection import table_azimuth_offset
from compass.utils.cslc_postcorrection import (CarrierPolynomial, Poly1d,
                                               TopsCarrier, postcorrect, wrap)

T0 = datetime.datetime(2026, 1, 4, 1, 51, 22)


def _table(**extra_s1a):
    s1a = {'platform': 'S1A', 'validityStart': None, 'validityStop': None,
           'azimuthTimeCorrection': -2.6e-05, 'auxIns': ['INS_A']}
    s1a.update(extra_s1a)
    return {'name': 'TABLE', 'entries': [
        s1a,
        {'platform': 'S1D', 'validityStart': None, 'validityStop': None,
         'azimuthTimeCorrection': 4.66e-05, 'auxIns': ['INS_D']}]}


def test_offset_for_platform():
    assert table_azimuth_offset(_table(), 'S1D', T0, 'IW1') == pytest.approx(4.66e-05)
    assert table_azimuth_offset(_table(), 'S1A', T0, 'IW1') == pytest.approx(-2.6e-05)


def test_reference_platform_is_zero():
    assert table_azimuth_offset(_table(), 'S1C', T0, 'IW1') == 0.0


def test_swath_residual_added():
    table = _table(swathResiduals={'IW2': {'azimuth': 1.0e-06}})
    assert table_azimuth_offset(table, 'S1A', T0, 'iw2') == pytest.approx(-2.5e-05)
    assert table_azimuth_offset(table, 'S1A', T0, 'IW1') == pytest.approx(-2.6e-05)


def test_validity_window():
    table = _table(validityStart='2026-02-01T00:00:00Z')
    with pytest.raises(ValueError):
        table_azimuth_offset(table, 'S1A', T0, 'IW1')


def test_ambiguous_entries_need_aux_ins():
    table = _table()
    table['entries'].append({'platform': 'S1A', 'validityStart': None, 'validityStop': None,
                             'azimuthTimeCorrection': -3.0e-05, 'auxIns': ['INS_A2']})
    with pytest.raises(ValueError):
        table_azimuth_offset(table, 'S1A', T0, 'IW1')
    assert table_azimuth_offset(table, 'S1A', T0, 'IW1', aux_ins='INS_A2') == pytest.approx(-3.0e-05)


def test_shifted_carrier_polynomial():
    doppler = Poly1d([-32.217, -10295.87, 1.351702e7], 800405.23, 149896229.0)
    fm_rate = Poly1d([-2329.59, 450035.94, -7.86031986e7], 799778.58, 149896229.0)
    carrier = TopsCarrier(172800.0, 0.0020555563, 1498, 799778.58, doppler, fm_rate,
                          0.0277571716, 7590.0, 0.05546576)
    dt = 4.66e-05
    shifted = CarrierPolynomial(carrier, 1498, 22084, 2.329562, time_offset=dt)
    tt, rr = np.meshgrid(172800.0 + np.linspace(0.1, 2.9, 40), np.linspace(8.0e5, 8.49e5, 40))
    np.testing.assert_allclose(wrap(shifted(tt, rr) - carrier(tt + dt, rr)), 0.0, atol=0.05)


def test_postcorrect_needs_one_correction(tmp_path):
    with pytest.raises(ValueError):
        postcorrect('in.h5', tmp_path / 'out.h5', 'dem.tif')
    with pytest.raises(ValueError):
        postcorrect('in.h5', tmp_path / 'out.h5', 'dem.tif', carrier_offset=1e-5, resample=True)
