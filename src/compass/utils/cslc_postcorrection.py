'''
Azimuth time post-correction of geocoded CSLC-S1 products

A geocoded CSLC holds, at each pixel, the radar sample at azimuth time
t_s = t_zd + az_lut(t_zd, r), re-ramped with the TOPS azimuth carrier
evaluated at t_s. Changing the azimuth LUT by dt moves the sample to
t_s + dt. That change is applied here to an existing product, from its own
contents and a DEM:

- phase screen: multiply by exp(i * (phi(t_s + dt) - phi(t_s))), with phi
  the azimuth carrier COMPASS evaluated; the image stays in place
- resample: also move the image so that each pixel holds the sample at
  t_s + dt, at the same slant range

The carrier polynomial is rebuilt from the burst metadata the same way
s1-reader's get_az_carrier_poly builds it.
'''
import datetime
import shutil

import h5py
import isce3
import numpy as np
import pyproj
from osgeo import gdal, osr
from scipy.ndimage import map_coordinates

BURST_META = 'metadata/processing_information/input_burst_metadata'
TIMING = 'metadata/processing_information/timing_corrections'
PARAMETERS = 'metadata/processing_information/parameters'


def orbit_from_product(h5):
    '''
    isce3 orbit and its reference epoch from a CSLC-S1 product

    Parameters
    ----------
    h5: h5py.File
        Open CSLC-S1 product

    Returns
    -------
    orbit: isce3.core.Orbit
        Orbit of the product
    epoch: datetime.datetime
        Orbit reference epoch
    '''
    grp = h5['metadata/orbit']
    epoch = datetime.datetime.fromisoformat(grp['reference_epoch'][()].decode())
    ref = isce3.core.DateTime(epoch)
    pos = np.stack([grp[f'position_{c}'][()] for c in 'xyz'], 1)
    vel = np.stack([grp[f'velocity_{c}'][()] for c in 'xyz'], 1)
    state_vectors = [isce3.core.StateVector(ref + isce3.core.TimeDelta(t), p, v)
                     for t, p, v in zip(grp['time'][()], pos, vel)]
    return isce3.core.Orbit(state_vectors, ref), epoch


class Poly1d:
    '''Polynomial in (x - mean) / std, as stored in CSLC-S1 burst metadata'''

    def __init__(self, coeffs, mean, std):
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def from_group(cls, grp):
        '''Polynomial from a product group with coeffs, mean and std'''
        return cls(grp['coeffs'][()], grp['mean'][()], grp['std'][()])

    def __call__(self, x):
        '''Evaluate at x'''
        x = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        return np.polynomial.polynomial.polyval(x, self.coeffs)


class TopsCarrier:
    '''
    TOPS azimuth carrier of a burst, as s1-reader's az_carrier_components

    Time arguments are seconds from the same epoch as sensing_start.
    '''

    def __init__(self, sensing_start, azimuth_time_interval, n_lines,
                 starting_range, doppler, azimuth_fm_rate, steering_rate,
                 velocity, wavelength):
        self.dta = float(azimuth_time_interval)
        self.sensing_start = float(sensing_start)
        self.t_eta0 = self.sensing_start + (int(n_lines) // 2) * self.dta
        self.r0 = float(starting_range)
        self.doppler = doppler
        self.fm_rate = azimuth_fm_rate
        self.ks = 2.0 * float(velocity) * float(steering_rate) / float(wavelength)

    def __call__(self, t, r):
        '''Carrier phase [rad] at azimuth time t and slant range r'''
        fdc = self.doppler(r)
        ka = self.fm_rate(r)
        eta_ref = self.doppler(self.r0) / self.fm_rate(self.r0) - fdc / ka
        kt = self.ks / (1.0 - self.ks / ka)
        eta = np.asarray(t) - self.t_eta0 - self.dta
        return np.pi * kt * (eta - eta_ref) ** 2


class CarrierPolynomial:
    '''
    Azimuth carrier as COMPASS geocodes it: the 2-D polynomial fit of
    s1-reader's get_az_carrier_poly (azimuth order 5, range order 3,
    every 50th line and 500th sample)
    '''

    AZ_ORDER, RG_ORDER, X_STEP, Y_STEP = 5, 3, 500, 50

    def __init__(self, carrier, n_lines, n_samples, range_pixel_spacing):
        x = np.arange(0, int(n_samples), self.X_STEP)
        y = np.arange(0, int(n_lines), self.Y_STEP)
        x_mesh, y_mesh = np.meshgrid(x, y)
        values = carrier(carrier.sensing_start + (y_mesh + 1) * carrier.dta,
                         carrier.r0 + x_mesh * range_pixel_spacing)
        rg = (carrier.r0 + (x_mesh + 1) * range_pixel_spacing).ravel()
        az = (carrier.sensing_start + (y_mesh + 1) * carrier.dta).ravel()
        self.x_min, self.x_norm = rg.min(), rg.max() - rg.min()
        self.y_min, self.y_norm = az.min(), az.max() - az.min()
        x_n = (rg - self.x_min) / self.x_norm
        y_n = (az - self.y_min) / self.y_norm
        big = max(self.AZ_ORDER, self.RG_ORDER)
        terms = [(i, j) for i in range(self.AZ_ORDER + 1)
                 for j in range(self.RG_ORDER + 1) if i + j <= big]
        design = np.stack([y_n ** i * x_n ** j for i, j in terms], 1)
        solution = np.linalg.lstsq(design, values.ravel(), rcond=1.0e-12)[0]
        self.coeffs = np.zeros((self.AZ_ORDER + 1, self.RG_ORDER + 1))
        for (i, j), value in zip(terms, solution):
            self.coeffs[i, j] = value

    def __call__(self, t, r):
        '''Carrier phase [rad] at azimuth time t and slant range r'''
        return np.polynomial.polynomial.polyval2d(
            (np.asarray(t) - self.y_min) / self.y_norm,
            (np.asarray(r) - self.x_min) / self.x_norm, self.coeffs)


def carrier_from_product(h5, orbit, epoch):
    '''
    Azimuth carrier polynomial of a CSLC-S1 product

    Parameters
    ----------
    h5: h5py.File
        Open CSLC-S1 product
    orbit: isce3.core.Orbit
        Orbit of the product
    epoch: datetime.datetime
        Orbit reference epoch; carrier times are seconds from it

    Returns
    -------
    CarrierPolynomial
        Carrier as evaluated by COMPASS
    '''
    meta = h5[BURST_META]
    n_lines, n_samples = (int(v) for v in meta['shape'][()])
    dta = float(meta['azimuth_time_interval'][()])
    start = (datetime.datetime.fromisoformat(meta['sensing_start'][()].decode())
             - epoch).total_seconds()
    _, vel = orbit.interpolate(start + 0.5 * n_lines * dta)
    carrier = TopsCarrier(start, dta, n_lines, meta['starting_range'][()],
                          Poly1d.from_group(meta['doppler']),
                          Poly1d.from_group(meta['azimuth_fm_rate']),
                          meta['azimuth_steering_rate'][()],
                          np.linalg.norm(vel), meta['wavelength'][()])
    return CarrierPolynomial(carrier, n_lines, n_samples,
                             float(meta['range_pixel_spacing'][()]))


class RegularLUT:
    '''
    Bilinear lookup of one or more arrays on a regular (azimuth, range)
    grid, extrapolating linearly outside it
    '''

    def __init__(self, t_start, t_spacing, r_start, r_spacing, *arrays):
        self.t0, self.dt = float(t_start), float(t_spacing)
        self.r0, self.dr = float(r_start), float(r_spacing)
        self.arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
        self.shape = self.arrays[0].shape

    @classmethod
    def from_product(cls, h5, *names):
        '''LUTs of the timing_corrections group, on start + i * spacing'''
        grp = h5[TIMING]
        return cls(grp['zero_doppler_time'][0], grp['zero_doppler_time_spacing'][()],
                   grp['slant_range'][0], grp['slant_range_spacing'][()],
                   *[grp[n][()] for n in names])

    def __call__(self, t, r):
        '''Values of each array at (t, r); one array if one was given'''
        ft = (np.asarray(t, dtype=np.float64) - self.t0) / self.dt
        fr = (np.asarray(r, dtype=np.float64) - self.r0) / self.dr
        i = np.clip(np.floor(ft).astype(np.int64), 0, self.shape[0] - 2)
        j = np.clip(np.floor(fr).astype(np.int64), 0, self.shape[1] - 2)
        wt, wr = ft - i, fr - j
        k = i * self.shape[1] + j
        n = self.shape[1]
        out = []
        for arr in self.arrays:
            flat = arr.ravel()
            out.append((1 - wt) * ((1 - wr) * flat[k] + wr * flat[k + 1])
                       + wt * ((1 - wr) * flat[k + n] + wr * flat[k + n + 1]))
        return out if len(out) > 1 else out[0]


def dem_on_product_grid(h5, dem_path):
    '''
    DEM heights resampled (bilinear) onto the geocoded grid of a product

    Parameters
    ----------
    h5: h5py.File
        Open CSLC-S1 product
    dem_path: str
        Any GDAL-readable DEM

    Returns
    -------
    np.ndarray
        Heights [m] with the shape of the product's data layers
    '''
    osr.SetPROJSearchPaths([pyproj.datadir.get_data_dir()])
    x = h5['data/x_coordinates'][()]
    y = h5['data/y_coordinates'][()]
    dx, dy = x[1] - x[0], y[1] - y[0]
    epsg = int(h5['data/projection'][()])
    warped = gdal.Warp('', str(dem_path), format='MEM', dstSRS=f'EPSG:{epsg}',
                       resampleAlg='bilinear',
                       outputBounds=(x[0] - dx / 2, y[-1] + dy / 2,
                                     x[-1] + dx / 2, y[0] - dy / 2),
                       width=len(x), height=len(y))
    return warped.ReadAsArray().astype(np.float32)


def coarse_radar_coordinates(h5, orbit, heights, step=24, dh=100.0):
    '''
    Zero-Doppler azimuth time and slant range on a coarse grid of the
    product, and their derivatives with height

    Parameters
    ----------
    h5: h5py.File
        Open CSLC-S1 product
    orbit: isce3.core.Orbit
        Orbit of the product
    heights: np.ndarray
        Heights on the product grid
    step: int
        Coarse grid spacing in pixels
    dh: float
        Height increment for the derivatives [m]

    Returns
    -------
    dict
        rows, cols: coarse grid indices; t, r, h: azimuth time, slant range
        and height there; dt_dh, dr_dh: derivatives with height
    '''
    x = h5['data/x_coordinates'][()]
    y = h5['data/y_coordinates'][()]
    epsg = int(h5['data/projection'][()])
    wavelength = float(h5[f'{BURST_META}/wavelength'][()])
    rows = np.r_[np.arange(0, len(y), step), len(y) - 1]
    cols = np.r_[np.arange(0, len(x), step), len(x) - 1]
    xx, yy = np.meshgrid(x[cols], y[rows])
    lon, lat = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True).transform(xx, yy)
    hgt = np.nan_to_num(heights[np.ix_(rows, cols)].astype(np.float64))
    ellipsoid = isce3.core.Ellipsoid()
    doppler = isce3.core.LUT2d()
    side = isce3.core.LookSide.Right
    t, r, t_h, r_h = (np.empty(xx.shape) for _ in range(4))
    for k in np.ndindex(xx.shape):
        llh = np.array([np.deg2rad(lon[k]), np.deg2rad(lat[k]), hgt[k]])
        t[k], r[k] = isce3.geometry.geo2rdr(llh, ellipsoid, orbit, doppler,
                                            wavelength, side)
        llh[2] += dh
        t_h[k], r_h[k] = isce3.geometry.geo2rdr(llh, ellipsoid, orbit, doppler,
                                                wavelength, side)
    return {'rows': rows, 'cols': cols, 't': t, 'r': r, 'h': hgt,
            'dt_dh': (t_h - t) / dh, 'dr_dh': (r_h - r) / dh}


def interpolate_rows(rows, cols, values, n_cols, out_rows):
    '''
    Separable bilinear interpolation of a coarse-grid field onto full rows

    Parameters
    ----------
    rows, cols: np.ndarray
        Coarse grid indices
    values: np.ndarray
        Field on the coarse grid
    n_cols: int
        Number of columns of the full grid
    out_rows: np.ndarray
        Full-grid rows to return

    Returns
    -------
    np.ndarray
        Field on out_rows x all columns
    '''
    along_x = np.stack([np.interp(np.arange(n_cols), cols, line) for line in values])
    j = np.clip(np.searchsorted(rows, out_rows, side='right') - 1, 0, len(rows) - 2)
    w = ((out_rows - rows[j]) / (rows[j + 1] - rows[j]))[:, None]
    return (1 - w) * along_x[j] + w * along_x[j + 1]


def wrap(phase):
    '''Phase wrapped to [-pi, pi)'''
    return np.remainder(phase + np.pi, 2 * np.pi) - np.pi


def _sampled_time(h5):
    '''Function of (t_zd, r) giving the azimuth time a product sampled'''
    params = h5[PARAMETERS]
    names = []
    if bool(params['bistatic_delay_applied'][()]):
        names.append('bistatic_delay')
    if bool(params['azimuth_fm_rate_applied'][()]):
        names.append('azimuth_fm_rate_mismatch')
    if not names:
        return lambda t, r: t
    lut = RegularLUT.from_product(h5, *names)

    def sampled(t, r):
        values = lut(t, r)
        values = values if isinstance(values, list) else [values]
        return t - sum(values)
    return sampled


def postcorrect(cslc_in, cslc_out, dem_path, time_shift, resample=False,
                block_rows=512, coarse_step=24, polarization=None):
    '''
    Move the azimuth sample time of a geocoded CSLC-S1 product by a
    per-pixel time shift, writing a corrected copy

    Parameters
    ----------
    cslc_in: str
        Input CSLC-S1 HDF5 product
    cslc_out: str
        Output path; a copy of cslc_in with the data layer and
        azimuth_carrier_phase replaced
    dem_path: str
        DEM for the pixel heights
    time_shift: callable
        time_shift(h5, t_zd, r) -> azimuth time shift [s] at zero-Doppler
        time t_zd and slant range r
    resample: bool
        If True, also move the image (resample); else phase screen only
    block_rows: int
        Rows processed at a time
    coarse_step: int
        Coarse geo2rdr grid spacing in pixels
    polarization: str or None
        Data layer to correct; the first polarization layer if None
    '''
    with h5py.File(cslc_in, 'r') as h5:
        orbit, epoch = orbit_from_product(h5)
        carrier = carrier_from_product(h5, orbit, epoch)
        sampled = _sampled_time(h5)
        pol = polarization or next(k for k in h5['data']
                                   if k in ('VV', 'HH', 'VH', 'HV'))
        heights = dem_on_product_grid(h5, dem_path)
        coarse = coarse_radar_coordinates(h5, orbit, heights, step=coarse_step)
        z = h5[f'data/{pol}'][()]
        carr = h5['data/azimuth_carrier_phase'][()]
        flat = h5['data/flattening_phase'][()]

        rows, cols = coarse['rows'], coarse['cols']
        t_c, r_c = coarse['t'], coarse['r']
        ts_c = sampled(t_c, r_c)
        if resample:
            shift_c = time_shift(h5, t_c, r_c)
            gt_y, gt_x = np.gradient(ts_c, rows, cols)
            gr_y, gr_x = np.gradient(r_c, rows, cols)
            det = gt_y * gr_x - gt_x * gr_y
            d_row_c = shift_c * gr_x / det
            d_col_c = -shift_c * gr_y / det

        n_rows, n_cols = z.shape
        z_out = np.empty_like(z)
        carr_out = np.empty_like(carr)
        for r0 in range(0, n_rows, block_rows):
            out_rows = np.arange(r0, min(r0 + block_rows, n_rows))

            def fine(values):
                return interpolate_rows(rows, cols, values, n_cols, out_rows)
            dh = heights[out_rows] - fine(coarse['h'])
            r_p = fine(r_c) + fine(coarse['dr_dh']) * dh
            t_p = fine(t_c) + fine(coarse['dt_dh']) * dh
            ts_p = sampled(t_p, r_p)
            phi_new = carrier(ts_p + time_shift(h5, t_p, r_p), r_p)
            dphi = wrap(phi_new - carrier(ts_p, r_p)).astype(np.float32)
            carr_out[out_rows] = wrap(carr[out_rows] + dphi)
            valid = np.isfinite(z[out_rows])
            if not resample:
                z_out[out_rows] = z[out_rows] * np.exp(1j * dphi).astype(np.complex64)
                continue
            lo, hi = max(0, r0 - 16), min(n_rows, out_rows[-1] + 17)
            base = z[lo:hi] * np.exp(-1j * (carr[lo:hi] + flat[lo:hi]))
            base = np.where(np.isfinite(base), base, 0)
            yy, xx = np.meshgrid(out_rows - lo, np.arange(n_cols), indexing='ij')
            coords = [yy + fine(d_row_c), xx + fine(d_col_c)]
            moved = (map_coordinates(base.real, coords, order=3, mode='nearest')
                     + 1j * map_coordinates(base.imag, coords, order=3, mode='nearest'))
            z_out[out_rows] = np.where(valid, moved * np.exp(1j * (phi_new + flat[out_rows])),
                                       np.nan).astype(z.dtype)

    shutil.copyfile(cslc_in, cslc_out)
    with h5py.File(cslc_out, 'r+') as h5:
        h5[f'data/{pol}'][...] = z_out
        h5['data/azimuth_carrier_phase'][...] = carr_out
