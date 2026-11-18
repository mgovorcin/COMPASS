#!/usr/bin/env python
'''
Post-correct the azimuth FM rate mismatch of an existing geocoded CSLC-S1
product to the sign used by COMPASS from this version on

Products that applied the FM rate mismatch (azimuth_fm_rate_applied = True)
sampled the radar data dt_FMM before the corrected time and move by
2 * dt_FMM; products that did not apply it move by dt_FMM.
'''
import argparse

import h5py
import numpy as np

from compass.utils.cslc_postcorrection import (PARAMETERS, RegularLUT,
                                               postcorrect)

FLAG = 'azimuth_fm_rate_postcorrection'


def fm_rate_time_shift(h5, t, r):
    '''
    Azimuth time shift [s] that brings a product's FM rate mismatch
    correction to the current sign

    Parameters
    ----------
    h5: h5py.File
        Open CSLC-S1 product
    t, r: np.ndarray
        Zero-Doppler azimuth time and slant range

    Returns
    -------
    np.ndarray
        Time shift with the shape of t
    '''
    applied = bool(h5[f'{PARAMETERS}/azimuth_fm_rate_applied'][()])
    dt_fmm = RegularLUT.from_product(h5, 'azimuth_fm_rate_mismatch')(t, r)
    return (2.0 if applied else 1.0) * dt_fmm


def get_parser():
    '''Command-line parser'''
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cslc', help='input CSLC-S1 HDF5 product')
    parser.add_argument('dem', help='DEM covering the product (any GDAL format)')
    parser.add_argument('output', help='output CSLC-S1 HDF5 product')
    parser.add_argument('--resample', action='store_true',
                        help='also move the image (default: phase screen only)')
    parser.add_argument('--block-rows', type=int, default=512,
                        help='rows processed at a time (default: 512)')
    return parser


def main(argv=None):
    '''Run the FM rate mismatch post-correction'''
    args = get_parser().parse_args(argv)
    with h5py.File(args.cslc, 'r') as h5:
        if FLAG in h5[PARAMETERS]:
            raise ValueError(f'{args.cslc} is already post-corrected')
    postcorrect(args.cslc, args.output, args.dem, fm_rate_time_shift,
                resample=args.resample, block_rows=args.block_rows)
    with h5py.File(args.output, 'r+') as h5:
        h5[PARAMETERS].create_dataset(
            FLAG, data=np.bytes_('resample' if args.resample else 'phase_screen'))
        h5[f'{PARAMETERS}/{FLAG}'].attrs['description'] = np.bytes_(
            'FM rate mismatch sign post-correction applied to this product')


if __name__ == '__main__':
    main()
