#!/usr/bin/env python
'''
Post-correct the Sentinel-1 platform azimuth timing offset of an existing
geocoded CSLC-S1 product from an OPERA_S1_AUX_TIMING table

The correction is phase-only, as the carrier_phase azimuth mode of the CSLC
SAS: the product is multiplied by exp(i * (P1 - P0)), P0 the azimuth carrier
of the burst and P1 that of the burst shifted by the table's azimuth time
correction (plus the swath residual, if the table has one).
'''
import argparse
import datetime
import json

import h5py
import numpy as np

from compass.utils.cslc_postcorrection import PARAMETERS, postcorrect

FLAG = 'azimuth_timing_postcorrection_offset'


def _parse_time(value):
    if value is None:
        return None
    return datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00')).replace(tzinfo=None)


def table_azimuth_offset(table, platform, sensing_time, swath, aux_ins=None):
    '''
    Azimuth time correction [s] for one burst from an OPERA_S1_AUX_TIMING table

    Parameters
    ----------
    table: dict
        Parsed OPERA_S1_AUX_TIMING JSON
    platform: str
        Mission identifier, e.g. 'S1A'
    sensing_time: datetime.datetime
        Burst sensing start (UTC, naive)
    swath: str
        Swath, e.g. 'IW2'
    aux_ins: str or None
        AUX_INS the IPF used; needed only when several entries cover the time

    Returns
    -------
    float
        Azimuth time correction [s]; 0 for a platform the table does not correct
    '''
    entries = [e for e in table['entries'] if e['platform'] == platform]
    if not entries:
        return 0.0
    in_time = []
    for entry in entries:
        start, stop = _parse_time(entry['validityStart']), _parse_time(entry['validityStop'])
        if (start is None or sensing_time >= start) and (stop is None or sensing_time < stop):
            in_time.append(entry)
    if aux_ins is not None:
        in_time = [e for e in in_time if aux_ins in e.get('auxIns', [])]
    if len(in_time) != 1:
        raise ValueError(f'{len(in_time)} {table.get("name", "table")} entries cover '
                         f'{platform} at {sensing_time}' +
                         ('' if aux_ins else '; give the AUX_INS to choose'))
    entry = in_time[0]
    residual = (entry.get('swathResiduals') or {}).get(swath.upper(), {})
    return float(entry['azimuthTimeCorrection']) + float(residual.get('azimuth', 0.0))


def product_burst(h5):
    '''(platform, sensing start, swath) of a CSLC-S1 product'''
    ident = h5['identification']
    platform = ident['mission_id'][()].decode()
    sensing = datetime.datetime.fromisoformat(ident['zero_doppler_start_time'][()].decode())
    swath = ident['burst_id'][()].decode().split('_')[-1].upper()
    return platform, sensing, swath


def get_parser():
    '''Command-line parser'''
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cslc', help='input CSLC-S1 HDF5 product')
    parser.add_argument('dem', help='DEM covering the product (any GDAL format)')
    parser.add_argument('timing_table', help='OPERA_S1_AUX_TIMING JSON')
    parser.add_argument('output', help='output CSLC-S1 HDF5 product')
    parser.add_argument('--aux-ins', default=None,
                        help='AUX_INS the IPF used, when several table entries apply')
    parser.add_argument('--block-rows', type=int, default=512,
                        help='rows processed at a time (default: 512)')
    return parser


def main(argv=None):
    '''Run the azimuth timing post-correction'''
    args = get_parser().parse_args(argv)
    with open(args.timing_table, encoding='utf-8') as fid:
        table = json.load(fid)
    with h5py.File(args.cslc, 'r') as h5:
        params = h5[PARAMETERS]
        if FLAG in params:
            raise ValueError(f'{args.cslc} is already post-corrected')
        mode = params.get('azimuth_timing_correction_mode')
        if mode is not None and mode[()].decode() != 'none':
            raise ValueError(f'{args.cslc} already applied an azimuth timing correction')
        platform, sensing, swath = product_burst(h5)
    offset = table_azimuth_offset(table, platform, sensing, swath, args.aux_ins)
    if offset == 0.0:
        raise ValueError(f'{table.get("name", "the table")} does not correct {platform}')
    postcorrect(args.cslc, args.output, args.dem, carrier_offset=offset,
                block_rows=args.block_rows)
    with h5py.File(args.output, 'r+') as h5:
        ds = h5[PARAMETERS].create_dataset(FLAG, data=offset)
        ds.attrs['units'] = np.bytes_('seconds')
        ds.attrs['description'] = np.bytes_(
            'Phase-only azimuth timing post-correction applied to this product')
        ds.attrs['timing_table'] = np.bytes_(table.get('name', ''))


if __name__ == '__main__':
    main()
