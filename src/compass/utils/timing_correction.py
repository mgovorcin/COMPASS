'''
Relative Sentinel-1 instrument timing table (OPERA_S1_AUX_TIMING)

ESA publishes each platform's timing errors in two auxiliary products: the
AUX_INS azimuthTimeBias / swstBias and the AUX_ITC azimuthCalibration /
rangeCalibration. The IPF does not apply them to the azimuth carrier, so
burst-overlap phase between platforms carries their difference.

This module turns those products into a small JSON table of corrections
*relative to one reference platform*, so that new products from one platform
(e.g. S1D) can be aligned with an archive that is left as it is (e.g. S1C).
With T_X the summed timing error of platform X, the correction is

    c_X = -(T_X - T_ref)

in seconds, added to the burst's azimuth (or two-way range) time. Platforms
not listed in the table are not corrected. Each entry also names the AUX_INS
generations it was derived for; a burst focused with any other generation is
"uncovered", which is how a new ESA generation, or an IPF that starts applying
the bias itself, is caught instead of being corrected twice.

The SAS only reads tables (``load_timing_table``, ``TimingTable.lookup``);
``build_timing_table`` and ``main`` are offline tools that make them.
'''
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from dataclasses import dataclass, field

PRODUCT_TYPE = 'OPERA_S1_AUX_TIMING'
CONVENTION = ('seconds; added to the burst azimuth time (range: two-way range '
              'time); correction = -(T_platform - T_reference)')

# Components that may be summed into the range correction. swstBias is left
# out by default: it has been nonzero since S1A, so the IPF almost certainly
# applies it at focusing and adding it again would double-count.
RANGE_COMPONENTS = ('rangeCalibration', 'swstBias')
DEFAULT_RANGE_COMPONENTS = ('rangeCalibration',)

# IW channels whose values must agree before one instrument-wide number is used
_IW_CHANNELS = [(swath, pol) for swath in ('iw1', 'iw2', 'iw3')
                for pol in ('vv', 'vh', 'hh', 'hv')]

_AUX_NAME = re.compile(
    r'^(?P<platform>S1[A-Z])_AUX_(?P<type>INS|ITC)_'
    r'V(?P<validity>\d{8}T\d{6})_G(?P<generation>\d{8}T\d{6})$')


class UncoveredBurstError(ValueError):
    '''A burst whose platform the table corrects, but not for this burst'''


@dataclass(frozen=True)
class TimingEntry:
    '''
    Correction for one platform over one validity window

    Attributes
    ----------
    platform: str
        Mission identifier, e.g. 'S1D'
    validity_start, validity_stop: datetime.datetime or None
        Sensing-time window the entry applies to (UTC); None leaves that end
        open
    azimuth_time_correction: float
        Correction added to the azimuth time [s]
    range_time_correction: float
        Correction added to the two-way range time [s]
    aux_ins: tuple of str
        AUX_INS product names (no .SAFE) the entry was derived for
    aux_itc: str or None
        AUX_ITC product name the entry was derived from
    components: dict
        The raw aux values used, for provenance
    range_components: tuple of str
        Which of RANGE_COMPONENTS were summed into the range correction
    '''
    platform: str
    validity_start: datetime.datetime | None
    validity_stop: datetime.datetime | None
    azimuth_time_correction: float
    range_time_correction: float
    aux_ins: tuple
    aux_itc: str | None
    components: dict = field(default_factory=dict, compare=False)
    range_components: tuple = DEFAULT_RANGE_COMPONENTS

    def covers_time(self, sensing_time):
        '''True if sensing_time falls in [validity_start, validity_stop)'''
        if self.validity_start is not None and sensing_time < self.validity_start:
            return False
        if self.validity_stop is not None and sensing_time >= self.validity_stop:
            return False
        return True


@dataclass(frozen=True)
class TimingTable:
    '''
    A relative timing table

    Attributes
    ----------
    name: str
        Product name (file name without .json), recorded in output metadata
    generation: datetime.datetime
        When the table was made (UTC)
    reference_platform: str
        Platform the corrections are relative to; its correction is zero
    entries: tuple of TimingEntry
    '''
    name: str
    generation: datetime.datetime
    reference_platform: str
    entries: tuple

    @property
    def platforms(self):
        '''Platforms the table corrects'''
        return sorted({e.platform for e in self.entries})

    def lookup(self, platform, sensing_time, aux_ins_name):
        '''
        Find the entry for one burst

        Parameters
        ----------
        platform: str
            Burst mission identifier, e.g. 'S1D'
        sensing_time: datetime.datetime
            Burst sensing start (UTC)
        aux_ins_name: str or None
            AUX_INS the IPF used, as recorded in manifest.safe

        Returns
        -------
        TimingEntry or None
            None when the table does not correct this platform at all (the
            reference platform, or one deliberately left out): zero correction
            by policy.

        Raises
        ------
        UncoveredBurstError
            The table corrects this platform but no entry covers the burst's
            sensing time, or the burst was focused with an AUX_INS generation
            no entry was derived for.
        '''
        candidates = [e for e in self.entries if e.platform == platform]
        if not candidates:
            return None

        in_time = [e for e in candidates if e.covers_time(sensing_time)]
        if not in_time:
            raise UncoveredBurstError(
                f'{self.name}: no {platform} entry is valid at {sensing_time}')

        for entry in in_time:
            if aux_ins_name in entry.aux_ins:
                return entry
        raise UncoveredBurstError(
            f'{self.name}: {platform} burst focused with AUX_INS '
            f'{aux_ins_name}, but the entry was derived for '
            f'{", ".join(in_time[0].aux_ins)}')


def parse_aux_name(path):
    '''
    Split an ESA aux product name into its parts

    Parameters
    ----------
    path: str
        Path or name of an AUX_INS/AUX_ITC product, with or without
        .SAFE/.SAFE.zip

    Returns
    -------
    dict
        platform, type ('INS' or 'ITC'), validity and generation (datetime),
        and name (no .SAFE)
    '''
    name = os.path.basename(path)
    for suffix in ('.zip', '.SAFE'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    match = _AUX_NAME.match(name)
    if match is None:
        raise ValueError(f'Not an S1 AUX_INS/AUX_ITC product name: {path}')
    parts = match.groupdict()
    for key in ('validity', 'generation'):
        parts[key] = _parse_compact_time(parts[key])
    parts['name'] = name
    return parts


def _parse_compact_time(text):
    return datetime.datetime.strptime(text, '%Y%m%dT%H%M%S')


def _instrument_value(values, what, path):
    '''One instrument-wide value, refusing to average a beam-dependent one'''
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(
            f'{what} differs between IW channels in {path} ({unique}); a '
            'single instrument-wide correction would be wrong')
    return unique[0]


def read_platform_timing(aux_ins_zip, aux_itc_zip):
    '''
    Read one platform's timing components from its AUX_INS and AUX_ITC

    Parameters
    ----------
    aux_ins_zip, aux_itc_zip: str
        Paths to the two .SAFE.zip products of the same platform

    Returns
    -------
    dict
        azimuthTimeBias, azimuthCalibration, swstBias, rangeCalibration [s],
        each the same across all IW swath/polarisation channels
    '''
    # Imported here: the SAS only reads tables, and s1reader's aux readers are
    # needed only to build one
    from s1reader.s1_annotation import AuxIns, AuxItc

    ins_platform = parse_aux_name(aux_ins_zip)['platform']
    itc_platform = parse_aux_name(aux_itc_zip)['platform']
    if ins_platform != itc_platform:
        raise ValueError(f'AUX_INS is {ins_platform} but AUX_ITC is '
                         f'{itc_platform}')

    ins = [AuxIns.load_from_zip_file(aux_ins_zip, pol, swath)
           for swath, pol in _IW_CHANNELS]
    itc = [AuxItc.load_from_zip_file(aux_itc_zip, pol, swath)
           for swath, pol in _IW_CHANNELS]
    return {
        'azimuthTimeBias': _instrument_value(
            [a.azimuth_time_bias for a in ins], 'azimuthTimeBias', aux_ins_zip),
        'azimuthCalibration': _instrument_value(
            [a.azimuth_time_total for a in itc], 'azimuth calibration',
            aux_itc_zip),
        'swstBias': _instrument_value(
            [a.swst_bias for a in ins], 'swstBias', aux_ins_zip),
        'rangeCalibration': _instrument_value(
            [a.range_time_total for a in itc], 'range calibration',
            aux_itc_zip),
    }


def _azimuth_total(components):
    return components['azimuthTimeBias'] + components['azimuthCalibration']


def _range_total(components, range_components):
    return sum(components[c] for c in range_components)


def _relative(total, reference_total):
    '''Correction -(T - T_ref); + 0.0 so equal totals write 0.0, not -0.0'''
    return -(total - reference_total) + 0.0


def _group_aux_files(aux_files):
    '''{platform: {'INS': path, 'ITC': path}}, one product of each per platform'''
    grouped = {}
    for path in aux_files:
        parts = parse_aux_name(path)
        slot = grouped.setdefault(parts['platform'], {})
        if parts['type'] in slot:
            raise ValueError(
                f'Two AUX_{parts["type"]} products for {parts["platform"]}: '
                f'{slot[parts["type"]]} and {path}. Pass exactly one; list '
                'other AUX_INS generations the entry should accept with '
                'also_accept_aux_ins.')
        slot[parts['type']] = path
    return grouped


def build_timing_table(aux_files, reference_platform, apply_platforms,
                       generation, range_components=DEFAULT_RANGE_COMPONENTS,
                       validity=None, also_accept_aux_ins=None):
    '''
    Build a relative timing table from ESA aux products

    Parameters
    ----------
    aux_files: list of str
        AUX_INS and AUX_ITC .SAFE.zip paths: one of each for the reference
        platform and for every platform in apply_platforms
    reference_platform: str
        Platform the corrections are relative to, e.g. 'S1C'
    apply_platforms: list of str
        Platforms to correct, e.g. ['S1D']; every other platform gets none
    generation: datetime.datetime
        Generation time of the table (UTC); part of its name
    range_components: tuple of str
        Which of RANGE_COMPONENTS to sum into the range correction
    validity: dict, optional
        {platform: (start, stop)} sensing-time windows; missing or None ends
        stay open
    also_accept_aux_ins: dict, optional
        {platform: [aux_ins_name, ...]} further AUX_INS generations the entry
        covers, e.g. an older generation the IPF used before ESA revised the
        file in place

    Returns
    -------
    dict
        The table as JSON-ready data; write it with ``write_timing_table``
    '''
    range_components = tuple(range_components)
    unknown = set(range_components) - set(RANGE_COMPONENTS)
    if unknown or not range_components:
        raise ValueError(f'range_components must be a non-empty subset of '
                         f'{RANGE_COMPONENTS}, got {range_components}')
    if reference_platform in apply_platforms:
        raise ValueError(f'{reference_platform} is the reference; it cannot '
                         'also be corrected')
    if not apply_platforms:
        raise ValueError('apply_platforms is empty: nothing to correct')
    validity = validity or {}
    also_accept_aux_ins = also_accept_aux_ins or {}

    grouped = _group_aux_files(aux_files)
    needed = [reference_platform, *apply_platforms]
    for platform in needed:
        missing = {'INS', 'ITC'} - set(grouped.get(platform, {}))
        if missing:
            raise ValueError(f'No AUX_{"/AUX_".join(sorted(missing))} given '
                             f'for {platform}')

    def components_of(platform):
        return read_platform_timing(grouped[platform]['INS'],
                                    grouped[platform]['ITC'])

    reference = components_of(reference_platform)
    entries = []
    for platform in apply_platforms:
        comps = components_of(platform)
        start, stop = validity.get(platform) or (None, None)
        aux_ins = [parse_aux_name(grouped[platform]['INS'])['name']]
        aux_ins += [n for n in also_accept_aux_ins.get(platform, [])
                    if n not in aux_ins]
        entries.append({
            'platform': platform,
            'validityStart': _format_time(start),
            'validityStop': _format_time(stop),
            'azimuthTimeCorrection': _relative(
                _azimuth_total(comps), _azimuth_total(reference)),
            'rangeTimeCorrection': _relative(
                _range_total(comps, range_components),
                _range_total(reference, range_components)),
            'rangeComponents': list(range_components),
            'auxIns': aux_ins,
            'auxItc': parse_aux_name(grouped[platform]['ITC'])['name'],
            'components': {**comps, 'reference': reference},
        })

    return {
        'productType': PRODUCT_TYPE,
        'name': timing_table_name(reference_platform, generation),
        'generation': _format_time(generation),
        'referencePlatform': reference_platform,
        'referenceAux': {
            'auxIns': parse_aux_name(grouped[reference_platform]['INS'])['name'],
            'auxItc': parse_aux_name(grouped[reference_platform]['ITC'])['name'],
        },
        'convention': CONVENTION,
        'entries': entries,
    }


def timing_table_name(reference_platform, generation):
    '''File stem of a table, e.g. OPERA_S1_AUX_TIMING_S1C-REF_G20261023T000000'''
    return (f'{PRODUCT_TYPE}_{reference_platform}-REF_'
            f'G{generation:%Y%m%dT%H%M%S}')


def _format_time(value):
    return None if value is None else value.strftime('%Y-%m-%dT%H:%M:%SZ')


def _parse_time(value, what):
    if value is None:
        return None
    try:
        return datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except (TypeError, ValueError):
        raise ValueError(f'{what} must be YYYY-MM-DDTHH:MM:SSZ or null, got '
                         f'{value!r}')


def write_timing_table(table, out_dir):
    '''
    Write a table built by ``build_timing_table`` as <name>.json

    Returns
    -------
    str
        Path written
    '''
    path = os.path.join(out_dir, f'{table["name"]}.json')
    with open(path, 'w') as f:
        json.dump(table, f, indent=2)
        f.write('\n')
    return path


_REQUIRED_ENTRY_KEYS = ('platform', 'validityStart', 'validityStop',
                        'azimuthTimeCorrection', 'rangeTimeCorrection',
                        'rangeComponents', 'auxIns', 'auxItc')


def _entry_from_json(raw, i):
    where = f'entries[{i}]'
    missing = [k for k in _REQUIRED_ENTRY_KEYS if k not in raw]
    if missing:
        raise ValueError(f'{where} lacks {", ".join(missing)}')
    if not re.fullmatch(r'S1[A-Z]', str(raw['platform'])):
        raise ValueError(f'{where}.platform must look like S1D, got '
                         f'{raw["platform"]!r}')
    for key in ('azimuthTimeCorrection', 'rangeTimeCorrection'):
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            raise ValueError(f'{where}.{key} must be a number in seconds')
    aux_ins = raw['auxIns']
    if (not isinstance(aux_ins, list) or not aux_ins
            or not all(isinstance(n, str) for n in aux_ins)):
        raise ValueError(f'{where}.auxIns must be a non-empty list of AUX_INS '
                         'names')
    range_components = tuple(raw['rangeComponents'])
    if not set(range_components) <= set(RANGE_COMPONENTS):
        raise ValueError(f'{where}.rangeComponents must be a subset of '
                         f'{RANGE_COMPONENTS}')

    entry = TimingEntry(
        platform=raw['platform'],
        validity_start=_parse_time(raw['validityStart'],
                                   f'{where}.validityStart'),
        validity_stop=_parse_time(raw['validityStop'], f'{where}.validityStop'),
        azimuth_time_correction=float(raw['azimuthTimeCorrection']),
        range_time_correction=float(raw['rangeTimeCorrection']),
        aux_ins=tuple(aux_ins),
        aux_itc=raw['auxItc'],
        components=dict(raw.get('components', {})),
        range_components=range_components,
    )
    if (entry.validity_start is not None and entry.validity_stop is not None
            and entry.validity_stop <= entry.validity_start):
        raise ValueError(f'{where}: validityStop is not after validityStart')
    return entry


def _windows_overlap(a, b):
    lo = datetime.datetime.min
    hi = datetime.datetime.max
    return ((a.validity_start or lo) < (b.validity_stop or hi)
            and (b.validity_start or lo) < (a.validity_stop or hi))


def load_timing_table(path):
    '''
    Read and validate a relative timing table

    Parameters
    ----------
    path: str
        Path to an OPERA_S1_AUX_TIMING .json file

    Returns
    -------
    TimingTable

    Raises
    ------
    ValueError
        If the file is not a valid table: wrong productType, missing or
        mistyped fields, the reference platform listed as corrected, or two
        entries for one platform with overlapping validity
    '''
    with open(path) as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f'{path} is not JSON: {err}')

    if raw.get('productType') != PRODUCT_TYPE:
        raise ValueError(f'{path}: productType must be {PRODUCT_TYPE}, got '
                         f'{raw.get("productType")!r}')
    for key in ('name', 'generation', 'referencePlatform', 'entries'):
        if key not in raw:
            raise ValueError(f'{path} lacks {key}')
    if not isinstance(raw['entries'], list):
        raise ValueError(f'{path}: entries must be a list')

    entries = tuple(_entry_from_json(e, i) for i, e in enumerate(raw['entries']))
    reference = raw['referencePlatform']
    for entry in entries:
        if entry.platform == reference:
            raise ValueError(f'{path}: {reference} is the reference platform '
                             'and cannot have an entry')
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if a.platform == b.platform and _windows_overlap(a, b):
                raise ValueError(f'{path}: two {a.platform} entries overlap in '
                                 'validity')

    return TimingTable(
        name=raw['name'],
        generation=_parse_time(raw['generation'], 'generation'),
        reference_platform=reference,
        entries=entries,
    )


def _platform_window(text):
    '''PLATFORM=START[/STOP] -> (platform, (start, stop))'''
    platform, _, window = text.partition('=')
    if not window:
        raise argparse.ArgumentTypeError(
            f'expected PLATFORM=START[/STOP], got {text!r}')
    start, _, stop = window.partition('/')
    parse = (lambda s: datetime.datetime.fromisoformat(s) if s else None)
    try:
        return platform, (parse(start), parse(stop))
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err))


def get_parser():
    '''Command line for building a table'''
    parser = argparse.ArgumentParser(
        description='Build an OPERA_S1_AUX_TIMING table of per-platform '
                    'timing corrections relative to a reference platform, '
                    'from ESA AUX_INS and AUX_ITC products.')
    parser.add_argument('aux_files', nargs='+',
                        help='AUX_INS and AUX_ITC .SAFE.zip files, one of each '
                             'per platform used')
    parser.add_argument('--reference', required=True,
                        help='Reference platform, e.g. S1C')
    parser.add_argument('--apply', nargs='+', required=True,
                        help='Platforms to correct, e.g. S1D')
    parser.add_argument('--range-components', nargs='+',
                        default=list(DEFAULT_RANGE_COMPONENTS),
                        choices=RANGE_COMPONENTS,
                        help='Components summed into the range correction '
                             '(default: rangeCalibration only)')
    parser.add_argument('--validity', type=_platform_window, action='append',
                        default=[], metavar='PLATFORM=START[/STOP]',
                        help='Sensing-time window for a platform, ISO dates; '
                             'open-ended by default')
    parser.add_argument('--also-accept-aux-ins', action='append', default=[],
                        metavar='AUX_INS_NAME',
                        help='Further AUX_INS generation an entry covers')
    parser.add_argument('--generation', type=datetime.datetime.fromisoformat,
                        default=None,
                        help='Table generation time, ISO (default: now, UTC)')
    parser.add_argument('-o', '--out-dir', default='.',
                        help='Directory to write the table into')
    return parser


def main(argv=None):
    '''Build and write a table; print its path'''
    args = get_parser().parse_args(argv)
    generation = args.generation or datetime.datetime.now(
        datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
    also = {}
    for name in args.also_accept_aux_ins:
        also.setdefault(parse_aux_name(name)['platform'], []).append(
            parse_aux_name(name)['name'])
    table = build_timing_table(
        args.aux_files, args.reference, args.apply, generation,
        range_components=args.range_components,
        validity=dict(args.validity),
        also_accept_aux_ins=also)
    path = write_timing_table(table, args.out_dir)
    print(path)
    return path


if __name__ == '__main__':
    main()
