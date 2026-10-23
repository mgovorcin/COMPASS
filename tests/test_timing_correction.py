'''
Test the relative S1 timing table: building it from ESA aux products, reading
it back, and looking bursts up in it

The AUX_INS products come from s1reader's packaged data; the AUX_ITC products
are small and live in tests/data/aux_itc.
'''
import datetime
import json
import pathlib

import pytest

from s1reader import s1_annotation

from compass.utils import timing_correction as tc

AUX_INS_DIR = pathlib.Path(s1_annotation.__file__).parent / 'data' / 'aux_ins'
AUX_ITC_DIR = pathlib.Path(__file__).parent / 'data' / 'aux_itc'

AUX_INS = {
    'S1A': 'S1A_AUX_INS_V20250601T000000_G20251015T084917',
    'S1C': 'S1C_AUX_INS_V20250224T000000_G20260604T134021',
    'S1D': 'S1D_AUX_INS_V20260319T000000_G20260603T113743',
}
AUX_ITC = {
    'S1A': 'S1A_AUX_ITC_V20160627T000000_G20230406T084701',
    'S1C': 'S1C_AUX_ITC_V20250519T094846_G20250521T125837',
    'S1D': 'S1D_AUX_ITC_V20260319T000000_G20260318T105930',
}
GENERATION = datetime.datetime(2026, 10, 23)


def aux_files(*platforms):
    '''AUX_INS + AUX_ITC paths for the given platforms'''
    files = []
    for p in platforms:
        files.append(str(AUX_INS_DIR / f'{AUX_INS[p]}.SAFE.zip'))
        files.append(str(AUX_ITC_DIR / f'{AUX_ITC[p]}.SAFE.zip'))
    return files


def s1d_table(**kwargs):
    '''The production case: S1D corrected relative to S1C'''
    return tc.build_timing_table(aux_files('S1C', 'S1D'), 'S1C', ['S1D'],
                                 GENERATION, **kwargs)


def written(tmp_path, table):
    return tc.load_timing_table(tc.write_timing_table(table, str(tmp_path)))


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def test_s1d_relative_to_s1c_matches_aux_chain():
    '''T_S1D - T_S1C = -66.299 - (-19.685) = -46.614 us, so the correction is
    +46.614 us. The archive ESD median for S1C-S1D is +47.48 us; this pins the
    sign the trade study confirmed on real seams.'''
    table = s1d_table()
    (entry,) = table['entries']
    assert entry['platform'] == 'S1D'
    assert entry['azimuthTimeCorrection'] == pytest.approx(46.614e-6, abs=1e-12)
    assert entry['rangeTimeCorrection'] == 0.0
    # -0.0 would read like a sign bug in a reviewed, config-controlled file
    assert '-0.0' not in json.dumps(table)


def test_s1a_relative_to_s1c_includes_itc():
    '''S1A: AUX_INS 0, AUX_ITC +6.3522 us, so T_S1A - T_S1C = +26.037 us. The
    ITC term is what brings the A-D archive residual from -8.5 to -2.2 us.'''
    table = tc.build_timing_table(aux_files('S1A', 'S1C'), 'S1C', ['S1A'],
                                  GENERATION)
    (entry,) = table['entries']
    assert entry['azimuthTimeCorrection'] == pytest.approx(-26.0372e-6,
                                                           abs=1e-12)
    # S1A rangeCalibration +0.74103 ns against S1C's 0
    assert entry['rangeTimeCorrection'] == pytest.approx(-7.4103e-10,
                                                         abs=1e-16)


def test_swst_bias_only_with_opt_in():
    '''swstBias is stored either way but summed into the range correction only
    when asked: S1D -6.4577 ns vs S1C -5.9341 ns.'''
    default = s1d_table()['entries'][0]
    with_swst = s1d_table(
        range_components=('rangeCalibration', 'swstBias'))['entries'][0]
    assert default['rangeComponents'] == ['rangeCalibration']
    assert default['components']['swstBias'] == pytest.approx(-6.4576909e-09)
    assert with_swst['rangeTimeCorrection'] == pytest.approx(
        -(-6.4576909e-09 - -5.9341056e-09), abs=1e-16)


def test_provenance_names_every_input():
    table = s1d_table()
    entry = table['entries'][0]
    assert entry['auxIns'] == [AUX_INS['S1D']]
    assert entry['auxItc'] == AUX_ITC['S1D']
    assert table['referenceAux'] == {'auxIns': AUX_INS['S1C'],
                                     'auxItc': AUX_ITC['S1C']}
    assert entry['components']['reference']['azimuthTimeBias'] == \
        pytest.approx(-19.685e-6)
    assert table['name'] == 'OPERA_S1_AUX_TIMING_S1C-REF_G20261023T000000'


def test_build_is_deterministic(tmp_path):
    '''Same inputs, same bytes: the table is reviewed and config-controlled'''
    a = tc.write_timing_table(s1d_table(), str(tmp_path))
    first = pathlib.Path(a).read_bytes()
    b = tc.write_timing_table(s1d_table(), str(tmp_path))
    assert pathlib.Path(b).read_bytes() == first


def test_also_accept_aux_ins_extends_coverage():
    older = 'S1D_AUX_INS_V20260319T000000_G20260101T000000'
    entry = s1d_table(also_accept_aux_ins={'S1D': [older]})['entries'][0]
    assert entry['auxIns'] == [AUX_INS['S1D'], older]


@pytest.mark.parametrize('kwargs,match', [
    ({'apply_platforms': ['S1C']}, 'is the reference'),
    ({'apply_platforms': []}, 'nothing to correct'),
    ({'apply_platforms': ['S1A']}, 'No AUX_INS/AUX_ITC given for S1A'),
    ({'range_components': ('rangeCalibration', 'bogus')}, 'range_components'),
    ({'range_components': ()}, 'range_components'),
])
def test_build_rejects_bad_requests(kwargs, match):
    args = dict(aux_files=aux_files('S1C', 'S1D'), reference_platform='S1C',
                apply_platforms=['S1D'], generation=GENERATION)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        tc.build_timing_table(**args)


def test_build_rejects_two_generations_of_one_product():
    files = aux_files('S1C', 'S1D') + [
        str(AUX_INS_DIR / f'{AUX_INS["S1D"]}.SAFE.zip').replace(
            'G20260603T113743', 'G20260101T000000')]
    with pytest.raises(ValueError, match='Two AUX_INS products for S1D'):
        tc.build_timing_table(files, 'S1C', ['S1D'], GENERATION)


@pytest.mark.parametrize('name', ['S1D_AUX_CAL_V20260319T000000_G20260318T105930',
                                  'S1D_AUX_INS_V2026_G2026', 'random.zip'])
def test_parse_aux_name_rejects_other_products(name):
    with pytest.raises(ValueError, match='Not an S1 AUX_INS/AUX_ITC'):
        tc.parse_aux_name(name)


def test_parse_aux_name_strips_suffixes():
    parts = tc.parse_aux_name(f'/x/{AUX_ITC["S1D"]}.SAFE.zip')
    assert parts['platform'] == 'S1D'
    assert parts['type'] == 'ITC'
    assert parts['name'] == AUX_ITC['S1D']
    assert parts['generation'] == datetime.datetime(2026, 3, 18, 10, 59, 30)


# --------------------------------------------------------------------------- #
# reading back
# --------------------------------------------------------------------------- #
def test_round_trip(tmp_path):
    table = written(tmp_path, s1d_table())
    assert table.name == 'OPERA_S1_AUX_TIMING_S1C-REF_G20261023T000000'
    assert table.generation == GENERATION
    assert table.reference_platform == 'S1C'
    assert table.platforms == ['S1D']
    (entry,) = table.entries
    assert entry.azimuth_time_correction == pytest.approx(46.614e-6, abs=1e-12)
    assert entry.aux_ins == (AUX_INS['S1D'],)
    assert entry.validity_start is None and entry.validity_stop is None


def raw_table(tmp_path, mutate):
    '''Write a valid table, let ``mutate`` break it, return its path'''
    raw = s1d_table()
    mutate(raw)
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(raw))
    return str(path)


@pytest.mark.parametrize('mutate,match', [
    (lambda t: t.update(productType='OTHER'), 'productType'),
    (lambda t: t.pop('referencePlatform'), 'lacks referencePlatform'),
    (lambda t: t['entries'][0].pop('auxIns'), r'entries\[0\] lacks auxIns'),
    (lambda t: t['entries'][0].update(auxIns=[]), 'auxIns must be'),
    (lambda t: t['entries'][0].update(azimuthTimeCorrection='46us'),
     'must be a number'),
    (lambda t: t['entries'][0].update(azimuthTimeCorrection=True),
     'must be a number'),
    (lambda t: t['entries'][0].update(platform='Sentinel-1D'),
     'platform must look like'),
    (lambda t: t['entries'][0].update(validityStart='2026-07-01'),
     'validityStart must be'),
    (lambda t: t['entries'][0].update(validityStart='2026-07-01T00:00:00Z',
                                      validityStop='2026-06-01T00:00:00Z'),
     'not after validityStart'),
    (lambda t: t['entries'][0].update(rangeComponents=['bogus']),
     'rangeComponents'),
    (lambda t: t['entries'][0].update(platform='S1C'),
     'S1C is the reference platform'),
    (lambda t: t['entries'].append(dict(t['entries'][0])), 'overlap'),
])
def test_load_rejects_invalid_tables(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        tc.load_timing_table(raw_table(tmp_path, mutate))


def test_load_rejects_non_json(tmp_path):
    path = tmp_path / 'table.json'
    path.write_text('not json')
    with pytest.raises(ValueError, match='is not JSON'):
        tc.load_timing_table(str(path))


def test_load_accepts_adjacent_windows(tmp_path):
    '''Two S1D entries that meet but do not overlap, e.g. either side of an
    instrument event, are a valid table.'''
    def split(t):
        late = dict(t['entries'][0])
        t['entries'][0]['validityStop'] = '2026-07-15T00:00:00Z'
        late['validityStart'] = '2026-07-15T00:00:00Z'
        t['entries'].append(late)
    table = tc.load_timing_table(raw_table(tmp_path, split))
    assert len(table.entries) == 2


# --------------------------------------------------------------------------- #
# lookup
# --------------------------------------------------------------------------- #
WHEN = datetime.datetime(2026, 8, 1, 13, 55)


@pytest.mark.parametrize('platform', ['S1C', 'S1A', 'S1B'])
def test_lookup_uncorrected_platform_is_none(tmp_path, platform):
    '''The reference, and platforms deliberately left out, get no entry: zero
    correction by policy, not an error.'''
    table = written(tmp_path, s1d_table())
    assert table.lookup(platform, WHEN, 'whatever') is None


def test_lookup_finds_covered_burst(tmp_path):
    table = written(tmp_path, s1d_table())
    entry = table.lookup('S1D', WHEN, AUX_INS['S1D'])
    assert entry.azimuth_time_correction == pytest.approx(46.614e-6, abs=1e-12)


@pytest.mark.parametrize('aux_ins_name', [
    'S1D_AUX_INS_V20260319T000000_G20270101T000000', None])
def test_lookup_refuses_other_aux_ins_generation(tmp_path, aux_ins_name):
    '''A generation the entry was not derived for -- ESA revised the file, or
    the IPF started applying the bias -- must not be corrected silently.'''
    table = written(tmp_path, s1d_table())
    with pytest.raises(tc.UncoveredBurstError, match='derived for'):
        table.lookup('S1D', WHEN, aux_ins_name)


def test_lookup_respects_validity(tmp_path):
    start = datetime.datetime(2026, 7, 15)
    table = written(tmp_path, s1d_table(validity={'S1D': (start, None)}))
    assert table.lookup('S1D', start, AUX_INS['S1D']) is not None
    with pytest.raises(tc.UncoveredBurstError, match='no S1D entry is valid'):
        table.lookup('S1D', start - datetime.timedelta(seconds=1),
                     AUX_INS['S1D'])


def test_validity_stop_is_exclusive():
    stop = datetime.datetime(2026, 7, 15)
    entry = tc.TimingEntry('S1D', None, stop, 0.0, 0.0, ('x',), None)
    assert entry.covers_time(stop - datetime.timedelta(microseconds=1))
    assert not entry.covers_time(stop)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def test_cli_builds_the_production_table(tmp_path, capsys):
    path = tc.main([*aux_files('S1C', 'S1D'), '--reference', 'S1C',
                    '--apply', 'S1D', '--generation', '2026-10-23',
                    '--validity', 'S1D=2026-07-15', '-o', str(tmp_path)])
    assert capsys.readouterr().out.strip() == path
    table = tc.load_timing_table(path)
    assert table.entries[0].validity_start == datetime.datetime(2026, 7, 15)
    assert table.entries[0].azimuth_time_correction == pytest.approx(
        46.614e-6, abs=1e-12)


def test_cli_also_accept_groups_by_platform(tmp_path):
    older = 'S1D_AUX_INS_V20260319T000000_G20260101T000000'
    path = tc.main([*aux_files('S1C', 'S1D'), '--reference', 'S1C',
                    '--apply', 'S1D', '--generation', '2026-10-23',
                    '--also-accept-aux-ins', older, '-o', str(tmp_path)])
    assert tc.load_timing_table(path).entries[0].aux_ins == (AUX_INS['S1D'],
                                                             older)


@pytest.mark.parametrize('argv', [
    ['--reference', 'S1C', '--apply', 'S1D'],           # no aux files
    ['x.zip', '--apply', 'S1D'],                        # no --reference
    ['x.zip', '--reference', 'S1C', '--apply', 'S1D',
     '--validity', 'S1D'],                              # malformed window
    ['x.zip', '--reference', 'S1C', '--apply', 'S1D',
     '--range-components', 'bogus'],
])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as err:
        tc.main(argv)
    assert err.value.code == 2


def test_cli_help():
    with pytest.raises(SystemExit) as err:
        tc.main(['--help'])
    assert err.value.code == 0
