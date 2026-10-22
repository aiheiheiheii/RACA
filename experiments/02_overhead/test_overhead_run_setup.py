"""Calibration snapshot invariants, independent of any real experiment run."""
from copy import deepcopy
import hashlib
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

setup=importlib.import_module('experiments.02_overhead.overhead_run_setup')
calibration_module=importlib.import_module('experiments.02_overhead.overhead_calibration')


def record(seed,d):
    costs=dict(cosine_per_pair_ms=.001,sha256_per_descriptor_ms=.002,
               scalar_compare_ms=.0001,scalar_modulo_ms=.0002,vector_per_byte_ms=.000001)
    timings=dict(cosine=costs['cosine_per_pair_ms']*calibration_module.COSINE_BATCH_PAIRS,
                 sha256=costs['sha256_per_descriptor_ms']*calibration_module.HASH_BATCH_DESCRIPTORS,
                 scalar_compare=costs['scalar_compare_ms']*calibration_module.SCALAR_BATCH_ITEMS,
                 scalar_modulo=costs['scalar_modulo_ms']*calibration_module.SCALAR_BATCH_ITEMS,
                 vector=costs['vector_per_byte_ms']*calibration_module.VECTOR_BATCH_BYTES)
    return calibration_module.KernelCalibration(
        costs,seed,d,(setup.CALIBRATION_POPULATION_N,d),
        hashlib.sha256(f'synthetic-{seed}-{d}'.encode()).hexdigest(),
        calibration_module.CALIBRATION_SAMPLE_LIMIT,sum(timings.values()),timings)


@pytest.fixture
def snapshot():
    return {setup.calibration_key(seed,d):record(seed,d).to_dict()
            for seed in (2026,2027) for d in (10,20)}


def test_new_generation_calibrates_full_population_exactly_once_per_seed_dimension(monkeypatch):
    loaded,calibrated=[],[]
    config=SimpleNamespace(corruption_rate_percent=3.,H=.5)
    class NoLabels(dict):
        def __getitem__(self,key):
            if key != 'X':
                raise AssertionError('Calibration may read raw descriptors only')
            return super().__getitem__(key)
    def loader(**kwargs):
        loaded.append(kwargs)
        return NoLabels(X=np.ones((setup.CALIBRATION_POPULATION_N,kwargs['dim']),np.float32))
    def calibrate(x,seed,d):
        assert x.shape==(setup.CALIBRATION_POPULATION_N,d)
        calibrated.append((seed,d))
        return record(seed,d)
    monkeypatch.setattr(setup,'calibrate_kernel_costs',calibrate)
    records,calibrations=setup.build_calibration_snapshot(loader,'synthetic-root',config,(2026,2027),(10,20))
    assert calibrated==[(2026,10),(2026,20),(2027,10),(2027,20)]
    assert [(item['seed'],item['dim']) for item in loaded]==calibrated
    assert all(item['n']==5000 and item['corruption_rate_percent']==3. and item['H']==.5 for item in loaded)
    assert len(records)==len(calibrations)==4
    assert all(records[setup.calibration_key(seed,d)]==value.to_dict()
               for (seed,d),value in calibrations.items())


def test_snapshot_hash_survives_json_roundtrip_and_key_reordering(snapshot):
    policy=setup.calibration_policy()
    first=setup.calibration_snapshot_sha256(snapshot,policy)
    restored=json.loads(json.dumps(snapshot,sort_keys=True))
    assert setup.calibration_snapshot_sha256(restored,policy)==first
    reversed_records=dict(reversed(list(snapshot.items())))
    assert setup.calibration_snapshot_sha256(reversed_records,dict(reversed(list(policy.items()))))==first
    assert set(setup.validate_calibration_snapshot(restored,policy,(2026,2027),(10,20)))=={
        (2026,10),(2026,20),(2027,10),(2027,20)}


def test_snapshot_hash_binds_both_unit_cost_record_and_policy(snapshot):
    policy=setup.calibration_policy()
    original=setup.calibration_snapshot_sha256(snapshot,policy)
    changed=deepcopy(snapshot)
    changed['seed2026_d10']['costs_ms']['cosine_per_pair_ms'] *=2
    assert setup.calibration_snapshot_sha256(changed,policy)!=original
    changed_policy=dict(policy,timed_batches_per_kernel=2)
    assert setup.calibration_snapshot_sha256(snapshot,changed_policy)!=original


@pytest.mark.parametrize('mutation',['missing','extra','seed','dimension','population','policy','tampered_cost'])
def test_corrupt_or_different_generation_snapshot_cannot_supply_unit_costs(snapshot,mutation):
    policy=setup.calibration_policy()
    if mutation=='missing':
        del snapshot['seed2026_d10']
    elif mutation=='extra':
        snapshot['seed2028_d10']=record(2028,10).to_dict()
    elif mutation=='seed':
        snapshot['seed2026_d10']=record(2028,10).to_dict()
    elif mutation=='dimension':
        snapshot['seed2026_d10']=record(2026,20).to_dict()
    elif mutation=='population':
        snapshot['seed2026_d10']['input_shape'][0]=4999
    elif mutation=='policy':
        policy['threadpool_limit']=2
    else:
        snapshot['seed2026_d10']['costs_ms']['cosine_per_pair_ms'] *=2
    with pytest.raises(ValueError):
        setup.validate_calibration_snapshot(snapshot,policy,(2026,2027),(10,20))
