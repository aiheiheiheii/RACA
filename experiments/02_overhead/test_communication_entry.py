"""Communication updates must never route through the timed experiment."""
from copy import deepcopy
import importlib
import json

import pytest

cli = importlib.import_module('experiments.02_overhead.run_overhead')
plotter = importlib.import_module('experiments.02_overhead.plot_overhead')
identity = importlib.import_module('experiments.02_overhead.overhead_identity')
runner = importlib.import_module('experiments.02_overhead.overhead_experiment')


@pytest.mark.parametrize('explicit',[False,True])
def test_complete_generation_routes_only_to_communication(tmp_path,monkeypatch,explicit):
    frozen = ('raw/overhead_runs.csv','summary/time_overhead_summary.csv',
              'summary/communication_overhead_summary.csv',
              'figures/Fig_O1_time_overhead.pdf','figures/Fig_O1_time_overhead.png')
    for relative in frozen:
        path = tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'frozen original '+relative.encode())
    manifest = tmp_path/'logs'/'run_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps(dict(status='complete',implementation_version='overhead-core-v4',
        derived_outputs=dict(state='complete'))),encoding='utf-8')
    original = {relative:(tmp_path/relative).read_bytes() for relative in frozen}
    def forbidden(**kwargs):
        pytest.fail('A communication-only update must not remeasure or materialize time')
    monkeypatch.setattr(runner,'run_experiment',forbidden)
    calls = []
    result = dict(status='complete',communication_fingerprint='synthetic-codec')
    def update(output,**kwargs):
        calls.append((output,kwargs))
        return result
    monkeypatch.setattr(cli,'_communication_outputs',update)
    args = ['--output-dir',str(tmp_path)] + (['--communication-only'] if explicit else [])
    assert cli.main(args) is result
    assert calls == [(tmp_path.resolve(),{})]
    assert {relative:(tmp_path/relative).read_bytes() for relative in frozen} == original


@pytest.mark.parametrize('conflict',['--force','--plots-only'])
def test_communication_flag_rejects_remeasurement_conflicts(tmp_path,conflict):
    with pytest.raises(SystemExit,match='cannot be combined'):
        cli.main(['--output-dir',str(tmp_path),'--communication-only',conflict])


@pytest.mark.parametrize('module',['raca_wire_codec.py','overhead_communication.py'])
def test_independent_transport_cannot_invalidate_existing_timed_core(module):
    assert not identity._is_computation_source('experiments/02_overhead/'+module)
    assert identity._is_computation_source('experiments/02_overhead/overhead_core_simulator.py')


def test_communication_plot_writes_only_o2(tmp_path,monkeypatch):
    overlay = importlib.import_module('experiments.02_overhead.overhead_communication')
    folder = tmp_path/'figures'
    folder.mkdir()
    for extension in ('pdf','png'):
        (folder/f'Fig_O1_time_overhead.{extension}').write_bytes(b'frozen O1')
    metadata = dict(base_run_fingerprint='synthetic-time',communication_fingerprint='synthetic-codec',
        schema='synthetic-encoded-schema',base_implementation_version='overhead-core-v4')
    monkeypatch.setattr(overlay,'validate_encoded_overlay',lambda *args,**kwargs:metadata)
    records = {(n,d,method):dict(N=n,d=d,method=method,n_valid='1',value='123',
        communication_fingerprint='synthetic-codec',communication_schema='synthetic-encoded-schema')
        for n in plotter.N_VALUES for d in plotter.D_PANEL_VALUES for method in plotter.METHODS}
    original = deepcopy(records)
    monkeypatch.setattr(plotter,'_read_summary',lambda *args:records)
    monkeypatch.setattr(plotter,'apply_style',lambda:'synthetic-font')
    calls = []
    def draw(given,location,kind,fingerprint,**kwargs):
        assert kind == 'communication'
        assert 'time experiment retains original uncompressed v4 transport' in kwargs['subject_suffix']
        calls.append(kind)
        for extension in ('pdf','png'):
            (location/f'Fig_O2_communication_overhead.{extension}').write_bytes(b'new O2')
        return dict(name='Fig_O2_communication_overhead',unit='KB')
    monkeypatch.setattr(plotter,'_five_panel',draw)
    result = plotter.plot_communication_results(tmp_path)
    assert result['figures_generated'] == ['Fig_O2_communication_overhead']
    assert calls == ['communication']
    assert records == original
    for extension in ('pdf','png'):
        assert (folder/f'Fig_O1_time_overhead.{extension}').read_bytes() == b'frozen O1'
        assert (folder/f'Fig_O2_communication_overhead.{extension}').read_bytes() == b'new O2'
