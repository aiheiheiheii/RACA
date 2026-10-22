"""No-data plotting cannot be reported as a successful figure delivery."""
import importlib
import json

import pytest

plot = importlib.import_module('experiments.02_overhead.plot_overhead')


def test_plot_value_does_not_require_the_legacy_mean_column():
    assert plot._mean({'value':'12.5','n_valid':'1'}) == 12.5
    assert plot._mean({'mean':'13.5','n_valid':'1'}) == 13.5


@pytest.mark.parametrize('stale_images', [False, True])
def test_plot_results_with_no_data_raises_even_when_old_files_exist(tmp_path, monkeypatch, stale_images):
    logs = tmp_path/'logs'
    logs.mkdir()
    manifest = dict(identity_schema='computation-v1', implementation_version=plot.OVERHEAD_IMPLEMENTATION_VERSION,
        run_fingerprint='synthetic-zero-data', status='complete', seeds=list(plot.SEEDS),
        methods=list(plot.METHODS), derived_outputs=dict(state='complete',run_fingerprint='synthetic-zero-data'))
    (logs/'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    records = {(n,d,method):dict(method=method,n_valid='0',value='nan',mean='nan')
        for n in plot.N_VALUES for d in plot.D_PANEL_VALUES for method in plot.METHODS}
    monkeypatch.setattr(plot, '_read_summary', lambda *args: records)
    monkeypatch.setattr(plot, 'apply_style', lambda: 'test-font')
    if stale_images:
        (tmp_path/'figures').mkdir()
        for name in ('Fig_O1_time_overhead','Fig_O2_communication_overhead'):
            (tmp_path/'figures'/f'{name}.png').write_bytes(b'old-stale-image')
    with pytest.raises(ValueError, match='No complete pair of result figures') as caught:
        plot.plot_results(tmp_path)
    assert str(tmp_path/'figures') in str(caught.value)
    assert not list((tmp_path/'figures').glob('*.pdf'))
