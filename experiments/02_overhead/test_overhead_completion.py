"""Synthetic checks for delivered figure paths and cache-run explanations.

The four byte fixtures stand for successful plotter saves. These checks concern
file delivery only; they never read or replace actual paper measurements.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

cli = importlib.import_module("experiments.02_overhead.run_overhead")
runner = importlib.import_module("experiments.02_overhead.overhead_experiment")
plotter = importlib.import_module("experiments.02_overhead.plot_overhead")


@pytest.fixture
def saved_figures(tmp_path):
    output = tmp_path / "result path with spaces"
    folder = output / "figures"
    folder.mkdir(parents=True)
    paths = [folder / f"{stem}.{extension}"
             for stem in cli.EXPECTED_FIGURE_STEMS for extension in ("pdf", "png")]
    for path in paths:
        path.write_bytes(b"synthetic nonempty figure")
    metadata = {
        "implementation_version": "synthetic-core-v3",
        "run_fingerprint": "synthetic-fingerprint",
        "figures_generated": list(cli.EXPECTED_FIGURE_STEMS),
        "status": "complete", "executed_configurations_this_invocation": 0,
        "resumed_configurations": 45,
    }
    return output, paths, metadata


def test_completion_reports_real_absolute_files_and_gallery(saved_figures, capsys):
    output, paths, metadata = saved_figures
    original = deepcopy(metadata)
    assert cli.complete_outputs(output, metadata) == paths
    stdout = capsys.readouterr().out
    assert "本次新运行 0 个配置" in stdout
    assert "全部配置命中缓存" in stdout and "未重新仿真" in stdout
    for path in paths:
        assert str(path.resolve()) in stdout
    gallery = (output / "figures" / "README.md").read_text(encoding="utf-8")
    for path in paths:
        assert f"<{path.resolve().as_posix()}>" in gallery
    assert "synthetic-core-v3" in gallery and "synthetic-fingerprint" in gallery
    assert "一个配对 seed" in gallery and "相同固定 B" in gallery
    assert "v2/v3" in gallery and "不同计费口径" in gallery
    assert "![时间开销]" in gallery and "![通信开销]" in gallery
    assert metadata == original


@pytest.mark.parametrize("index", range(4))
def test_missing_expected_file_fails_with_its_absolute_path(saved_figures, index, capsys):
    output, paths, metadata = saved_figures
    paths[index].unlink()
    with pytest.raises(RuntimeError, match="Missing or empty output files") as caught:
        cli.complete_outputs(output, metadata)
    assert str(paths[index]) in str(caught.value)
    assert not (output / "figures" / "README.md").exists()
    assert not capsys.readouterr().out


@pytest.mark.parametrize("index", range(4))
def test_empty_expected_file_cannot_count_as_success(saved_figures, index):
    output, paths, metadata = saved_figures
    paths[index].write_bytes(b"")
    with pytest.raises(RuntimeError) as caught:
        cli.complete_outputs(output, metadata)
    assert str(paths[index]) in str(caught.value)
    assert not (output / "figures" / "README.md").exists()


def test_directory_at_figure_path_cannot_count_as_file(saved_figures):
    output, paths, metadata = saved_figures
    paths[0].unlink()
    paths[0].mkdir()
    with pytest.raises(RuntimeError, match="Missing or empty output files"):
        cli.complete_outputs(output, metadata)


@pytest.mark.parametrize("declared", [[], None, ["Fig_O1_time_overhead"],
                                      ["Fig_O1_time_overhead", "Fig_O1_time_overhead"]])
def test_empty_or_partial_generation_rejects_stale_files(saved_figures, declared):
    output, paths, metadata = saved_figures
    metadata["figures_generated"] = declared
    with pytest.raises(RuntimeError, match="Expected O1/O2 figures") as caught:
        cli.complete_outputs(output, metadata)
    assert str(output / "figures") in str(caught.value)
    assert all(path.is_file() for path in paths)
    assert not (output / "figures" / "README.md").exists()


def test_partially_reused_run_reports_new_and_cached_counts(saved_figures, capsys):
    output, _, metadata = saved_figures
    metadata.update(executed_configurations_this_invocation=20, resumed_configurations=25)
    cli.complete_outputs(output, metadata)
    stdout = capsys.readouterr().out
    assert "本次新运行 20 个配置；验证并恢复缓存 25 个配置" in stdout
    assert "全部配置命中缓存" not in stdout


def test_main_run_preserves_return_and_shows_completion(saved_figures, monkeypatch, capsys):
    output, paths, metadata = saved_figures
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return metadata

    monkeypatch.setattr(runner, "run_experiment", run)
    assert cli.main(["--output-dir", str(output)]) is metadata
    assert calls == [{"output_dir": output.resolve(), "resume": True, "force": False}]
    assert str(paths[0]) in capsys.readouterr().out


def test_main_missing_figure_raises_instead_of_returning_success(saved_figures, monkeypatch):
    output, paths, metadata = saved_figures
    paths[-1].unlink()
    monkeypatch.setattr(runner, "run_experiment", lambda **kwargs: metadata)
    with pytest.raises(RuntimeError) as caught:
        cli.main(["--output-dir", str(output)])
    assert str(paths[-1]) in str(caught.value)


def test_plots_only_delivers_same_paths_without_simulator(saved_figures, monkeypatch, capsys):
    output, paths, metadata = saved_figures
    metadata["results_implementation_version"] = metadata.pop("implementation_version")
    calls = []

    def plot(root):
        calls.append(root)
        return metadata

    def forbidden_run(**kwargs):
        pytest.fail("--plots-only must not invoke the experiment")

    monkeypatch.setattr(plotter, "plot_results", plot)
    monkeypatch.setattr(runner, "run_experiment", forbidden_run)
    assert cli.main(["--plots-only", "--output-dir", str(output)]) is metadata
    assert calls == [output.resolve()]
    stdout = capsys.readouterr().out
    assert "本次仅根据已保存统计重绘，未执行新仿真" in stdout
    assert "全部配置命中缓存" not in stdout
    assert all(str(path) in stdout for path in paths)
    style = json.loads((output / "logs" / "plot_style.json").read_text(encoding="utf-8"))
    assert style["run_fingerprint"] == "synthetic-fingerprint"
    assert (output / "figures" / "README.md").is_file()


def test_interruption_is_not_reported_as_delivered_figures(tmp_path, monkeypatch):
    manifest = {"status": "interrupted", "saved_configurations": 3}
    monkeypatch.setattr(runner, "run_experiment", lambda **kwargs: manifest)
    assert cli.main(["--output-dir", str(tmp_path)]) is manifest
    assert not (tmp_path / "figures" / "README.md").exists()
