"""PyCharm right-click Run: all 45 paired core-overhead configurations."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "02_overhead"
EXPECTED_FIGURE_STEMS = ("Fig_O1_time_overhead", "Fig_O2_communication_overhead")
# Imports may read dataset source but must never create __pycache__ there.
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TNSE Group 02: computation and communication overhead")
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT,
                        help="Optional result root; default is resolved from this script")
    parser.add_argument("--plots-only", action="store_true",
                        help="Redraw saved figures; an encoded communication generation redraws only O2")
    parser.add_argument("--communication-only", action="store_true",
                        help="Replay encoded communication and redraw only O2; preserve saved time results")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume matching completed checkpoints (also the default)")
    parser.add_argument("--force", action="store_true",
                        help="Create a fresh shared calibration and recompute all 45 configurations")
    return parser.parse_args(argv)


def complete_outputs(output_dir, metadata, *, plots_only=False):
    """Verify this generation's four figures and publish their exact locations.

    The simulator owns numerical results and the plotter owns figure contents.
    This final check prevents an empty figure list or a missing/empty saved
    file from being reported as a successfully delivered pair of figures.
    """
    output = Path(output_dir).resolve()
    folder = output / "figures"
    declared = metadata.get("figures_generated", [])
    if (not isinstance(declared, (list, tuple))
            or len(declared) != len(EXPECTED_FIGURE_STEMS)
            or set(declared) != set(EXPECTED_FIGURE_STEMS)):
        raise RuntimeError(
            "未生成完整的时间/通信两张结果图；不能报告绘图成功。 "
            f"Expected O1/O2 figures at {folder}; generated={declared}. "
            "请查看 summary 与 logs 中的配对数据或绘图错误。"
        )
    paths = [folder / f"{stem}.{extension}"
             for stem in EXPECTED_FIGURE_STEMS for extension in ("pdf", "png")]
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "结果图文件缺失或为空；不能报告绘图成功。 "
            "Missing or empty output files:\n" + "\n".join(str(path) for path in missing)
        )

    version = metadata.get("implementation_version",
                           metadata.get("results_implementation_version", "未记录"))
    fingerprint = metadata.get("run_fingerprint", "未记录")
    gallery = [
        "# 第二组时间与通信开销结果图", "",
        f"实现版本：`{version}`。运行身份：`{fingerprint}`。", "",
        "来源为第二组核心在线步骤实验：每个 N/d 点使用一个配对 seed，"
        "四种方法执行相同固定 B 个目标；没有多轮平均。", "",
        "该目录保存上述运行身份对应的当前结果；v2/v3采用不同计费口径，不能复用为v4测量。", "",
        f"[运行来源与计费配置](<{(output / 'logs' / 'run_manifest.json').as_posix()}>) · "
        f"[时间统计](<{(output / 'summary' / 'time_overhead_summary.csv').as_posix()}>) · "
        f"[通信统计](<{(output / 'summary' / 'communication_overhead_summary.csv').as_posix()}>)", "",
    ]
    for title, stem in zip(("时间开销", "通信开销"), EXPECTED_FIGURE_STEMS):
        pdf_path = (folder / f"{stem}.pdf").as_posix()
        png_path = (folder / f"{stem}.png").as_posix()
        gallery.extend([f"## {title}", "",
                        f"[PDF](<{pdf_path}>) · [PNG](<{png_path}>)", "",
                        f"![{title}](<{png_path}>)", ""])
    storage = importlib.import_module("experiments.02_overhead.overhead_storage")
    storage.atomic_write_text(folder / "README.md", "\n".join(gallery))
    if plots_only:
        print("本次仅根据已保存统计重绘，未执行新仿真。", flush=True)
    else:
        executed = int(metadata.get("executed_configurations_this_invocation", 0))
        resumed = int(metadata.get("resumed_configurations", 0))
        print(f"本次新运行 {executed} 个配置；验证并恢复缓存 {resumed} 个配置。", flush=True)
        if executed == 0 and resumed:
            print("全部配置命中缓存：本次用时仅包含缓存核验、导出和绘图；未重新仿真。", flush=True)
    print(f"结果图目录：{folder}", flush=True)
    for path in paths:
        print(str(path), flush=True)
    print(f"图片预览索引：{folder / 'README.md'}", flush=True)
    return paths


def _has_complete_saved_run(output):
    """An existing measured generation defaults to communication-only work."""
    path = output/'logs'/'run_manifest.json'
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return False
    return (manifest.get('status')=='complete'
            and manifest.get('implementation_version')=='overhead-core-v4'
            and manifest.get('derived_outputs',{}).get('state')=='complete'
            and all((output/relative).is_file() for relative in (
                'raw/overhead_runs.csv','summary/time_overhead_summary.csv',
                'summary/communication_overhead_summary.csv',
                'figures/Fig_O1_time_overhead.pdf','figures/Fig_O1_time_overhead.png')))


def _communication_outputs(output, *, replay=True):
    """Commit O2's independent provenance without touching the O1 generation."""
    overlay = importlib.import_module('experiments.02_overhead.overhead_communication')
    plotter = importlib.import_module('experiments.02_overhead.plot_overhead')
    storage = importlib.import_module('experiments.02_overhead.overhead_storage')
    with storage.output_directory_lock(output):
        if replay:
            overlay.replay_communication(output)
        metadata = plotter.plot_communication_results(output)
        storage.atomic_write_json(output/'logs'/'communication_plot_style.json',metadata)
        completed = overlay.finalize_communication(output,metadata)
        folder = output/'figures'
        gallery = ['# 第二组时间与通信开销结果图','',
            '时间图保留原v4实测结果；通信图采用判决等价的渐进描述符编码独立回放。',
            '通信计入精化请求、低位回复和原始回退；压缩CPU及精化时延未计入原时间图。','',
            f'时间来源：`{completed["base_run_fingerprint"]}`。',
            f'通信来源：`{completed["communication_fingerprint"]}`。','',
            f'[时间原运行清单](<{(output/"logs"/"run_manifest.json").as_posix()}>) · '
            f'[通信编码清单](<{(output/"logs"/"communication_manifest.json").as_posix()}>)','']
        for title, stem in zip(('时间开销','通信开销'),EXPECTED_FIGURE_STEMS):
            gallery.extend([f'## {title}','',
                f'[PDF](<{(folder/f"{stem}.pdf").as_posix()}>) · '
                f'[PNG](<{(folder/f"{stem}.png").as_posix()}>)','',
                f'![{title}](<{(folder/f"{stem}.png").as_posix()}>)',''])
        storage.atomic_write_text(folder/'README.md','\n'.join(gallery))
    print('仅更新通信编码与通信图；原时间统计、时间图及checkpoint逐字节保留。',flush=True)
    print(f'通信来源：{completed["communication_fingerprint"]}',flush=True)
    for extension in ('pdf','png'):
        print(str(output/'figures'/f'Fig_O2_communication_overhead.{extension}'),flush=True)
    return completed


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    try:
        if args.communication_only:
            if args.force or args.plots_only:
                raise SystemExit('--communication-only cannot be combined with --force or --plots-only.')
            return _communication_outputs(output)
        if args.plots_only:
            if args.force:
                raise SystemExit("--force and --plots-only describe different operations; use one.")
            if (output/'logs'/'communication_manifest.json').is_file():
                return _communication_outputs(output,replay=False)
            module = importlib.import_module("experiments.02_overhead.plot_overhead")
            storage=importlib.import_module('experiments.02_overhead.overhead_storage')
            with storage.output_directory_lock(output):
                metadata=module.plot_results(output)
                storage.atomic_write_json(output/'logs'/'plot_style.json',metadata)
                complete_outputs(output, metadata, plots_only=True)
            return metadata
        if not args.force and _has_complete_saved_run(output):
            return _communication_outputs(output)
        module = importlib.import_module("experiments.02_overhead.overhead_experiment")
        manifest = module.run_experiment(output_dir=output, resume=args.resume, force=args.force)
        if manifest.get("status") != "interrupted":
            complete_outputs(output, manifest)
            if _has_complete_saved_run(output):
                return _communication_outputs(output)
        return manifest
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing experiment dependency: {exc.name}. In this project's interpreter run "
            f'python -m pip install -r "{PROJECT_ROOT / "requirements.txt"}"'
        ) from exc


if __name__ == "__main__":
    main()
