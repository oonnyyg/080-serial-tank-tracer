"""Command line entry: ``python -m tracer_train input.json --out dir``."""

from __future__ import annotations

import argparse
import sys

from .model import ScenarioError, load_scenarios
from .report import write_outputs
from .solver import simulate


def run(input_path: str, output_dir: str) -> int:
    scenarios = load_scenarios(input_path)

    results = [simulate(scenario) for scenario in scenarios]

    safe_names: set[str] = set()
    for result in results:
        safe_name = (
            result.scenario.name.strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace("\x00", "_")
        )
        if safe_name in safe_names:
            raise ScenarioError(
                f"情景名称 {result.scenario.name!r} 生成的输出文件名与其他情景冲突"
            )
        safe_names.add(safe_name)

    for result in results:
        paths = write_outputs(result, output_dir)
        print(f"情景 {result.scenario.name}: 已写出 {paths['csv']}, {paths['json']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tracer_train",
        description="串联全混池示踪分析 (后向欧拉, 标准库实现)",
    )
    parser.add_argument("input", help="输入 JSON 文件路径")
    parser.add_argument("--out", required=True, help="输出目录")
    args = parser.parse_args(argv)

    try:
        return run(args.input, args.out)
    except ScenarioError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
