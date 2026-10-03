"""命令行入口：``python -m tracer_train 输入.json --out 输出目录``。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .model import SimulationError, simulate_scenario


def _scenario_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return cleaned or "scenario"


def load_input(path: Path) -> list:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SimulationError(f"无法读取输入文件 {path}：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise SimulationError(f"输入文件不是合法 JSON（{path}:{exc.lineno}）：{exc.msg}") from exc

    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    else:
        scenarios = data
    if not isinstance(scenarios, list) or not scenarios:
        raise SimulationError("输入必须包含非空的情景数组（顶层数组或 {\"scenarios\": [...]}）")
    return scenarios


def write_results(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    summaries = []

    for result in results:
        base = _scenario_filename(result["name"])
        filename = f"{base}.csv"
        suffix = 1
        while filename in used_names:
            suffix += 1
            filename = f"{base}_{suffix}.csv"
        used_names.add(filename)

        csv_path = out_dir / filename
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_s"]
                + [f"tank_{i + 1}_concentration" for i in range(len(result["volumes"]))]
            )
            for row in result["rows"]:
                writer.writerow([repr(row["time"])] + [repr(c) for c in row["concentrations"]])

        summaries.append(
            {
                "name": result["name"],
                "csv_file": filename,
                "volumes": result["volumes"],
                "initial_concentrations": result["initial_concentrations"],
                "flow_rate": result["flow_rate"],
                "end_time_s": result["end_time"],
                "step_size_s": result["step_size"],
                "inlet_schedule": result["inlet_schedule"],
                "recorded_step_count": len(result["rows"]) - 1,
                "mass_balance": result["mass_balance"],
            }
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"scenarios": summaries}, handle, ensure_ascii=False, indent=2)


def run(input_path: Path, out_dir: Path) -> int:
    raw_scenarios = load_input(input_path)

    # 先整批校验/计算：任何情景失败都不写出成功文件。
    results = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_scenarios):
        result = simulate_scenario(raw)
        if result["name"] in seen_names:
            raise SimulationError(f"情景 {result['name']!r}：名称不唯一")
        seen_names.add(result["name"])
        results.append(result)

    write_results(results, out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tracer_train",
        description="串联全混池示踪分析（后向欧拉，标准库实现）",
    )
    parser.add_argument("input", type=Path, help="输入 JSON 文件")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    args = parser.parse_args(argv)

    try:
        return run(args.input, args.out)
    except SimulationError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
