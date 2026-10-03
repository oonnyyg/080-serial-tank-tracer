"""Input model and validation for serial-tank tracer scenarios."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


class ScenarioError(ValueError):
    """A scenario is invalid; message identifies the scenario and reason."""


@dataclass(frozen=True)
class InletPoint:
    """Inlet concentration becomes ``concentration`` at ``time`` seconds."""

    time: float
    concentration: float


@dataclass(frozen=True)
class Scenario:
    """One fully validated tanks-in-series scenario.

    Volumes are ordered along the water flow; every tank shares the same
    constant ``flow_rate``. Time is in seconds.
    """

    name: str
    volumes: tuple[float, ...]
    initial_concentrations: tuple[float, ...]
    flow_rate: float
    end_time: float
    step_size: float
    inlet_schedule: tuple[InletPoint, ...]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: object, label: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ScenarioError(f"{label} 必须是有限数值")
    return float(value)


def _validate_scenario(raw: object, index: int, seen_names: set[str]) -> Scenario:
    where = f"情景 #{index + 1}"
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: 必须是对象 (JSON object)")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{where}: name 必须是非空字符串")
    name = name.strip()
    if name in seen_names:
        raise ScenarioError(f"情景名称重复: {name!r}")
    where = f"情景 {name!r}"
    seen_names.add(name)

    volumes_raw = raw.get("volumes")
    if not isinstance(volumes_raw, list) or not volumes_raw:
        raise ScenarioError(f"{where}: volumes 必须是非空数组")
    volumes: list[float] = []
    for idx, value in enumerate(volumes_raw):
        volume = _finite_number(value, f"{where}: volumes[{idx}]")
        if volume <= 0.0:
            raise ScenarioError(
                f"{where}: volumes[{idx}] 必须为正数, 实际为 {volume}"
            )
        volumes.append(volume)

    initial_raw = raw.get("initial_concentrations")
    if not isinstance(initial_raw, list):
        raise ScenarioError(f"{where}: initial_concentrations 必须是数组")
    if len(initial_raw) != len(volumes):
        raise ScenarioError(
            f"{where}: initial_concentrations 长度 ({len(initial_raw)}) "
            f"必须与 volumes 长度 ({len(volumes)}) 相同"
        )
    initial: list[float] = []
    for idx, value in enumerate(initial_raw):
        concentration = _finite_number(
            value, f"{where}: initial_concentrations[{idx}]"
        )
        if concentration < 0.0:
            raise ScenarioError(
                f"{where}: initial_concentrations[{idx}] 不能为负, "
                f"实际为 {concentration}"
            )
        initial.append(concentration)

    flow_rate = _finite_number(raw.get("flow_rate"), f"{where}: flow_rate")
    if flow_rate <= 0.0:
        raise ScenarioError(
            f"{where}: flow_rate 必须为正数, 实际为 {flow_rate}"
        )

    end_time = _finite_number(raw.get("end_time"), f"{where}: end_time")
    if end_time <= 0.0:
        raise ScenarioError(
            f"{where}: end_time 必须是有限正数, 实际为 {end_time}"
        )

    step_size = _finite_number(raw.get("step_size"), f"{where}: step_size")
    if step_size <= 0.0:
        raise ScenarioError(
            f"{where}: step_size 必须是有限正数, 实际为 {step_size}"
        )

    schedule_raw = raw.get("inlet_schedule")
    if not isinstance(schedule_raw, list) or not schedule_raw:
        raise ScenarioError(f"{where}: inlet_schedule 必须是非空数组")
    schedule: list[InletPoint] = []
    previous_time = -math.inf
    for idx, point in enumerate(schedule_raw):
        label = f"{where}: inlet_schedule[{idx}]"
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ScenarioError(f"{label} 必须是含两个元素的数组 [time, concentration]")
        point_time = _finite_number(point[0], f"{label} 的 time")
        concentration = _finite_number(point[1], f"{label} 的 concentration")
        if concentration < 0.0:
            raise ScenarioError(
                f"{label} 的 concentration 不能为负, 实际为 {concentration}"
            )
        if idx == 0:
            if point_time != 0.0:
                raise ScenarioError(
                    f"{label}: 入口时间表必须从零开始, 实际起点为 {point_time}"
                )
        elif point_time <= previous_time:
            raise ScenarioError(
                f"{label}: 时间节点必须严格递增, "
                f"{point_time} 未晚于前一节点 {previous_time}"
            )
        if point_time > end_time:
            raise ScenarioError(
                f"{label}: 时间节点 {point_time} 超过终止时刻 {end_time}"
            )
        previous_time = point_time
        schedule.append(InletPoint(point_time, concentration))

    return Scenario(
        name=name,
        volumes=tuple(volumes),
        initial_concentrations=tuple(initial),
        flow_rate=flow_rate,
        end_time=end_time,
        step_size=step_size,
        inlet_schedule=tuple(schedule),
    )


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load and validate every scenario from a JSON file.

    Raises ScenarioError (with the offending scenario identified) for any
    invalid input; callers must treat that as a whole-batch failure.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ScenarioError(f"无法读取输入文件 {source}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"输入文件不是合法 JSON: {exc}") from exc

    if isinstance(data, dict) and "scenarios" in data:
        raw_scenarios = data["scenarios"]
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ScenarioError("scenarios 必须是非空数组")
    elif isinstance(data, dict) and "volumes" in data:
        raw_scenarios = [data]
    else:
        raise ScenarioError(
            "输入必须是单个情景对象, 或含非空 \"scenarios\" 数组的对象"
        )

    seen: set[str] = set()
    scenarios = [
        _validate_scenario(raw, idx, seen)
        for idx, raw in enumerate(raw_scenarios)
    ]
    return scenarios
