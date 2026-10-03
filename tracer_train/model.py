"""串联全混池模型：输入校验、后向欧拉推进与质量收支。

输入结构（单个情景）::

    {
      "name": "情景名",
      "volumes": [100.0, 100.0],          # 各池体积（水流顺序）
      "initial_concentrations": [0.0, 0.0],
      "flow_rate": 5.0,                    # 恒定流量
      "end_time": 60.0,                    # 终止时刻（秒）
      "inlet_schedule": [[0.0, 10.0], [20.0, 0.0]],  # [时刻, 入口浓度]
      "step_size": 1.0                     # 可选，缺省 1.0
    }

单位由输入自行约定，全程保持一致：
浓度 C [质量/体积]，流量 Q [体积/时间]，体积 V [体积]，质量 V*C [质量]。
"""

from __future__ import annotations

import math
from typing import Any


class SimulationError(ValueError):
    """非法输入情景；消息中包含情景定位与具体原因。"""


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_scenario(raw: Any) -> dict:
    """校验并规范化单个情景字典。

    非法时抛出 :class:`SimulationError`，错误消息包含定位信息。
    """
    if not isinstance(raw, dict):
        raise SimulationError("情景必须是 JSON 对象")

    label = raw.get("name")
    if not isinstance(label, str) or not label.strip():
        raise SimulationError("情景缺少非空字符串字段 name")
    loc = f"情景 {label!r}"

    def fail(field: str, reason: str) -> None:
        raise SimulationError(f"{loc}：{field} {reason}")

    volumes = raw.get("volumes")
    if not isinstance(volumes, list) or not volumes:
        fail("volumes", "必须是非空数组")
    for index, volume in enumerate(volumes):
        if not _is_finite_number(volume) or volume <= 0:
            fail(f"volumes[{index}]", "必须是有限正数")

    tank_count = len(volumes)
    initial = raw.get("initial_concentrations")
    if not isinstance(initial, list) or len(initial) != tank_count:
        fail("initial_concentrations", f"必须是长度为 {tank_count} 的数组")
    for index, concentration in enumerate(initial):
        if not _is_finite_number(concentration) or concentration < 0:
            fail(f"initial_concentrations[{index}]", "必须是有限非负数")

    flow_rate = raw.get("flow_rate")
    if not _is_finite_number(flow_rate) or flow_rate <= 0:
        fail("flow_rate", "必须是有限正数")

    end_time = raw.get("end_time")
    if not _is_finite_number(end_time) or end_time <= 0:
        fail("end_time", "必须是有限正数（秒）")

    schedule = raw.get("inlet_schedule")
    if not isinstance(schedule, list) or not schedule:
        fail("inlet_schedule", "必须是非空数组，元素为 [时刻, 浓度]")
    parsed_schedule: list[tuple[float, float]] = []
    previous_time: float | None = None
    for index, node in enumerate(schedule):
        if (
            not isinstance(node, list)
            or len(node) != 2
            or not _is_finite_number(node[0])
            or not _is_finite_number(node[1])
        ):
            fail(f"inlet_schedule[{index}]", "必须是 [时刻, 浓度]，且均为有限数")
        node_time, node_concentration = float(node[0]), float(node[1])
        if node_concentration < 0:
            fail(f"inlet_schedule[{index}] 浓度", "必须为非负数")
        if index == 0:
            if node_time != 0.0:
                fail("inlet_schedule[0] 时刻", "必须从零开始（0）")
        elif node_time <= previous_time:
            fail(f"inlet_schedule[{index}] 时刻", "必须严格递增")
        if node_time > end_time:
            fail(f"inlet_schedule[{index}] 时刻", "不能超过终止时刻")
        previous_time = node_time
        parsed_schedule.append((node_time, node_concentration))

    step_size = raw.get("step_size", 1.0)
    if not _is_finite_number(step_size) or step_size <= 0:
        fail("step_size", "必须是有限正数（秒）")

    return {
        "name": label.strip(),
        "volumes": [float(v) for v in volumes],
        "initial_concentrations": [float(c) for c in initial],
        "flow_rate": float(flow_rate),
        "end_time": float(end_time),
        "inlet_schedule": parsed_schedule,
        "step_size": float(step_size),
    }


def build_time_grid(scenario: dict) -> list[float]:
    """构造步边界：零、正步长网格、入口变化点与终止时刻的并集。"""
    end_time = scenario["end_time"]
    step_size = scenario["step_size"]

    points = {0.0, end_time}
    points.update(node_time for node_time, _ in scenario["inlet_schedule"])
    steps = int(end_time // step_size)
    for index in range(1, steps + 1):
        point = index * step_size
        if point < end_time:
            points.add(point)

    grid = sorted(points)
    deduped = [grid[0]]
    for point in grid[1:]:
        if point - deduped[-1] > 1e-12:
            deduped.append(point)
    if deduped[-1] != end_time:
        deduped.append(end_time)
    return deduped


def inlet_concentration_at(scenario: dict, time: float) -> float:
    """返回给定时间所在区间的入口浓度（取不晚于该时刻的最后一个节点值）。"""
    result = scenario["inlet_schedule"][0][1]
    for node_time, node_concentration in scenario["inlet_schedule"]:
        if node_time <= time + 1e-12:
            result = node_concentration
        else:
            break
    return result


def simulate_scenario(raw: Any) -> dict:
    """校验并推进单个情景，返回轨迹与质量收支结果。"""
    scenario = validate_scenario(raw)
    volumes = scenario["volumes"]
    flow_rate = scenario["flow_rate"]
    tank_count = len(volumes)
    grid = build_time_grid(scenario)

    concentrations = list(scenario["initial_concentrations"])
    rows: list[dict] = [
        {"time": 0.0, "concentrations": list(concentrations)}
    ]

    total_in = 0.0   # 累计入口输入质量 = Σ Q·Δt·C_in(区间)
    total_out = 0.0  # 累计流出质量     = Σ Q·Δt·C_last(本步末)

    for step_index in range(1, len(grid)):
        time_start = grid[step_index - 1]
        time_end = grid[step_index]
        delta_t = time_end - time_start
        c_in = inlet_concentration_at(scenario, time_start)

        upstream = c_in
        new_concentrations: list[float] = [0.0] * tank_count
        for tank_index in range(tank_count):
            # 后向欧拉：V_i (C_i^n - C_i^{n-1}) / Δt
            #          = Q (C_{i-1}^n - C_i^n)
            alpha = flow_rate * delta_t / volumes[tank_index]
            new_concentrations[tank_index] = (
                concentrations[tank_index] + alpha * upstream
            ) / (1.0 + alpha)
            upstream = new_concentrations[tank_index]

        concentrations = new_concentrations
        total_in += flow_rate * delta_t * c_in
        total_out += flow_rate * delta_t * concentrations[-1]
        rows.append({"time": time_end, "concentrations": list(concentrations)})

    initial_mass = sum(
        volume * concentration
        for volume, concentration in zip(volumes, scenario["initial_concentrations"])
    )
    final_mass = sum(
        volume * concentration
        for volume, concentration in zip(volumes, concentrations)
    )
    residual = initial_mass + total_in - total_out - final_mass

    return {
        "name": scenario["name"],
        "flow_rate": flow_rate,
        "volumes": volumes,
        "initial_concentrations": scenario["initial_concentrations"],
        "end_time": scenario["end_time"],
        "step_size": scenario["step_size"],
        "inlet_schedule": [list(node) for node in scenario["inlet_schedule"]],
        "time_grid": grid,
        "rows": rows,
        "mass_balance": {
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "cumulative_input": total_in,
            "cumulative_output": total_out,
            "residual": residual,
        },
    }
