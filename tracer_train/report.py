"""CSV trajectory and JSON summary output."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .solver import SimulationResult


def render_step_csv(result: SimulationResult) -> str:
    """Initial state plus every step-end concentration row."""
    tank_count = len(result.scenario.volumes)
    header = [
        "time_s",
        *[f"c_tank_{idx + 1}" for idx in range(tank_count)],
        "cumulative_input_mass",
        "cumulative_output_mass",
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for record in result.records:
        writer.writerow(
            [
                repr(record.time),
                *[repr(value) for value in record.concentrations],
                repr(record.cumulative_input),
                repr(record.cumulative_output),
            ]
        )
    return buffer.getvalue()


def build_summary(result: SimulationResult) -> dict:
    scenario = result.scenario
    return {
        "name": scenario.name,
        "flow_rate": scenario.flow_rate,
        "end_time_s": scenario.end_time,
        "step_size_s": scenario.step_size,
        "tank_volumes": list(scenario.volumes),
        "initial_concentrations": list(scenario.initial_concentrations),
        "grid_boundaries": [record.time for record in result.records],
        "initial_mass": result.initial_mass,
        "final_mass": result.final_mass,
        "cumulative_input_mass": result.cumulative_input,
        "cumulative_output_mass": result.cumulative_output,
        "mass_balance_residual": result.residual,
        "mass_balance_formula": (
            "residual = final_mass - initial_mass "
            "- cumulative_input_mass + cumulative_output_mass"
        ),
        "units": {
            "time": "s",
            "flow_rate": "volume/s (input volume unit per second)",
            "concentration": "mass/volume (input units preserved)",
            "mass": "input mass unit, no normalization applied",
        },
    }


def render_summary_json(result: SimulationResult) -> str:
    return json.dumps(build_summary(result), ensure_ascii=False, indent=2)


def write_outputs(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    """Write ``<name>_steps.csv`` and ``<name>_summary.json``; return paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = (
        result.scenario.name.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace("\x00", "_")
    )
    csv_path = directory / f"{safe_name}_steps.csv"
    json_path = directory / f"{safe_name}_summary.json"
    csv_path.write_text(render_step_csv(result), encoding="utf-8")
    json_path.write_text(render_summary_json(result), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}
