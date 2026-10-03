"""Backward-Euler solver for tanks in series."""

from __future__ import annotations

from dataclasses import dataclass

from .model import Scenario

# Floating point tolerance for merging coincident grid boundaries.
GRID_EPS = 1e-12


@dataclass
class StepRecord:
    """Concentrations and cumulative mass flows at one grid boundary."""

    time: float
    concentrations: tuple[float, ...]
    cumulative_input: float
    cumulative_output: float


@dataclass
class SimulationResult:
    """Full trajectory plus mass-balance summary for one scenario."""

    scenario: Scenario
    records: list[StepRecord]
    initial_mass: float
    final_mass: float
    cumulative_input: float
    cumulative_output: float
    residual: float


def build_grid(scenario: Scenario) -> list[float]:
    """Sorted step boundaries: uniform grid union change points and end time.

    The grid starts at zero, advances by finite positive ``step_size`` while
    strictly below the end time, and always records the exact end time.
    Inlet change points are inserted as boundaries, so integration never
    strides across a change.
    """
    points: list[float] = [0.0]
    step = scenario.step_size
    end = scenario.end_time

    count = 0
    while True:
        value = step * count
        if value >= end:
            break
        points.append(value)
        count += 1
    points.append(end)

    for point in scenario.inlet_schedule:
        if point.time > 0.0:
            points.append(point.time)

    points.sort()
    merged: list[float] = []
    for value in points:
        if not merged or abs(value - merged[-1]) > GRID_EPS:
            merged.append(value)
    merged[0] = 0.0
    merged[-1] = end
    return merged


def _inlet_concentration(scenario: Scenario, time: float) -> float:
    """Schedule value valid at ``time`` (new value applies from its node on)."""
    value = scenario.inlet_schedule[0].concentration
    for point in scenario.inlet_schedule:
        if point.time <= time + GRID_EPS:
            value = point.concentration
        else:
            break
    return value


def simulate(scenario: Scenario) -> SimulationResult:
    """Advance every tank with coupled backward Euler over the grid."""
    grid = build_grid(scenario)
    flow = scenario.flow_rate
    volumes = scenario.volumes
    tank_count = len(volumes)

    current = list(scenario.initial_concentrations)
    initial_mass = sum(
        volume * concentration
        for volume, concentration in zip(volumes, current)
    )

    cumulative_input = 0.0
    cumulative_output = 0.0
    records = [
        StepRecord(0.0, tuple(current), 0.0, 0.0)
    ]

    for left, right in zip(grid, grid[1:]):
        dt = right - left
        inlet = _inlet_concentration(scenario, left)
        updated = [0.0] * tank_count
        for tank, volume in enumerate(volumes):
            upstream = inlet if tank == 0 else updated[tank - 1]
            updated[tank] = (
                current[tank] * volume + flow * dt * upstream
            ) / (volume + flow * dt)
        current = updated

        cumulative_input += flow * dt * inlet
        cumulative_output += flow * dt * current[-1]
        records.append(
            StepRecord(
                time=right,
                concentrations=tuple(current),
                cumulative_input=cumulative_input,
                cumulative_output=cumulative_output,
            )
        )

    final_mass = sum(
        volume * concentration
        for volume, concentration in zip(volumes, current)
    )
    residual = final_mass - initial_mass - cumulative_input + cumulative_output

    return SimulationResult(
        scenario=scenario,
        records=records,
        initial_mass=initial_mass,
        final_mass=final_mass,
        cumulative_input=cumulative_input,
        cumulative_output=cumulative_output,
        residual=residual,
    )
