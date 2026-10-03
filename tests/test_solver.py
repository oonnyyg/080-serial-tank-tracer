import math

from tracer_train.model import Scenario, InletPoint
from tracer_train.solver import build_grid, simulate


def make_scenario(**overrides):
    values = dict(
        name="s",
        volumes=(100.0, 100.0),
        initial_concentrations=(0.0, 0.0),
        flow_rate=10.0,
        end_time=30.0,
        step_size=10.0,
        inlet_schedule=(InletPoint(0.0, 2.0),),
    )
    values.update(overrides)
    return Scenario(**values)


def test_grid_includes_change_points_and_exact_end():
    scenario = make_scenario(
        step_size=7.0,
        inlet_schedule=(InletPoint(0.0, 1.0), InletPoint(5.0, 0.0)),
    )
    grid = build_grid(scenario)
    assert grid[0] == 0.0
    assert math.isclose(grid[-1], 30.0)
    assert 5.0 in grid
    assert all(right > left for left, right in zip(grid, grid[1:]))


def test_change_point_never_straddled():
    scenario = make_scenario(
        step_size=7.0,
        inlet_schedule=(InletPoint(0.0, 2.0), InletPoint(8.0, 0.0)),
    )
    grid = build_grid(scenario)
    assert 8.0 in grid


def test_constant_inlet_single_tank_backward_euler_value():
    scenario = make_scenario(
        volumes=(100.0,),
        initial_concentrations=(0.0,),
        end_time=10.0,
        step_size=10.0,
        inlet_schedule=(InletPoint(0.0, 1.0),),
    )
    result = simulate(scenario)
    # c_new = Q*dt / (V + Q*dt) * c_in = 100/200 * 1
    assert math.isclose(result.records[-1].concentrations[0], 0.5)


def test_coupled_advance_uses_upstream_step_end_value():
    scenario = make_scenario(
        volumes=(100.0, 100.0),
        initial_concentrations=(0.0, 0.0),
        end_time=10.0,
        step_size=10.0,
        inlet_schedule=(InletPoint(0.0, 2.0),),
    )
    result = simulate(scenario)
    c1 = 100.0 * 2.0 / 200.0
    c2 = (10.0 * 10.0 * c1) / (100.0 + 100.0)
    final = result.records[-1].concentrations
    assert math.isclose(final[0], c1)
    assert math.isclose(final[1], c2)


def test_initial_mass_present_is_accounted_for():
    scenario = make_scenario(
        volumes=(100.0, 50.0),
        initial_concentrations=(3.0, 2.0),
        inlet_schedule=(InletPoint(0.0, 0.0),),
    )
    result = simulate(scenario)
    assert math.isclose(result.initial_mass, 400.0)


def test_mass_balance_residual_closes_on_each_grid():
    scenario = make_scenario(
        volumes=(80.0, 120.0, 60.0),
        initial_concentrations=(1.0, 0.5, 0.2),
        step_size=3.0,
        inlet_schedule=(InletPoint(0.0, 2.0), InletPoint(7.0, 0.5)),
    )
    result = simulate(scenario)
    assert math.isclose(result.residual, 0.0, abs_tol=1e-9)
    assert math.isclose(result.records[-1].time, 30.0)


def test_step_size_changes_numerical_result_but_not_balance():
    coarse = simulate(make_scenario(step_size=10.0))
    fine = simulate(make_scenario(step_size=1.0))
    assert not math.isclose(
        coarse.records[-1].concentrations[1],
        fine.records[-1].concentrations[1],
    )
    assert math.isclose(coarse.residual, 0.0, abs_tol=1e-9)
    assert math.isclose(fine.residual, 0.0, abs_tol=1e-9)
