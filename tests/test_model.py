import math
import unittest

from tracer_train.model import (
    SimulationError,
    build_time_grid,
    simulate_scenario,
    validate_scenario,
)


def base_scenario(**overrides):
    scenario = {
        "name": "base",
        "volumes": [100.0, 100.0],
        "initial_concentrations": [0.0, 0.0],
        "flow_rate": 10.0,
        "end_time": 50.0,
        "inlet_schedule": [[0.0, 5.0], [20.0, 0.0]],
        "step_size": 5.0,
    }
    scenario.update(overrides)
    return scenario


class ValidationTests(unittest.TestCase):
    def test_accepts_valid_scenario(self):
        result = simulate_scenario(base_scenario())
        self.assertEqual(result["name"], "base")

    def test_rejects_bad_volume(self):
        with self.assertRaisesRegex(SimulationError, "volumes\[1\]"):
            simulate_scenario(base_scenario(volumes=[100.0, 0.0]))
        with self.assertRaisesRegex(SimulationError, "有限正数"):
            simulate_scenario(base_scenario(volumes=[100.0, math.inf]))
        with self.assertRaisesRegex(SimulationError, "volumes\[0\]"):
            simulate_scenario(base_scenario(volumes=[-1.0]))

    def test_rejects_negative_concentration(self):
        with self.assertRaisesRegex(SimulationError, "initial_concentrations\[0\]"):
            simulate_scenario(base_scenario(initial_concentrations=[-0.1, 0.0]))
        with self.assertRaisesRegex(SimulationError, "inlet_schedule"):
            simulate_scenario(
                base_scenario(inlet_schedule=[[0.0, -1.0]])
            )

    def test_rejects_bad_flow_and_end_time(self):
        with self.assertRaisesRegex(SimulationError, "flow_rate"):
            simulate_scenario(base_scenario(flow_rate=0.0))
        with self.assertRaisesRegex(SimulationError, "end_time"):
            simulate_scenario(base_scenario(end_time=-3.0))

    def test_rejects_bad_schedule(self):
        with self.assertRaisesRegex(SimulationError, "从零开始"):
            simulate_scenario(base_scenario(inlet_schedule=[[1.0, 5.0]]))
        with self.assertRaisesRegex(SimulationError, "严格递增"):
            simulate_scenario(
                base_scenario(inlet_schedule=[[0.0, 5.0], [20.0, 1.0], [20.0, 2.0]])
            )
        with self.assertRaisesRegex(SimulationError, "不能超过终止时刻"):
            simulate_scenario(
                base_scenario(inlet_schedule=[[0.0, 5.0], [51.0, 0.0]])
            )

    def test_rejects_bad_step_size(self):
        with self.assertRaisesRegex(SimulationError, "step_size"):
            simulate_scenario(base_scenario(step_size=0.0))

    def test_rejects_missing_or_duplicate_style_name(self):
        with self.assertRaisesRegex(SimulationError, "name"):
            simulate_scenario(base_scenario(name=""))


class GridTests(unittest.TestCase):
    def test_grid_is_union_of_points(self):
        grid = build_time_grid(validate_scenario(base_scenario()))
        # 0、5 的网格点、变化点 20、终止 50
        self.assertEqual(grid[0], 0.0)
        self.assertIn(20.0, grid)
        self.assertEqual(grid[-1], 50.0)
        for a, b in zip(grid, grid[1:]):
            self.assertLess(a, b)

    def test_end_time_always_recorded_even_without_grid_point(self):
        scenario = base_scenario(end_time=23.0, step_size=5.0)
        result = simulate_scenario(scenario)
        self.assertAlmostEqual(result["time_grid"][-1], 23.0)
        self.assertAlmostEqual(result["rows"][-1]["time"], 23.0)


class SimulationTests(unittest.TestCase):
    def test_initial_row_is_initial_state(self):
        result = simulate_scenario(
            base_scenario(initial_concentrations=[2.0, 3.0])
        )
        self.assertEqual(result["rows"][0]["time"], 0.0)
        self.assertEqual(result["rows"][0]["concentrations"], [2.0, 3.0])

    def test_single_tank_step_matches_backward_euler_formula(self):
        # V=100, Q=10, Δt=5, C0=0, Cin=5：α = 0.5；C1 = 0.5*5/1.5
        scenario = {
            "name": "one",
            "volumes": [100.0],
            "initial_concentrations": [0.0],
            "flow_rate": 10.0,
            "end_time": 5.0,
            "inlet_schedule": [[0.0, 5.0]],
            "step_size": 5.0,
        }
        result = simulate_scenario(scenario)
        self.assertAlmostEqual(result["rows"][1]["concentrations"][0], 0.5 * 5.0 / 1.5)

    def test_tanks_are_coupled_using_upstream_end_of_step(self):
        # 两池同参数：首池末值 x = α*Cin/(1+α)；
        # 次池末值 y = α*x/(1+α)（初始均为 0）。
        scenario = {
            "name": "two",
            "volumes": [100.0, 100.0],
            "initial_concentrations": [0.0, 0.0],
            "flow_rate": 10.0,
            "end_time": 5.0,
            "inlet_schedule": [[0.0, 6.0]],
            "step_size": 5.0,
        }
        result = simulate_scenario(scenario)
        x = 0.5 * 6.0 / 1.5
        y = 0.5 * x / 1.5
        c1, c2 = result["rows"][1]["concentrations"]
        self.assertAlmostEqual(c1, x)
        self.assertAlmostEqual(c2, y)

    def test_mass_balance_residual_small(self):
        result = simulate_scenario(base_scenario())
        mb = result["mass_balance"]
        scale = mb["initial_mass"] + mb["cumulative_input"] + 1.0
        self.assertAlmostEqual(mb["residual"], 0.0, delta=1e-9 * scale)

    def test_discrete_balance_is_exact(self):
        # 后向欧拉按本步末浓度记账，离散残差应接近机器精度。
        result = simulate_scenario(
            base_scenario(
                volumes=[37.0, 88.0, 12.5],
                initial_concentrations=[1.5, 0.2, 7.0],
                flow_rate=4.2,
                end_time=123.4,
                inlet_schedule=[[0.0, 9.0], [13.7, 2.5], [60.0, 11.0]],
                step_size=1.23,
            )
        )
        mb = result["mass_balance"]
        scale = mb["initial_mass"] + mb["cumulative_input"]
        self.assertAlmostEqual(mb["residual"], 0.0, delta=1e-10 * max(scale, 1.0))


if __name__ == "__main__":
    unittest.main()
