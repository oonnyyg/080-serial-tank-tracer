import csv
import json
import tempfile
import unittest
from pathlib import Path

from tracer_train.cli import main


SCENARIO_A = {
    "name": "pulse",
    "volumes": [100.0, 100.0],
    "initial_concentrations": [0.0, 0.0],
    "flow_rate": 10.0,
    "end_time": 20.0,
    "inlet_schedule": [[0.0, 8.0], [5.0, 0.0]],
    "step_size": 2.0,
}


class CliTests(unittest.TestCase):
    def test_writes_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.json"
            out_dir = tmp_path / "out"
            input_path.write_text(json.dumps([SCENARIO_A]), encoding="utf-8")

            code = main([str(input_path), "--out", str(out_dir)])
            self.assertEqual(code, 0)

            csv_path = out_dir / "pulse.csv"
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["time_s", "tank_1_concentration", "tank_2_concentration"])
            self.assertAlmostEqual(float(rows[1][0]), 0.0)
            self.assertAlmostEqual(float(rows[-1][0]), 20.0)
            self.assertAlmostEqual(float(rows[2][0]), 2.0)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["scenarios"]), 1)
            mb = summary["scenarios"][0]["mass_balance"]
            self.assertIn("residual", mb)
            self.assertIn("initial_mass", mb)

    def test_batch_failure_writes_nothing(self):
        bad = dict(SCENARIO_A, name="bad", volumes=[0.0, 100.0])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.json"
            out_dir = tmp_path / "out"
            input_path.write_text(json.dumps([SCENARIO_A, bad]), encoding="utf-8")

            code = main([str(input_path), "--out", str(out_dir)])
            self.assertEqual(code, 2)
            self.assertFalse(out_dir.exists())

    def test_duplicate_names_fail_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.json"
            out_dir = tmp_path / "out"
            input_path.write_text(json.dumps([SCENARIO_A, dict(SCENARIO_A)]), encoding="utf-8")

            code = main([str(input_path), "--out", str(out_dir)])
            self.assertEqual(code, 2)
            self.assertFalse(out_dir.exists())

    def test_multiple_scenarios_named_files(self):
        other = dict(SCENARIO_A, name="step", flow_rate=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "in.json"
            out_dir = tmp_path / "out"
            input_path.write_text(
                json.dumps({"scenarios": [SCENARIO_A, other]}), encoding="utf-8"
            )

            code = main([str(input_path), "--out", str(out_dir)])
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "pulse.csv").exists())
            self.assertTrue((out_dir / "step.csv").exists())


if __name__ == "__main__":
    unittest.main()
