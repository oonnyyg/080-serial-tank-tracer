import json
import subprocess
import sys

import pytest


def write_input(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def scenario(name, **overrides):
    data = {
        "name": name,
        "volumes": [50.0, 75.0],
        "initial_concentrations": [0.0, 0.0],
        "flow_rate": 5.0,
        "end_time": 20.0,
        "step_size": 2.0,
        "inlet_schedule": [[0, 1.0], [6, 0.0]],
    }
    data.update(overrides)
    return data


def run_cli(input_path, output_dir):
    return subprocess.run(
        [sys.executable, "-m", "tracer_train", str(input_path), "--out", str(output_dir)],
        capture_output=True,
        text=True,
    )


def test_cli_writes_csv_and_json(tmp_path):
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "out"
    write_input(input_path, {"scenarios": [scenario("a"), scenario("b")]})

    completed = run_cli(input_path, output_dir)
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "a_steps.csv").exists()
    assert (output_dir / "a_summary.json").exists()
    assert (output_dir / "b_steps.csv").exists()
    assert (output_dir / "b_summary.json").exists()

    summary = json.loads((output_dir / "a_summary.json").read_text(encoding="utf-8"))
    assert summary["cumulative_input_mass"] == pytest.approx(30.0)


def test_invalid_batch_writes_no_success_files(tmp_path):
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "out"
    write_input(
        input_path,
        {"scenarios": [scenario("good"), scenario("bad", flow_rate=-1)]},
    )

    completed = run_cli(input_path, output_dir)
    assert completed.returncode == 1
    assert "flow_rate" in completed.stderr
    assert not output_dir.exists() or not any(output_dir.iterdir())
