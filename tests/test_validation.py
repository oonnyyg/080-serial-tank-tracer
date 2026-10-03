import json

import pytest

from tracer_train.model import ScenarioError, load_scenarios


def base_scenario(**overrides):
    scenario = {
        "name": "ok",
        "volumes": [10.0, 20.0],
        "initial_concentrations": [1.0, 0.0],
        "flow_rate": 2.0,
        "end_time": 30.0,
        "step_size": 5.0,
        "inlet_schedule": [[0, 1.0], [10, 0.0]],
    }
    scenario.update(overrides)
    return scenario


def write_input(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_single_scenario_object(tmp_path):
    path = write_input(tmp_path, base_scenario())
    scenarios = load_scenarios(path)
    assert len(scenarios) == 1
    assert scenarios[0].name == "ok"


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"volumes": []}, "volumes 必须是非空数组"),
        ({"volumes": [10, 0]}, "必须为正数"),
        ({"volumes": [10, float("nan")]}, "有限数值"),
        ({"initial_concentrations": [1.0]}, "长度"),
        ({"initial_concentrations": [1.0, -0.1]}, "不能为负"),
        ({"flow_rate": 0}, "必须为正数"),
        ({"end_time": -1}, "有限正数"),
        ({"step_size": 0}, "必须是有限正数"),
        ({"inlet_schedule": [[1, 1]]}, "必须从零开始"),
        ({"inlet_schedule": [[0, 1], [5, 1], [5, 2]]}, "严格递增"),
        ({"inlet_schedule": [[0, 1], [40, 0]]}, "超过终止时刻"),
        ({"inlet_schedule": [[0, -1]]}, "不能为负"),
    ],
)
def test_invalid_fields_are_reported(tmp_path, change, fragment):
    path = write_input(tmp_path, base_scenario(**change))
    with pytest.raises(ScenarioError, match=fragment):
        load_scenarios(path)


def test_duplicate_names_fail_as_a_batch(tmp_path):
    payload = {
        "scenarios": [base_scenario(name="same"), base_scenario(name="same ")]
    }
    path = write_input(tmp_path, payload)
    with pytest.raises(ScenarioError, match="名称重复"):
        load_scenarios(path)


def test_invalid_json_message(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScenarioError, match="合法 JSON"):
        load_scenarios(path)
