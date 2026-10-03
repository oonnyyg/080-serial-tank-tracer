"""Serial continuously-stirred tank tracer analysis.

Tanks-in-series model solved with backward Euler, plus an exact-on-grid
tracer mass balance. Standard library only.
"""

from .model import Scenario, ScenarioError, load_scenarios
from .solver import SimulationResult, StepRecord, simulate

__all__ = [
    "Scenario",
    "ScenarioError",
    "SimulationResult",
    "StepRecord",
    "load_scenarios",
    "simulate",
]
