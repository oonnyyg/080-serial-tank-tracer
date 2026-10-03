"""串联全混池示踪分析工具（仅标准库）。"""

__all__ = ["validate_scenario", "simulate_scenario", "SimulationError"]

from .model import SimulationError, simulate_scenario, validate_scenario
