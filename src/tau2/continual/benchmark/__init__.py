"""
Benchmark module for continual learning evaluation.

This module provides the core benchmark framework that is
SEPARATE from any specific learning algorithm.
"""

from tau2.continual.benchmark.agent_interface import (
    ContinualAgent,
    Experience,
    AgentResponse,
    DummyAgent,
)
from tau2.continual.benchmark.benchmark import ContinualBenchmark
from tau2.continual.benchmark.protocol import (
    EvaluationProtocol,
    ProtocolConfig,
    PhaseConfig,
    EvaluationPhase,
)
from tau2.continual.benchmark.metrics import (
    ContinualMetrics,
    MetricsComputer,
)

__all__ = [
    # Agent interface
    "ContinualAgent",
    "Experience",
    "AgentResponse",
    "DummyAgent",
    # Benchmark
    "ContinualBenchmark",
    # Protocol
    "EvaluationProtocol",
    "ProtocolConfig",
    "PhaseConfig",
    "EvaluationPhase",
    # Metrics
    "ContinualMetrics",
    "MetricsComputer",
]
