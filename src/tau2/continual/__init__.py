"""
Continual Learning Module for τ²-Bench

This module provides tools for evaluating agent tool-use capabilities
in a continual learning setting.

Architecture:
- benchmark/: Core benchmark framework (agent interface, protocol, metrics)
- baselines/: Example baseline implementations
- curriculum/: Curriculum and stage definitions
- evaluation/: Evaluation tools and metrics
- training/: Training utilities (data conversion, etc.)
"""

# Curriculum components
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.curriculum.stage import LearningStage, LearningMaterial

# Benchmark framework (algorithm-agnostic)
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
    ContinualMetrics as BenchmarkMetrics,
    MetricsComputer,
)

# Evaluation tools
from tau2.continual.evaluation.metrics import ContinualMetrics
from tau2.continual.evaluation.evaluator import ContinualLearningEvaluator
from tau2.continual.evaluation.tool_analysis import (
    ToolCallEvaluation,
    ToolPerformanceTracker,
)

# Baselines (example implementations)
from tau2.continual.baselines.vanilla_finetune import VanillaFinetuneAgent
from tau2.continual.baselines.icl_baseline import ICLBaselineAgent

# Runner
from tau2.continual.runner import ContinualBenchmarkRunner

__all__ = [
    # Curriculum
    "Curriculum",
    "LearningStage",
    "LearningMaterial",
    # Benchmark framework
    "ContinualAgent",
    "Experience",
    "AgentResponse",
    "DummyAgent",
    "ContinualBenchmark",
    "EvaluationProtocol",
    "ProtocolConfig",
    "PhaseConfig",
    "EvaluationPhase",
    "BenchmarkMetrics",
    "MetricsComputer",
    # Evaluation
    "ContinualMetrics",
    "ContinualLearningEvaluator",
    "ToolCallEvaluation",
    "ToolPerformanceTracker",
    # Baselines
    "VanillaFinetuneAgent",
    "ICLBaselineAgent",
    # Runner
    "ContinualBenchmarkRunner",
]
