"""
Continual Learning Results Data Model

This module defines data structures for storing and analyzing
continual learning evaluation results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from tau2.data_model.simulation import SimulationRun, RewardInfo


class TrainingMode(str, Enum):
    """Training mode for continual learning."""
    VANILLA_SFT = "vanilla_sft"
    ICL = "icl"
    GRPO = "grpo"  # Group Relative Policy Optimization
    GRPO_ONLINE = "grpo_online"  # GRPO with immediate updates after each experience
    NONE = "none"


class ToolPerformance(BaseModel):
    """Performance metrics for a single tool."""

    tool_name: str = Field(description="Name of the tool")
    selection_accuracy: float = Field(description="Accuracy of selecting this tool when needed")
    invocation_accuracy: float = Field(description="Accuracy of invoking with correct parameters")
    output_usage_accuracy: float = Field(description="Accuracy of using tool output correctly")
    total_calls: int = Field(description="Total number of calls to this tool", default=0)
    correct_calls: int = Field(description="Number of correct calls", default=0)

    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy combining all three aspects."""
        return (self.selection_accuracy + self.invocation_accuracy + self.output_usage_accuracy) / 3


class StageResult(BaseModel):
    """Results for a single learning stage."""

    stage_id: str = Field(description="Unique identifier for the stage")
    stage_name: str = Field(description="Human-readable name of the stage", default="")

    # Learning phase results
    learning_runs: Annotated[
        list[SimulationRun],
        Field(description="Simulation runs from learning phase", default_factory=list)
    ]
    learning_reward: float = Field(description="Average reward during learning", default=0.0)

    # Evaluation phase results
    eval_runs: Annotated[
        list[SimulationRun],
        Field(description="Simulation runs from evaluation phase", default_factory=list)
    ]
    eval_reward: float = Field(description="Average reward during evaluation", default=0.0)
    pass_k_rates: Annotated[
        dict[int, float],
        Field(description="Pass@k rates for different k values", default_factory=dict)
    ]

    # Retention phase results
    retention_runs: Annotated[
        list[SimulationRun],
        Field(description="Simulation runs from retention testing", default_factory=list)
    ]
    retention_reward: float = Field(description="Average reward on retention tasks", default=0.0)

    # Tool-level performance
    tool_performance: Annotated[
        dict[str, ToolPerformance],
        Field(description="Performance metrics per tool", default_factory=dict)
    ]
    new_tool_success_rate: float = Field(description="Success rate on new tools", default=0.0)
    changed_tool_success_rate: float = Field(description="Success rate on changed tools", default=0.0)

    # Stage status
    passed: bool = Field(description="Whether the stage was passed", default=False)

    def get_new_tool_tasks(self) -> list[str]:
        """Get task IDs that involve new tools."""
        # This would be populated based on curriculum configuration
        return []


class ForgettingAnalysis(BaseModel):
    """Analysis of forgetting for a specific tool."""

    tool_name: str = Field(description="Name of the tool")
    learned_stage: str = Field(description="Stage where tool was learned")
    learned_accuracy: float = Field(description="Accuracy when first learned")
    max_accuracy: float = Field(description="Maximum accuracy achieved")
    max_accuracy_stage: str = Field(description="Stage where max accuracy was achieved")
    final_accuracy: float = Field(description="Final accuracy")
    forgetting: float = Field(description="Amount of forgetting (max - final)")
    retention: float = Field(description="Retention rate (final / learned)")
    performance_curve: Annotated[
        list[tuple[str, float]],
        Field(description="Performance over stages", default_factory=list)
    ]


class GeneralizationMetrics(BaseModel):
    """Metrics for measuring generalization capabilities."""

    # Tool composition generalization
    tool_composition_seen_accuracy: float = Field(
        description="Accuracy on seen tool combinations", default=0.0
    )
    tool_composition_unseen_accuracy: float = Field(
        description="Accuracy on unseen tool combinations", default=0.0
    )
    tool_composition_gap: float = Field(
        description="Gap between seen and unseen", default=0.0
    )
    num_unseen_combinations: int = Field(
        description="Number of unseen tool combinations tested", default=0
    )

    # Parameter generalization
    parameter_seen_accuracy: float = Field(
        description="Accuracy with seen parameter values", default=0.0
    )
    parameter_unseen_accuracy: float = Field(
        description="Accuracy with unseen parameter values", default=0.0
    )
    parameter_gap: float = Field(
        description="Gap between seen and unseen parameters", default=0.0
    )

    # Cross-domain generalization (optional)
    cross_domain_transfer_gain: Optional[float] = Field(
        description="Transfer gain to other domains", default=None
    )


class ContinualLearningResults(BaseModel):
    """Complete results from a continual learning evaluation."""

    # Identification
    curriculum_id: str = Field(description="ID of the curriculum used")
    curriculum_name: str = Field(description="Name of the curriculum", default="")

    # Configuration
    agent_config: Annotated[
        dict[str, Any],
        Field(description="Agent configuration", default_factory=dict)
    ]
    training_mode: TrainingMode = Field(
        description="Training mode used", default=TrainingMode.ICL
    )

    # Stage results
    stage_results: Annotated[
        list[StageResult],
        Field(description="Results for each stage", default_factory=list)
    ]

    # Overall metrics (computed after all stages)
    overall_metrics: Annotated[
        dict[str, float],
        Field(description="Overall computed metrics", default_factory=dict)
    ]

    # Forgetting analysis
    forgetting_analysis: Annotated[
        dict[str, ForgettingAnalysis],
        Field(description="Per-tool forgetting analysis", default_factory=dict)
    ]

    # Generalization metrics
    generalization_metrics: Optional[GeneralizationMetrics] = Field(
        description="Generalization capability metrics", default=None
    )

    # Learning curve data
    learning_curve: Annotated[
        list[dict[str, Any]],
        Field(description="Learning curve data points", default_factory=list)
    ]

    # Baseline comparison
    baseline_results: Optional["ContinualLearningResults"] = Field(
        description="Results from baseline agent for comparison", default=None
    )

    # Metadata
    metadata: Annotated[
        dict[str, Any],
        Field(description="Additional metadata", default_factory=dict)
    ]
    timestamp: str = Field(
        description="Timestamp of evaluation",
        default_factory=lambda: datetime.now().isoformat()
    )

    def add_stage_result(self, result: StageResult) -> None:
        """Add a stage result to the results."""
        self.stage_results.append(result)

    def get_stage_result(self, stage_id: str) -> Optional[StageResult]:
        """Get result for a specific stage."""
        for result in self.stage_results:
            if result.stage_id == stage_id:
                return result
        return None

    def save(self, path: Path | str) -> None:
        """Save results to a JSON file."""
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "ContinualLearningResults":
        """Load results from a JSON file."""
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the results."""
        return {
            "curriculum_id": self.curriculum_id,
            "num_stages_completed": len(self.stage_results),
            "training_mode": self.training_mode.value,
            "overall_metrics": self.overall_metrics,
            "final_eval_reward": self.stage_results[-1].eval_reward if self.stage_results else 0.0,
        }
