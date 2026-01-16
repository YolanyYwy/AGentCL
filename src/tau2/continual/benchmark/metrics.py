"""
Metrics Computation for Continual Learning Benchmark

This module provides standardized metric computation that is
independent of the learning algorithm.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from tau2.data_model.continual_results import (
    ContinualLearningResults,
    StageResult,
    ForgettingAnalysis,
)


@dataclass
class ContinualMetrics:
    """All metrics computed by the benchmark."""

    # Basic metrics
    average_reward: float = 0.0
    final_reward: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)

    # Continual learning metrics
    forward_transfer: float = 0.0
    forward_transfer_per_stage: dict[str, float] = field(default_factory=dict)
    backward_transfer: float = 0.0
    backward_transfer_per_stage: dict[str, float] = field(default_factory=dict)

    # Forgetting metrics
    average_forgetting: float = 0.0
    tool_level_forgetting: dict[str, float] = field(default_factory=dict)
    tool_level_retention: dict[str, float] = field(default_factory=dict)

    # Learning efficiency
    learning_efficiency: float = 0.0
    aulc: float = 0.0  # Area Under Learning Curve

    # Generalization metrics
    tool_composition_generalization: float = 0.0
    parameter_generalization: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "basic": {
                "average_reward": self.average_reward,
                "final_reward": self.final_reward,
                "pass_at_k": self.pass_at_k,
            },
            "continual_learning": {
                "forward_transfer": self.forward_transfer,
                "forward_transfer_per_stage": self.forward_transfer_per_stage,
                "backward_transfer": self.backward_transfer,
                "backward_transfer_per_stage": self.backward_transfer_per_stage,
                "average_forgetting": self.average_forgetting,
                "tool_level_forgetting": self.tool_level_forgetting,
            },
            "learning_efficiency": {
                "learning_efficiency": self.learning_efficiency,
                "aulc": self.aulc,
            },
            "generalization": {
                "tool_composition": self.tool_composition_generalization,
                "parameter": self.parameter_generalization,
            },
        }

    def summary(self) -> str:
        """Generate formatted summary."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           Continual Learning Benchmark Results                ║
╠══════════════════════════════════════════════════════════════╣
║ Basic Metrics                                                 ║
║   Average Reward:        {self.average_reward:>8.4f}                         ║
║   Final Reward:          {self.final_reward:>8.4f}                         ║
║   Pass@1:                {self.pass_at_k.get(1, 0):>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Continual Learning Metrics                                    ║
║   Forward Transfer:      {self.forward_transfer:>8.4f}                         ║
║   Backward Transfer:     {self.backward_transfer:>8.4f}                         ║
║   Average Forgetting:    {self.average_forgetting:>8.4f}                         ║
║   Learning Efficiency:   {self.learning_efficiency:>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Generalization Metrics                                        ║
║   Tool Composition:      {self.tool_composition_generalization:>8.4f}                         ║
║   Parameter Gen.:        {self.parameter_generalization:>8.4f}                         ║
╚══════════════════════════════════════════════════════════════╝
"""


class MetricsComputer:
    """
    Computes all benchmark metrics from evaluation results.

    This class is algorithm-agnostic - it only looks at the results
    produced by running the evaluation protocol.
    """

    def __init__(self, baseline_reward: float = 0.5):
        """
        Initialize metrics computer.

        Args:
            baseline_reward: Baseline reward for computing transfer
                            (e.g., random agent performance)
        """
        self.baseline_reward = baseline_reward

    def compute_all(
        self,
        results: ContinualLearningResults,
    ) -> dict[str, Any]:
        """
        Compute all metrics from results.

        Args:
            results: Complete evaluation results

        Returns:
            Dictionary with all computed metrics
        """
        metrics = ContinualMetrics()

        stage_results = results.stage_results
        if not stage_results:
            return metrics.to_dict()

        # Basic metrics
        metrics.average_reward = self._compute_average_reward(stage_results)
        metrics.final_reward = stage_results[-1].eval_reward
        metrics.pass_at_k = self._compute_pass_at_k(stage_results)

        # Continual learning metrics
        fwt = self._compute_forward_transfer(stage_results)
        metrics.forward_transfer = fwt["average"]
        metrics.forward_transfer_per_stage = fwt["per_stage"]

        bwt = self._compute_backward_transfer(stage_results)
        metrics.backward_transfer = bwt["average"]
        metrics.backward_transfer_per_stage = bwt["per_stage"]

        # Forgetting metrics
        forgetting = self._compute_forgetting(results.forgetting_analysis)
        metrics.average_forgetting = forgetting["average"]
        metrics.tool_level_forgetting = forgetting["per_tool"]
        metrics.tool_level_retention = forgetting["retention"]

        # Learning efficiency
        efficiency = self._compute_learning_efficiency(stage_results)
        metrics.learning_efficiency = efficiency["normalized"]
        metrics.aulc = efficiency["aulc"]

        return metrics.to_dict()

    def _compute_average_reward(self, stage_results: list[StageResult]) -> float:
        """Compute average reward across all stages."""
        rewards = [sr.eval_reward for sr in stage_results]
        return float(np.mean(rewards)) if rewards else 0.0

    def _compute_pass_at_k(
        self,
        stage_results: list[StageResult],
        max_k: int = 4,
    ) -> dict[int, float]:
        """Compute Pass@k metrics."""
        pass_k = {}

        for k in range(1, max_k + 1):
            # Aggregate from all stages
            pass_rates = []
            for sr in stage_results:
                if k in sr.pass_k_rates:
                    pass_rates.append(sr.pass_k_rates[k])

            pass_k[k] = float(np.mean(pass_rates)) if pass_rates else 0.0

        return pass_k

    def _compute_forward_transfer(
        self,
        stage_results: list[StageResult],
    ) -> dict[str, Any]:
        """
        Compute forward transfer.

        FWT measures how much learning in previous stages helps
        with new tasks/tools.

        FWT_i = Acc(new_tools @ stage_i) - baseline
        """
        fwt_per_stage = {}

        for i, sr in enumerate(stage_results):
            if i == 0:
                fwt_per_stage[sr.stage_id] = 0.0
                continue

            # Use new tool success rate if available
            new_tool_acc = sr.new_tool_success_rate
            if new_tool_acc > 0:
                fwt = new_tool_acc - self.baseline_reward
            else:
                # Fall back to eval reward
                fwt = sr.eval_reward - self.baseline_reward

            fwt_per_stage[sr.stage_id] = fwt

        avg_fwt = float(np.mean(list(fwt_per_stage.values()))) if fwt_per_stage else 0.0

        return {
            "average": avg_fwt,
            "per_stage": fwt_per_stage,
        }

    def _compute_backward_transfer(
        self,
        stage_results: list[StageResult],
    ) -> dict[str, Any]:
        """
        Compute backward transfer.

        BWT measures how learning new things affects old knowledge.
        Positive = improvement, Negative = forgetting.

        BWT_i = Acc(old_tasks @ stage_i) - Acc(old_tasks @ stage_learned)
        """
        bwt_per_stage = {}

        for i, sr in enumerate(stage_results):
            if i == 0 or sr.retention_reward == 0:
                bwt_per_stage[sr.stage_id] = 0.0
                continue

            # Compare retention to original performance
            original_reward = stage_results[0].eval_reward
            bwt = sr.retention_reward - original_reward
            bwt_per_stage[sr.stage_id] = bwt

        avg_bwt = float(np.mean(list(bwt_per_stage.values()))) if bwt_per_stage else 0.0

        return {
            "average": avg_bwt,
            "per_stage": bwt_per_stage,
        }

    def _compute_forgetting(
        self,
        forgetting_analysis: dict[str, ForgettingAnalysis],
    ) -> dict[str, Any]:
        """Compute forgetting metrics from analysis."""
        per_tool = {}
        retention = {}

        for tool_name, analysis in forgetting_analysis.items():
            per_tool[tool_name] = analysis.forgetting
            retention[tool_name] = analysis.retention

        avg_forgetting = float(np.mean(list(per_tool.values()))) if per_tool else 0.0

        return {
            "average": avg_forgetting,
            "per_tool": per_tool,
            "retention": retention,
        }

    def _compute_learning_efficiency(
        self,
        stage_results: list[StageResult],
    ) -> dict[str, Any]:
        """
        Compute learning efficiency.

        Uses Area Under Learning Curve (AULC) normalized by
        number of learning examples.
        """
        if not stage_results:
            return {"aulc": 0.0, "normalized": 0.0}

        # Build learning curve
        learning_curve = [sr.eval_reward for sr in stage_results]

        # Compute AULC
        aulc = float(np.trapz(learning_curve)) / len(learning_curve)

        # Count total learning examples
        total_examples = sum(len(sr.learning_runs) for sr in stage_results)

        # Normalized efficiency
        normalized = aulc / total_examples if total_examples > 0 else 0.0

        return {
            "aulc": aulc,
            "normalized": normalized,
        }
