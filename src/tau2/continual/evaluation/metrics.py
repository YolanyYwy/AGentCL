"""
Continual Learning Metrics

This module provides metrics computation for continual learning evaluation,
including forward transfer, backward transfer, forgetting, and generalization.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from tau2.data_model.continual_results import (
    StageResult,
    ContinualLearningResults,
    ForgettingAnalysis,
    GeneralizationMetrics,
)
from tau2.data_model.simulation import SimulationRun


@dataclass
class ContinualMetrics:
    """Complete set of continual learning metrics."""

    # Basic metrics
    average_reward: float = 0.0
    final_reward: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)

    # Continual learning metrics
    forward_transfer: float = 0.0
    forward_transfer_per_stage: dict[str, float] = field(default_factory=dict)

    backward_transfer: float = 0.0
    backward_transfer_per_stage: dict[str, float] = field(default_factory=dict)

    # Tool-level metrics
    tool_level_forgetting: dict[str, float] = field(default_factory=dict)
    tool_level_retention: dict[str, float] = field(default_factory=dict)
    average_forgetting: float = 0.0

    # Learning efficiency
    learning_efficiency: float = 0.0
    learning_efficiency_per_stage: dict[str, float] = field(default_factory=dict)
    aulc: float = 0.0  # Area Under Learning Curve

    # Generalization metrics
    tool_composition_generalization: float = 0.0
    parameter_generalization: float = 0.0
    cross_domain_generalization: Optional[float] = None

    # Fine-grained tool metrics
    tool_selection_accuracy: float = 0.0
    tool_invocation_accuracy: float = 0.0
    tool_output_usage_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
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
                "tool_level_retention": self.tool_level_retention,
            },
            "learning_efficiency": {
                "learning_efficiency": self.learning_efficiency,
                "learning_efficiency_per_stage": self.learning_efficiency_per_stage,
                "aulc": self.aulc,
            },
            "generalization": {
                "tool_composition": self.tool_composition_generalization,
                "parameter": self.parameter_generalization,
                "cross_domain": self.cross_domain_generalization,
            },
            "fine_grained": {
                "tool_selection": self.tool_selection_accuracy,
                "tool_invocation": self.tool_invocation_accuracy,
                "tool_output_usage": self.tool_output_usage_accuracy,
            },
        }

    def summary(self) -> str:
        """Generate a formatted summary of the metrics."""
        avg_forgetting = np.mean(list(self.tool_level_forgetting.values())) if self.tool_level_forgetting else 0.0

        return f"""
╔══════════════════════════════════════════════════════════════╗
║              Continual Learning Evaluation Summary            ║
╠══════════════════════════════════════════════════════════════╣
║ Basic Metrics                                                 ║
║   Average Reward:        {self.average_reward:>8.4f}                         ║
║   Final Reward:          {self.final_reward:>8.4f}                         ║
║   Pass@1:                {self.pass_at_k.get(1, 0):>8.4f}                         ║
║   Pass@4:                {self.pass_at_k.get(4, 0):>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Continual Learning Metrics                                    ║
║   Forward Transfer:      {self.forward_transfer:>8.4f}                         ║
║   Backward Transfer:     {self.backward_transfer:>8.4f}                         ║
║   Avg Tool Forgetting:   {avg_forgetting:>8.4f}                         ║
║   Learning Efficiency:   {self.learning_efficiency:>8.4f}                         ║
║   AULC:                  {self.aulc:>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Generalization Metrics                                        ║
║   Tool Composition:      {self.tool_composition_generalization:>8.4f}                         ║
║   Parameter Gen.:        {self.parameter_generalization:>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Fine-Grained Tool Metrics                                     ║
║   Tool Selection:        {self.tool_selection_accuracy:>8.4f}                         ║
║   Tool Invocation:       {self.tool_invocation_accuracy:>8.4f}                         ║
║   Output Usage:          {self.tool_output_usage_accuracy:>8.4f}                         ║
╚══════════════════════════════════════════════════════════════╝
"""


def compute_continual_metrics(
    results: ContinualLearningResults,
    baseline_results: Optional[ContinualLearningResults] = None,
) -> ContinualMetrics:
    """
    Compute all continual learning metrics from evaluation results.

    Args:
        results: The continual learning evaluation results
        baseline_results: Optional baseline results for computing transfer

    Returns:
        ContinualMetrics with all computed metrics
    """
    metrics = ContinualMetrics()

    stage_results = results.stage_results
    if not stage_results:
        return metrics

    # Basic metrics
    metrics.average_reward = _compute_average_reward(stage_results)
    metrics.final_reward = stage_results[-1].eval_reward if stage_results else 0.0
    metrics.pass_at_k = _compute_pass_at_k(stage_results)

    # Forward transfer
    fwt_results = compute_forward_transfer(stage_results, baseline_results)
    metrics.forward_transfer = fwt_results["average"]
    metrics.forward_transfer_per_stage = fwt_results["per_stage"]

    # Backward transfer
    bwt_results = compute_backward_transfer(stage_results)
    metrics.backward_transfer = bwt_results["average"]
    metrics.backward_transfer_per_stage = bwt_results["per_stage"]

    # Tool-level forgetting
    forgetting_results = compute_tool_level_forgetting(results.forgetting_analysis)
    metrics.tool_level_forgetting = forgetting_results["forgetting"]
    metrics.tool_level_retention = forgetting_results["retention"]
    metrics.average_forgetting = forgetting_results["average_forgetting"]

    # Learning efficiency
    efficiency_results = compute_learning_efficiency(stage_results)
    metrics.learning_efficiency = efficiency_results["normalized_efficiency"]
    metrics.learning_efficiency_per_stage = efficiency_results["per_stage"]
    metrics.aulc = efficiency_results["aulc"]

    # Generalization metrics
    if results.generalization_metrics:
        metrics.tool_composition_generalization = results.generalization_metrics.tool_composition_unseen_accuracy
        metrics.parameter_generalization = results.generalization_metrics.parameter_unseen_accuracy
        metrics.cross_domain_generalization = results.generalization_metrics.cross_domain_transfer_gain

    # Fine-grained tool metrics
    fine_grained = _compute_fine_grained_metrics(stage_results)
    metrics.tool_selection_accuracy = fine_grained["selection"]
    metrics.tool_invocation_accuracy = fine_grained["invocation"]
    metrics.tool_output_usage_accuracy = fine_grained["output_usage"]

    return metrics


def _compute_average_reward(stage_results: list[StageResult]) -> float:
    """Compute average reward across all stages."""
    if not stage_results:
        return 0.0
    rewards = [sr.eval_reward for sr in stage_results]
    return float(np.mean(rewards))


def _compute_pass_at_k(stage_results: list[StageResult]) -> dict[int, float]:
    """Compute Pass@k metrics."""
    # Aggregate pass@k from all stages
    all_pass_k: dict[int, list[float]] = {}

    for sr in stage_results:
        for k, rate in sr.pass_k_rates.items():
            if k not in all_pass_k:
                all_pass_k[k] = []
            all_pass_k[k].append(rate)

    return {k: float(np.mean(rates)) for k, rates in all_pass_k.items()}


def compute_forward_transfer(
    stage_results: list[StageResult],
    baseline_results: Optional[ContinualLearningResults] = None,
) -> dict[str, Any]:
    """
    Compute forward transfer metrics.

    FWT = Acc(new_tasks | current_stage) - Acc(new_tasks | baseline)

    Args:
        stage_results: Results from each stage
        baseline_results: Optional baseline for comparison

    Returns:
        Dictionary with per-stage and average FWT
    """
    fwt_per_stage = {}

    for i, sr in enumerate(stage_results):
        if i == 0:
            # First stage has no forward transfer
            fwt_per_stage[sr.stage_id] = 0.0
            continue

        # Get new tool success rate as proxy for forward transfer
        new_tool_acc = sr.new_tool_success_rate

        # If we have baseline results, compute relative to baseline
        if baseline_results and i < len(baseline_results.stage_results):
            baseline_acc = baseline_results.stage_results[i].new_tool_success_rate
            fwt = new_tool_acc - baseline_acc
        else:
            # Without baseline, use 0.5 as random baseline
            fwt = new_tool_acc - 0.5

        fwt_per_stage[sr.stage_id] = fwt

    avg_fwt = float(np.mean(list(fwt_per_stage.values()))) if fwt_per_stage else 0.0

    return {
        "per_stage": fwt_per_stage,
        "average": avg_fwt,
    }


def compute_backward_transfer(
    stage_results: list[StageResult],
) -> dict[str, Any]:
    """
    Compute backward transfer metrics.

    BWT = Acc(old_tasks | final_stage) - Acc(old_tasks | learned_stage)

    Positive BWT indicates improvement, negative indicates forgetting.

    Args:
        stage_results: Results from each stage

    Returns:
        Dictionary with per-stage and average BWT
    """
    bwt_per_stage = {}

    for i, sr in enumerate(stage_results):
        if i == 0 or not sr.retention_runs:
            bwt_per_stage[sr.stage_id] = 0.0
            continue

        # Compare retention performance to original performance
        retention_reward = sr.retention_reward

        # Get the original performance on these tasks
        # (approximated by the first stage's eval reward)
        original_reward = stage_results[0].eval_reward

        bwt = retention_reward - original_reward
        bwt_per_stage[sr.stage_id] = bwt

    avg_bwt = float(np.mean(list(bwt_per_stage.values()))) if bwt_per_stage else 0.0

    return {
        "per_stage": bwt_per_stage,
        "average": avg_bwt,
    }


def compute_tool_level_forgetting(
    forgetting_analysis: dict[str, ForgettingAnalysis],
) -> dict[str, Any]:
    """
    Compute tool-level forgetting metrics.

    Args:
        forgetting_analysis: Per-tool forgetting analysis

    Returns:
        Dictionary with forgetting and retention per tool
    """
    forgetting = {}
    retention = {}

    for tool_name, analysis in forgetting_analysis.items():
        forgetting[tool_name] = analysis.forgetting
        retention[tool_name] = analysis.retention

    avg_forgetting = float(np.mean(list(forgetting.values()))) if forgetting else 0.0

    return {
        "forgetting": forgetting,
        "retention": retention,
        "average_forgetting": avg_forgetting,
    }


def compute_learning_efficiency(
    stage_results: list[StageResult],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute learning efficiency metrics.

    Uses Area Under Learning Curve (AULC) normalized by number of examples.

    Args:
        stage_results: Results from each stage
        threshold: Accuracy threshold for samples_to_threshold

    Returns:
        Dictionary with efficiency metrics
    """
    if not stage_results:
        return {
            "aulc": 0.0,
            "normalized_efficiency": 0.0,
            "per_stage": {},
            "samples_to_threshold": -1,
        }

    # Build learning curve from stage rewards
    learning_curve = [sr.eval_reward for sr in stage_results]

    # Compute AULC
    aulc = float(np.trapz(learning_curve)) / len(learning_curve) if learning_curve else 0.0

    # Count total learning examples
    total_examples = sum(len(sr.learning_runs) for sr in stage_results)

    # Normalized efficiency
    normalized_efficiency = aulc / total_examples if total_examples > 0 else 0.0

    # Samples to threshold
    samples_to_threshold = -1
    cumulative_samples = 0
    for sr in stage_results:
        cumulative_samples += len(sr.learning_runs)
        if sr.eval_reward >= threshold:
            samples_to_threshold = cumulative_samples
            break

    # Per-stage efficiency
    per_stage = {}
    for sr in stage_results:
        num_examples = len(sr.learning_runs)
        if num_examples > 0:
            per_stage[sr.stage_id] = sr.eval_reward / num_examples
        else:
            per_stage[sr.stage_id] = 0.0

    return {
        "aulc": aulc,
        "normalized_efficiency": normalized_efficiency,
        "per_stage": per_stage,
        "samples_to_threshold": samples_to_threshold,
    }


def _compute_fine_grained_metrics(
    stage_results: list[StageResult],
) -> dict[str, float]:
    """Compute fine-grained tool metrics from stage results."""
    total_selection = 0.0
    total_invocation = 0.0
    total_output = 0.0
    count = 0

    for sr in stage_results:
        for tool_name, perf in sr.tool_performance.items():
            total_selection += perf.selection_accuracy
            total_invocation += perf.invocation_accuracy
            total_output += perf.output_usage_accuracy
            count += 1

    if count == 0:
        return {
            "selection": 0.0,
            "invocation": 0.0,
            "output_usage": 0.0,
        }

    return {
        "selection": total_selection / count,
        "invocation": total_invocation / count,
        "output_usage": total_output / count,
    }


def compute_tool_composition_generalization(
    eval_runs: list[SimulationRun],
    training_tool_combinations: set[tuple[str, ...]],
) -> dict[str, Any]:
    """
    Compute tool composition generalization.

    Measures performance on unseen tool combinations.

    Args:
        eval_runs: Evaluation simulation runs
        training_tool_combinations: Tool combinations seen during training

    Returns:
        Dictionary with generalization metrics
    """
    seen_results = []
    unseen_results = []

    for run in eval_runs:
        # Extract tool combination from run
        tools_used = set()
        for msg in run.messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.add(tc.name)

        tool_combo = tuple(sorted(tools_used))

        reward = run.reward_info.reward if run.reward_info else 0.0

        if tool_combo in training_tool_combinations:
            seen_results.append(reward)
        else:
            unseen_results.append(reward)

    seen_acc = float(np.mean(seen_results)) if seen_results else 0.0
    unseen_acc = float(np.mean(unseen_results)) if unseen_results else 0.0

    return {
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "generalization_gap": seen_acc - unseen_acc,
        "num_seen": len(seen_results),
        "num_unseen": len(unseen_results),
        "num_unseen_combinations": len(set(
            tuple(sorted(set(
                tc.name for msg in run.messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls
                for tc in msg.tool_calls
            )))
            for run in eval_runs
            if tuple(sorted(set(
                tc.name for msg in run.messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls
                for tc in msg.tool_calls
            ))) not in training_tool_combinations
        )),
    }


def compute_parameter_generalization(
    eval_runs: list[SimulationRun],
    training_param_values: dict[str, set[str]],
) -> dict[str, Any]:
    """
    Compute parameter generalization.

    Measures performance with unseen parameter values.

    Args:
        eval_runs: Evaluation simulation runs
        training_param_values: Parameter values seen during training
                              Format: {"tool.param": set of values}

    Returns:
        Dictionary with generalization metrics
    """
    seen_correct = []
    unseen_correct = []

    for run in eval_runs:
        reward = run.reward_info.reward if run.reward_info else 0.0

        has_unseen = False
        for msg in run.messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    for param_name, param_value in tc.arguments.items():
                        key = f"{tc.name}.{param_name}"
                        if key in training_param_values:
                            if str(param_value) not in training_param_values[key]:
                                has_unseen = True
                                break
                    if has_unseen:
                        break
            if has_unseen:
                break

        if has_unseen:
            unseen_correct.append(reward)
        else:
            seen_correct.append(reward)

    seen_acc = float(np.mean(seen_correct)) if seen_correct else 0.0
    unseen_acc = float(np.mean(unseen_correct)) if unseen_correct else 0.0

    return {
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "generalization_gap": seen_acc - unseen_acc,
        "num_seen": len(seen_correct),
        "num_unseen": len(unseen_correct),
    }
