"""
Tool Analysis

This module provides fine-grained analysis of tool usage in
continual learning evaluation.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from tau2.data_model.message import Message, AssistantMessage, ToolMessage, ToolCall
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task, Action


@dataclass
class ToolCallEvaluation:
    """Fine-grained evaluation of a single tool call."""

    # Basic info
    tool_name: str
    task_id: str
    message_index: int

    # Three-way evaluation
    tool_selection_correct: bool = False
    tool_invocation_valid: bool = False
    tool_output_used_correctly: bool = False

    # Additional details
    expected_tool: Optional[str] = None
    argument_errors: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def fully_correct(self) -> bool:
        """Check if the tool call is fully correct."""
        return (
            self.tool_selection_correct and
            self.tool_invocation_valid and
            self.tool_output_used_correctly
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "task_id": self.task_id,
            "message_index": self.message_index,
            "tool_selection_correct": self.tool_selection_correct,
            "tool_invocation_valid": self.tool_invocation_valid,
            "tool_output_used_correctly": self.tool_output_used_correctly,
            "fully_correct": self.fully_correct,
            "expected_tool": self.expected_tool,
            "argument_errors": self.argument_errors,
            "notes": self.notes,
        }


def evaluate_tool_call(
    tool_call: ToolCall,
    expected_actions: list[Action],
    run: SimulationRun,
    message_index: int,
) -> ToolCallEvaluation:
    """
    Evaluate a single tool call against expected actions.

    Args:
        tool_call: The tool call to evaluate
        expected_actions: List of expected actions from the task
        run: The simulation run containing this tool call
        message_index: Index of the message containing this tool call

    Returns:
        ToolCallEvaluation with detailed results
    """
    evaluation = ToolCallEvaluation(
        tool_name=tool_call.name,
        task_id=run.task_id,
        message_index=message_index,
    )

    # 1. Check tool selection
    expected_tool_names = [a.name for a in expected_actions]
    evaluation.tool_selection_correct = tool_call.name in expected_tool_names

    if not evaluation.tool_selection_correct:
        evaluation.expected_tool = expected_tool_names[0] if expected_tool_names else None
        evaluation.notes = f"Wrong tool selected. Expected one of: {expected_tool_names}"
        return evaluation

    # 2. Check tool invocation (arguments)
    matching_action = None
    for action in expected_actions:
        if action.name == tool_call.name:
            if action.compare_with_tool_call(tool_call):
                matching_action = action
                evaluation.tool_invocation_valid = True
                break

    if not evaluation.tool_invocation_valid:
        # Find argument mismatches
        for action in expected_actions:
            if action.name == tool_call.name:
                errors = _find_argument_errors(tool_call, action)
                evaluation.argument_errors = errors
                evaluation.notes = f"Argument errors: {errors}"
                break
        return evaluation

    # 3. Check output usage (simplified - checks if conversation continues appropriately)
    evaluation.tool_output_used_correctly = _check_output_usage(
        run.messages, message_index
    )

    if not evaluation.tool_output_used_correctly:
        evaluation.notes = "Tool output may not have been used correctly"

    return evaluation


def _find_argument_errors(tool_call: ToolCall, expected_action: Action) -> list[str]:
    """Find specific argument errors between tool call and expected action."""
    errors = []

    compare_args = expected_action.compare_args
    if compare_args is None:
        compare_args = list(expected_action.arguments.keys())

    for arg_name in compare_args:
        expected_value = expected_action.arguments.get(arg_name)
        actual_value = tool_call.arguments.get(arg_name)

        if arg_name not in tool_call.arguments:
            errors.append(f"Missing argument: {arg_name}")
        elif actual_value != expected_value:
            errors.append(
                f"Wrong value for {arg_name}: "
                f"expected {expected_value}, got {actual_value}"
            )

    return errors


def _check_output_usage(messages: list[Message], tool_call_index: int) -> bool:
    """
    Check if the tool output was used appropriately.

    This is a simplified check that looks for:
    - A tool result message following the tool call
    - Subsequent assistant message that references the result
    """
    # Find the tool result
    tool_result = None
    for i in range(tool_call_index + 1, len(messages)):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            tool_result = msg
            break
        elif isinstance(msg, AssistantMessage):
            # No tool result before next assistant message
            break

    if tool_result is None or tool_result.error:
        return False

    # Check if there's a subsequent assistant message
    # (indicating the conversation continued)
    for i in range(tool_call_index + 1, len(messages)):
        msg = messages[i]
        if isinstance(msg, AssistantMessage):
            # Conversation continued after tool call
            return True

    # If this was the last action and the run was successful, consider it correct
    return True


class ToolPerformanceTracker:
    """
    Tracks tool performance across stages for continual learning analysis.
    """

    def __init__(self):
        """Initialize the tracker."""
        # tool_name -> stage_id -> list of evaluations
        self.evaluations: dict[str, dict[str, list[ToolCallEvaluation]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # tool_name -> stage where it was first introduced
        self.tool_introduction_stage: dict[str, str] = {}

        # Aggregated statistics
        self._stats_cache: Optional[dict] = None

    def add_evaluation(
        self,
        evaluation: ToolCallEvaluation,
        stage_id: str,
    ) -> None:
        """Add an evaluation result."""
        self.evaluations[evaluation.tool_name][stage_id].append(evaluation)
        self._stats_cache = None  # Invalidate cache

    def add_evaluations_from_run(
        self,
        run: SimulationRun,
        task: Task,
        stage_id: str,
    ) -> list[ToolCallEvaluation]:
        """
        Evaluate all tool calls in a run and add them to the tracker.

        Returns:
            List of evaluations for this run
        """
        evaluations = []

        expected_actions = []
        if task.evaluation_criteria and task.evaluation_criteria.actions:
            expected_actions = task.evaluation_criteria.actions

        for i, msg in enumerate(run.messages):
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    evaluation = evaluate_tool_call(
                        tool_call=tool_call,
                        expected_actions=expected_actions,
                        run=run,
                        message_index=i,
                    )
                    self.add_evaluation(evaluation, stage_id)
                    evaluations.append(evaluation)

        return evaluations

    def set_tool_introduction_stage(self, tool_name: str, stage_id: str) -> None:
        """Record when a tool was first introduced."""
        if tool_name not in self.tool_introduction_stage:
            self.tool_introduction_stage[tool_name] = stage_id

    def get_tool_accuracy(
        self,
        tool_name: str,
        stage_id: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Get accuracy metrics for a tool.

        Args:
            tool_name: Name of the tool
            stage_id: Optional stage to filter by

        Returns:
            Dictionary with selection, invocation, and usage accuracy
        """
        if stage_id:
            evals = self.evaluations.get(tool_name, {}).get(stage_id, [])
        else:
            evals = []
            for stage_evals in self.evaluations.get(tool_name, {}).values():
                evals.extend(stage_evals)

        if not evals:
            return {
                "selection_accuracy": 0.0,
                "invocation_accuracy": 0.0,
                "output_usage_accuracy": 0.0,
                "overall_accuracy": 0.0,
                "num_calls": 0,
            }

        num_calls = len(evals)
        selection_correct = sum(1 for e in evals if e.tool_selection_correct)
        invocation_valid = sum(1 for e in evals if e.tool_invocation_valid)
        output_correct = sum(1 for e in evals if e.tool_output_used_correctly)
        fully_correct = sum(1 for e in evals if e.fully_correct)

        return {
            "selection_accuracy": selection_correct / num_calls,
            "invocation_accuracy": invocation_valid / num_calls,
            "output_usage_accuracy": output_correct / num_calls,
            "overall_accuracy": fully_correct / num_calls,
            "num_calls": num_calls,
        }

    def get_tool_performance_curve(
        self,
        tool_name: str,
    ) -> list[tuple[str, float]]:
        """
        Get the performance curve for a tool across stages.

        Returns:
            List of (stage_id, accuracy) tuples
        """
        curve = []
        tool_stages = self.evaluations.get(tool_name, {})

        for stage_id in sorted(tool_stages.keys()):
            accuracy = self.get_tool_accuracy(tool_name, stage_id)
            curve.append((stage_id, accuracy["overall_accuracy"]))

        return curve

    def compute_tool_forgetting(self, tool_name: str) -> dict[str, Any]:
        """
        Compute forgetting metrics for a tool.

        Returns:
            Dictionary with forgetting analysis
        """
        curve = self.get_tool_performance_curve(tool_name)

        if len(curve) < 2:
            return {
                "forgetting": 0.0,
                "retention": 1.0,
                "max_accuracy": curve[0][1] if curve else 0.0,
                "final_accuracy": curve[-1][1] if curve else 0.0,
                "learned_accuracy": curve[0][1] if curve else 0.0,
            }

        learned_acc = curve[0][1]
        max_acc = max(acc for _, acc in curve)
        final_acc = curve[-1][1]

        forgetting = max(0, max_acc - final_acc)
        retention = final_acc / learned_acc if learned_acc > 0 else 0.0

        return {
            "forgetting": forgetting,
            "retention": retention,
            "max_accuracy": max_acc,
            "final_accuracy": final_acc,
            "learned_accuracy": learned_acc,
            "max_stage": next(s for s, a in curve if a == max_acc),
            "performance_curve": curve,
        }

    def get_all_tool_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all tracked tools."""
        stats = {}
        for tool_name in self.evaluations.keys():
            stats[tool_name] = {
                "accuracy": self.get_tool_accuracy(tool_name),
                "forgetting": self.compute_tool_forgetting(tool_name),
                "introduction_stage": self.tool_introduction_stage.get(tool_name),
            }
        return stats

    def get_stage_summary(self, stage_id: str) -> dict[str, Any]:
        """Get a summary of tool performance for a specific stage."""
        stage_tools = {}

        for tool_name, stages in self.evaluations.items():
            if stage_id in stages:
                stage_tools[tool_name] = self.get_tool_accuracy(tool_name, stage_id)

        # Aggregate metrics
        if not stage_tools:
            return {
                "num_tools": 0,
                "avg_selection_accuracy": 0.0,
                "avg_invocation_accuracy": 0.0,
                "avg_output_usage_accuracy": 0.0,
                "avg_overall_accuracy": 0.0,
                "per_tool": {},
            }

        num_tools = len(stage_tools)
        avg_selection = sum(t["selection_accuracy"] for t in stage_tools.values()) / num_tools
        avg_invocation = sum(t["invocation_accuracy"] for t in stage_tools.values()) / num_tools
        avg_output = sum(t["output_usage_accuracy"] for t in stage_tools.values()) / num_tools
        avg_overall = sum(t["overall_accuracy"] for t in stage_tools.values()) / num_tools

        return {
            "num_tools": num_tools,
            "avg_selection_accuracy": avg_selection,
            "avg_invocation_accuracy": avg_invocation,
            "avg_output_usage_accuracy": avg_output,
            "avg_overall_accuracy": avg_overall,
            "per_tool": stage_tools,
        }

    def get_new_tool_performance(
        self,
        stage_id: str,
        new_tools: list[str],
    ) -> dict[str, float]:
        """Get performance metrics specifically for new tools in a stage."""
        if not new_tools:
            return {
                "selection_accuracy": 0.0,
                "invocation_accuracy": 0.0,
                "output_usage_accuracy": 0.0,
                "overall_accuracy": 0.0,
            }

        all_evals = []
        for tool_name in new_tools:
            evals = self.evaluations.get(tool_name, {}).get(stage_id, [])
            all_evals.extend(evals)

        if not all_evals:
            return {
                "selection_accuracy": 0.0,
                "invocation_accuracy": 0.0,
                "output_usage_accuracy": 0.0,
                "overall_accuracy": 0.0,
            }

        num_calls = len(all_evals)
        return {
            "selection_accuracy": sum(1 for e in all_evals if e.tool_selection_correct) / num_calls,
            "invocation_accuracy": sum(1 for e in all_evals if e.tool_invocation_valid) / num_calls,
            "output_usage_accuracy": sum(1 for e in all_evals if e.tool_output_used_correctly) / num_calls,
            "overall_accuracy": sum(1 for e in all_evals if e.fully_correct) / num_calls,
        }
