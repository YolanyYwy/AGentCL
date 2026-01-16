"""
Task Selector

This module provides utilities for selecting and filtering tasks
based on various criteria for continual learning evaluation.
"""

from typing import Optional

from tau2.data_model.tasks import Task, Action
from tau2.continual.curriculum.stage import LearningStage


class TaskSelector:
    """
    Utility class for selecting tasks based on various criteria.

    Used to filter tasks for different phases of continual learning
    evaluation (learning, evaluation, retention).
    """

    def __init__(self, tasks: list[Task]):
        """
        Initialize the task selector.

        Args:
            tasks: List of all available tasks
        """
        self.tasks = {task.id: task for task in tasks}
        self._tool_to_tasks: dict[str, list[str]] = {}
        self._build_tool_index()

    def _build_tool_index(self) -> None:
        """Build an index mapping tools to tasks that use them."""
        for task_id, task in self.tasks.items():
            if task.evaluation_criteria and task.evaluation_criteria.actions:
                for action in task.evaluation_criteria.actions:
                    tool_name = action.name
                    if tool_name not in self._tool_to_tasks:
                        self._tool_to_tasks[tool_name] = []
                    if task_id not in self._tool_to_tasks[tool_name]:
                        self._tool_to_tasks[tool_name].append(task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        return self.tasks.get(task_id)

    def get_tasks(self, task_ids: list[str]) -> list[Task]:
        """Get multiple tasks by their IDs."""
        return [self.tasks[tid] for tid in task_ids if tid in self.tasks]

    def get_tasks_using_tool(self, tool_name: str) -> list[Task]:
        """Get all tasks that use a specific tool."""
        task_ids = self._tool_to_tasks.get(tool_name, [])
        return self.get_tasks(task_ids)

    def get_tasks_using_tools(self, tool_names: list[str]) -> list[Task]:
        """Get all tasks that use any of the specified tools."""
        task_ids = set()
        for tool_name in tool_names:
            task_ids.update(self._tool_to_tasks.get(tool_name, []))
        return self.get_tasks(list(task_ids))

    def get_tasks_using_only_tools(self, tool_names: set[str]) -> list[Task]:
        """Get tasks that only use tools from the specified set."""
        result = []
        for task in self.tasks.values():
            if self._task_uses_only_tools(task, tool_names):
                result.append(task)
        return result

    def _task_uses_only_tools(self, task: Task, allowed_tools: set[str]) -> bool:
        """Check if a task only uses tools from the allowed set."""
        if not task.evaluation_criteria or not task.evaluation_criteria.actions:
            return True  # Tasks without actions are allowed

        for action in task.evaluation_criteria.actions:
            if action.name not in allowed_tools:
                return False
        return True

    def get_tools_used_by_task(self, task_id: str) -> list[str]:
        """Get all tools used by a specific task."""
        task = self.tasks.get(task_id)
        if not task or not task.evaluation_criteria or not task.evaluation_criteria.actions:
            return []

        return list(set(action.name for action in task.evaluation_criteria.actions))

    def get_task_complexity(self, task_id: str) -> int:
        """
        Get the complexity of a task based on number of actions.

        Returns:
            Number of expected actions, or 0 if no actions defined
        """
        task = self.tasks.get(task_id)
        if not task or not task.evaluation_criteria or not task.evaluation_criteria.actions:
            return 0
        return len(task.evaluation_criteria.actions)

    def filter_tasks_by_complexity(
        self,
        task_ids: list[str],
        min_actions: int = 0,
        max_actions: int = 100
    ) -> list[str]:
        """Filter tasks by complexity (number of actions)."""
        result = []
        for task_id in task_ids:
            complexity = self.get_task_complexity(task_id)
            if min_actions <= complexity <= max_actions:
                result.append(task_id)
        return result

    def get_tasks_for_stage(self, stage: LearningStage) -> dict[str, list[Task]]:
        """
        Get all tasks organized by phase for a learning stage.

        Returns:
            Dictionary with keys 'learning', 'eval', 'retention'
        """
        return {
            "learning": self.get_tasks(stage.learning_tasks),
            "eval": self.get_tasks(stage.eval_tasks),
            "retention": self.get_tasks(stage.retention_tasks),
        }

    def validate_stage_tasks(self, stage: LearningStage) -> dict[str, list[str]]:
        """
        Validate that all tasks in a stage exist and use appropriate tools.

        Returns:
            Dictionary with 'missing' and 'invalid_tools' lists
        """
        all_task_ids = stage.get_all_tasks()
        available_tools = set(stage.available_tools)

        missing = []
        invalid_tools = []

        for task_id in all_task_ids:
            if task_id not in self.tasks:
                missing.append(task_id)
                continue

            task_tools = set(self.get_tools_used_by_task(task_id))
            if not task_tools.issubset(available_tools):
                invalid_tools.append(task_id)

        return {
            "missing": missing,
            "invalid_tools": invalid_tools,
        }

    def get_tool_combinations_in_tasks(self, task_ids: list[str]) -> set[tuple[str, ...]]:
        """
        Get all unique tool combinations used across tasks.

        Returns:
            Set of tuples, each representing a tool combination
        """
        combinations = set()
        for task_id in task_ids:
            tools = self.get_tools_used_by_task(task_id)
            if tools:
                combinations.add(tuple(sorted(tools)))
        return combinations

    def get_parameter_values_in_tasks(
        self,
        task_ids: list[str]
    ) -> dict[str, set[str]]:
        """
        Get all parameter values used across tasks.

        Returns:
            Dictionary mapping "tool.param" to set of values
        """
        param_values: dict[str, set[str]] = {}

        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if not task or not task.evaluation_criteria or not task.evaluation_criteria.actions:
                continue

            for action in task.evaluation_criteria.actions:
                for param_name, param_value in action.arguments.items():
                    key = f"{action.name}.{param_name}"
                    if key not in param_values:
                        param_values[key] = set()
                    param_values[key].add(str(param_value))

        return param_values

    def suggest_retention_tasks(
        self,
        current_stage_index: int,
        stages: list[LearningStage],
        num_tasks: int = 3
    ) -> list[str]:
        """
        Suggest retention tasks for a stage based on previous stages.

        Args:
            current_stage_index: Index of the current stage
            stages: List of all stages
            num_tasks: Number of retention tasks to suggest

        Returns:
            List of suggested task IDs
        """
        if current_stage_index == 0:
            return []

        # Collect tasks from previous stages
        previous_tasks = []
        for i in range(current_stage_index):
            previous_tasks.extend(stages[i].eval_tasks)

        # Prioritize tasks that use tools from earlier stages
        # that are still available in the current stage
        current_tools = set(stages[current_stage_index].available_tools)

        valid_tasks = []
        for task_id in previous_tasks:
            task_tools = set(self.get_tools_used_by_task(task_id))
            if task_tools.issubset(current_tools):
                valid_tasks.append(task_id)

        # Return up to num_tasks, prioritizing diversity
        return valid_tasks[:num_tasks]
