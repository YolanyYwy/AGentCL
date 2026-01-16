"""
Evaluation Protocol for Continual Learning Benchmark

This module defines the standard evaluation protocol that ensures
fair and reproducible evaluation across different algorithms.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.continual.curriculum.stage import LearningStage
from tau2.continual.benchmark.agent_interface import ContinualAgent, Experience, AgentResponse


class EvaluationPhase(str, Enum):
    """Phases in the evaluation protocol."""
    LEARNING = "learning"
    EVALUATION = "evaluation"
    RETENTION = "retention"


@dataclass
class PhaseConfig:
    """Configuration for an evaluation phase."""
    num_trials: int = 4
    collect_experiences: bool = False  # Whether to collect experiences for learning
    record_tool_calls: bool = True


@dataclass
class ProtocolConfig:
    """Configuration for the evaluation protocol."""

    # Phase configurations
    learning_phase: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        num_trials=3,
        collect_experiences=True,
    ))
    evaluation_phase: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        num_trials=4,
        collect_experiences=False,
    ))
    retention_phase: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        num_trials=4,
        collect_experiences=False,
    ))

    # Simulation settings
    max_steps: int = 30
    max_errors: int = 5

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"

    # Logging
    verbose: bool = True
    log_level: str = "INFO"


class EvaluationProtocol:
    """
    Standard evaluation protocol for continual learning.

    The protocol ensures fair comparison by:
    1. Fixed order of stages
    2. Standardized learning/evaluation/retention phases
    3. Consistent metric computation
    4. Reproducible random seeds

    Protocol Flow:
    ```
    for each stage in curriculum:
        1. on_stage_start(stage)
        2. LEARNING PHASE:
           - Run learning tasks
           - Collect experiences
           - Call agent.learn(experiences)
        3. EVALUATION PHASE:
           - Run evaluation tasks
           - Compute metrics
        4. RETENTION PHASE (if retention_tasks exist):
           - Run retention tasks
           - Measure forgetting
        5. on_stage_end(stage, metrics)
        6. Save checkpoint (optional)
    ```
    """

    def __init__(
        self,
        config: Optional[ProtocolConfig] = None,
        task_executor: Optional[Callable] = None,
    ):
        """
        Initialize the evaluation protocol.

        Args:
            config: Protocol configuration
            task_executor: Function to execute a single task
                          Signature: (task, agent, tools) -> SimulationRun
        """
        self.config = config or ProtocolConfig()
        self.task_executor = task_executor

        # State tracking
        self.current_stage: Optional[LearningStage] = None
        self.current_phase: Optional[EvaluationPhase] = None

    def run_stage(
        self,
        stage: LearningStage,
        agent: ContinualAgent,
        tasks: dict[str, Task],
        tools: list[dict],
    ) -> dict[str, Any]:
        """
        Run the complete evaluation protocol for a single stage.

        Args:
            stage: The learning stage to evaluate
            agent: The continual learning agent
            tasks: Dictionary mapping task_id to Task
            tools: Available tool schemas

        Returns:
            Dictionary with stage results and metrics
        """
        self.current_stage = stage

        # Notify agent of stage start
        agent.on_stage_start(stage)

        results = {
            "stage_id": stage.stage_id,
            "stage_name": stage.stage_name,
        }

        # 1. Learning Phase
        if stage.learning_tasks:
            learning_results = self._run_learning_phase(
                stage, agent, tasks, tools
            )
            results["learning"] = learning_results

        # 2. Evaluation Phase
        eval_results = self._run_evaluation_phase(
            stage, agent, tasks, tools
        )
        results["evaluation"] = eval_results

        # 3. Retention Phase
        if stage.retention_tasks:
            retention_results = self._run_retention_phase(
                stage, agent, tasks, tools
            )
            results["retention"] = retention_results

        # Notify agent of stage end
        agent.on_stage_end(stage, results)

        # Save checkpoint
        if self.config.save_checkpoints:
            checkpoint_path = f"{self.config.checkpoint_dir}/{stage.stage_id}"
            agent.save_checkpoint(checkpoint_path)

        return results

    def _run_learning_phase(
        self,
        stage: LearningStage,
        agent: ContinualAgent,
        tasks: dict[str, Task],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Run the learning phase."""
        self.current_phase = EvaluationPhase.LEARNING

        if self.config.verbose:
            print(f"  Learning Phase: {len(stage.learning_tasks)} tasks")

        # Collect experiences
        experiences = []
        runs = []

        for task_id in stage.learning_tasks:
            task = tasks.get(task_id)
            if task is None:
                continue

            for trial in range(self.config.learning_phase.num_trials):
                run = self._execute_task(task, agent, tools)
                runs.append(run)

                # Convert to Experience
                exp = self._run_to_experience(run, task, stage.stage_id)
                experiences.append(exp)

        # Call agent's learn method
        learn_stats = agent.learn(stage, experiences)

        return {
            "num_tasks": len(stage.learning_tasks),
            "num_experiences": len(experiences),
            "successful_experiences": sum(1 for e in experiences if e.success),
            "learn_stats": learn_stats,
            "runs": runs,
        }

    def _run_evaluation_phase(
        self,
        stage: LearningStage,
        agent: ContinualAgent,
        tasks: dict[str, Task],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Run the evaluation phase."""
        self.current_phase = EvaluationPhase.EVALUATION

        if self.config.verbose:
            print(f"  Evaluation Phase: {len(stage.eval_tasks)} tasks")

        runs = []
        for task_id in stage.eval_tasks:
            task = tasks.get(task_id)
            if task is None:
                continue

            for trial in range(self.config.evaluation_phase.num_trials):
                run = self._execute_task(task, agent, tools)
                runs.append(run)

        # Compute metrics
        rewards = [r.reward_info.reward if r.reward_info else 0.0 for r in runs]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        return {
            "num_tasks": len(stage.eval_tasks),
            "num_runs": len(runs),
            "average_reward": avg_reward,
            "runs": runs,
        }

    def _run_retention_phase(
        self,
        stage: LearningStage,
        agent: ContinualAgent,
        tasks: dict[str, Task],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Run the retention phase."""
        self.current_phase = EvaluationPhase.RETENTION

        if self.config.verbose:
            print(f"  Retention Phase: {len(stage.retention_tasks)} tasks")

        runs = []
        for task_id in stage.retention_tasks:
            task = tasks.get(task_id)
            if task is None:
                continue

            for trial in range(self.config.retention_phase.num_trials):
                run = self._execute_task(task, agent, tools)
                runs.append(run)

        rewards = [r.reward_info.reward if r.reward_info else 0.0 for r in runs]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        return {
            "num_tasks": len(stage.retention_tasks),
            "num_runs": len(runs),
            "average_reward": avg_reward,
            "runs": runs,
        }

    def _execute_task(
        self,
        task: Task,
        agent: ContinualAgent,
        tools: list[dict],
    ) -> SimulationRun:
        """Execute a single task."""
        if self.task_executor:
            return self.task_executor(task, agent, tools)
        else:
            raise NotImplementedError(
                "Task executor not provided. "
                "Either provide a task_executor or override _execute_task."
            )

    def _run_to_experience(
        self,
        run: SimulationRun,
        task: Task,
        stage_id: str,
    ) -> Experience:
        """Convert a SimulationRun to an Experience."""
        # Extract tool calls from messages
        tool_calls = []
        tool_results = []

        for msg in run.messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)
            if hasattr(msg, 'content') and msg.role == 'tool':
                tool_results.append(msg.content or "")

        # Get expected actions
        expected_actions = []
        if task.evaluation_criteria and task.evaluation_criteria.actions:
            expected_actions = [
                {"name": a.name, "arguments": a.arguments}
                for a in task.evaluation_criteria.actions
            ]

        reward = run.reward_info.reward if run.reward_info else 0.0

        return Experience(
            task_id=run.task_id,
            messages=run.messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            reward=reward,
            success=reward > 0,
            expected_actions=expected_actions,
            stage_id=stage_id,
        )
