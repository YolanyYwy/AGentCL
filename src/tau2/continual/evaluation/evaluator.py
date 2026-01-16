"""
Continual Learning Evaluator

This module provides the main evaluator for running continual
learning experiments on τ²-Bench.
"""

from collections import defaultdict
from typing import Any, Optional

from loguru import logger

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.data_model.continual_results import (
    ContinualLearningResults,
    StageResult,
    ToolPerformance,
    ForgettingAnalysis,
    GeneralizationMetrics,
    TrainingMode,
)
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.curriculum.stage import LearningStage
from tau2.continual.curriculum.task_selector import TaskSelector
from tau2.continual.evaluation.tool_analysis import (
    ToolPerformanceTracker,
    evaluate_tool_call,
)
from tau2.continual.evaluation.metrics import (
    ContinualMetrics,
    compute_continual_metrics,
    compute_tool_composition_generalization,
    compute_parameter_generalization,
)
from tau2.continual.benchmark.agent_interface import Experience
from tau2.run import run_task
from tau2.evaluator.evaluator import EvaluationType


class ContinualLearningEvaluator:
    """
    Main evaluator for continual learning experiments.

    Orchestrates the learning and evaluation phases across
    multiple stages of a curriculum.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        domain: str,
        agent_llm: str,
        user_llm: str,
        agent_type: str = "llm_agent",
        user_type: str = "user_simulator",
        llm_args_agent: Optional[dict] = None,
        llm_args_user: Optional[dict] = None,
        max_steps: int = 30,
        max_errors: int = 5,
        seed: Optional[int] = None,
        training_mode: TrainingMode = TrainingMode.ICL,
        verbose: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            curriculum: The curriculum to evaluate
            domain: Domain name (e.g., "airline")
            agent_llm: LLM model for the agent
            user_llm: LLM model for the user simulator
            agent_type: Type of agent to use
            user_type: Type of user simulator to use
            llm_args_agent: Additional LLM arguments for agent
            llm_args_user: Additional LLM arguments for user
            max_steps: Maximum steps per simulation
            max_errors: Maximum errors before termination
            seed: Random seed
            training_mode: Training mode (ICL, vanilla_sft, none)
            verbose: Whether to print progress
        """
        self.curriculum = curriculum
        self.domain = domain
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        self.agent_type = agent_type
        self.user_type = user_type
        self.llm_args_agent = llm_args_agent or {"temperature": 0.0}
        self.llm_args_user = llm_args_user or {"temperature": 0.0}
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.seed = seed
        self.training_mode = training_mode
        self.verbose = verbose

        # Task selector (populated when tasks are loaded)
        self.task_selector: Optional[TaskSelector] = None

        # Tool performance tracker
        self.tool_tracker = ToolPerformanceTracker()

        # Results storage
        self.results = ContinualLearningResults(
            curriculum_id=curriculum.curriculum_id,
            curriculum_name=curriculum.curriculum_name,
            agent_config={
                "llm": agent_llm,
                "agent_type": agent_type,
                "llm_args": llm_args_agent,
            },
            training_mode=training_mode,
        )

        # Track tool combinations and parameter values for generalization
        self.training_tool_combinations: set[tuple[str, ...]] = set()
        self.training_param_values: dict[str, set[str]] = defaultdict(set)

        # ICL baseline (if using ICL mode)
        self.icl_baseline = None
        if training_mode == TrainingMode.ICL:
            from tau2.continual.training.icl_baseline import ICLContinualBaseline
            self.icl_baseline = ICLContinualBaseline()

        # GRPO agent (if using GRPO mode)
        self.grpo_agent = None
        if training_mode in (TrainingMode.GRPO, TrainingMode.GRPO_ONLINE):
            from tau2.continual.baselines.grpo_agent import GRPOContinualAgent, GRPOConfig
            grpo_config = GRPOConfig(
                model_name_or_path=agent_llm,
                update_after_each_experience=(training_mode == TrainingMode.GRPO_ONLINE),
            )
            self.grpo_agent = GRPOContinualAgent(config=grpo_config)

    def set_tasks(self, tasks: list[Task]) -> None:
        """Set the tasks for evaluation."""
        self.curriculum.set_task_cache(tasks)
        self.task_selector = TaskSelector(tasks)

        # Validate curriculum tasks
        available_ids = set(task.id for task in tasks)
        missing = self.curriculum.validate_tasks(available_ids)
        if missing:
            logger.warning(f"Missing tasks in curriculum: {missing}")

    def run_full_evaluation(self) -> ContinualLearningResults:
        """
        Run the complete continual learning evaluation.

        Returns:
            ContinualLearningResults with all metrics
        """
        if self.task_selector is None:
            raise ValueError("Tasks not set. Call set_tasks() first.")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting Continual Learning Evaluation")
            print(f"Curriculum: {self.curriculum.curriculum_name}")
            print(f"Domain: {self.domain}")
            print(f"Stages: {len(self.curriculum.stages)}")
            print(f"Training Mode: {self.training_mode.value}")
            print(f"{'='*60}")

        # Run each stage
        for stage_idx, stage in enumerate(self.curriculum.stages):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Stage {stage_idx + 1}/{len(self.curriculum.stages)}: {stage.stage_name}")
                print(f"Tools: {len(stage.available_tools)} ({len(stage.new_tools)} new)")
                print(f"{'='*60}")

            # Register new tools
            for tool_name in stage.new_tools:
                self.tool_tracker.set_tool_introduction_stage(tool_name, stage.stage_id)

            # Execute stage
            stage_result = self._run_stage(stage)

            # Record result
            self.results.stage_results.append(stage_result)

            if self.verbose:
                print(f"\n  Stage Results:")
                print(f"    Learning Reward: {stage_result.learning_reward:.4f}")
                print(f"    Eval Reward: {stage_result.eval_reward:.4f}")
                if stage_result.retention_reward > 0:
                    print(f"    Retention Reward: {stage_result.retention_reward:.4f}")
                print(f"    Passed: {stage_result.passed}")

            # Check if we should continue
            if not stage_result.passed:
                if self.verbose:
                    print(f"\n  Stage not passed. Stopping evaluation.")
                break

        # Compute forgetting analysis
        self._compute_forgetting_analysis()

        # Compute generalization metrics
        self._compute_generalization_metrics()

        # Compute final metrics
        self.results.overall_metrics = compute_continual_metrics(self.results).to_dict()

        return self.results

    def _run_stage(self, stage: LearningStage) -> StageResult:
        """Run a single learning stage."""
        # 1. Learning phase
        learning_runs = self._run_learning_phase(stage)
        learning_reward = self._compute_average_reward(learning_runs)

        # Update ICL baseline if using ICL mode
        if self.icl_baseline:
            successful_runs = [r for r in learning_runs if r.reward_info and r.reward_info.reward > 0]
            self.icl_baseline.learn_stage(stage, successful_runs)

        # Track tool combinations and parameters from learning
        self._update_training_statistics(learning_runs)

        # 2. Evaluation phase
        eval_runs = self._run_evaluation_phase(stage)
        eval_reward = self._compute_average_reward(eval_runs)
        pass_k_rates = self._compute_pass_k(eval_runs, stage.num_eval_trials)

        # 3. Retention phase
        retention_runs = []
        retention_reward = 0.0
        if stage.retention_tasks:
            retention_runs = self._run_retention_phase(stage)
            retention_reward = self._compute_average_reward(retention_runs)

        # 4. Compute tool performance
        tool_performance = self._compute_tool_performance(
            eval_runs + retention_runs, stage
        )
        new_tool_success = self._compute_new_tool_success(eval_runs, stage)

        # 5. Determine if stage passed
        passed = eval_reward >= stage.min_pass_rate

        return StageResult(
            stage_id=stage.stage_id,
            stage_name=stage.stage_name,
            learning_runs=learning_runs,
            learning_reward=learning_reward,
            eval_runs=eval_runs,
            eval_reward=eval_reward,
            pass_k_rates=pass_k_rates,
            retention_runs=retention_runs,
            retention_reward=retention_reward,
            tool_performance=tool_performance,
            new_tool_success_rate=new_tool_success,
            passed=passed,
        )

    def _run_learning_phase(self, stage: LearningStage) -> list[SimulationRun]:
        """Run the learning phase for a stage."""
        if self.verbose:
            print(f"\n  Learning Phase: {len(stage.learning_tasks)} tasks")

        runs = []
        experiences = []  # Collect experiences for batch learning modes

        for task_id in stage.learning_tasks:
            task = self.curriculum.get_task(task_id)
            if task is None:
                logger.warning(f"Task {task_id} not found, skipping")
                continue

            for trial in range(stage.num_learning_trials):
                run = self._execute_task(task, stage, is_learning=True)
                runs.append(run)

                # Track tool performance
                self.tool_tracker.add_evaluations_from_run(run, task, stage.stage_id)

                # For GRPO_ONLINE mode: update immediately after each experience
                if self.training_mode == TrainingMode.GRPO_ONLINE and self.grpo_agent:
                    experience = self._run_to_experience(run, task, stage.stage_id)
                    update_stats = self.grpo_agent.learn_single_experience(experience, stage)
                    if self.verbose and update_stats.get("status") == "updated":
                        print(f"    [GRPO] Updated after task {task_id}, "
                              f"loss: {update_stats.get('loss', 0):.4f}, "
                              f"total updates: {update_stats.get('total_updates', 0)}")
                else:
                    # Collect experience for batch learning
                    experience = self._run_to_experience(run, task, stage.stage_id)
                    experiences.append(experience)

        # For batch GRPO mode: update after all experiences collected
        if self.training_mode == TrainingMode.GRPO and self.grpo_agent and experiences:
            learn_stats = self.grpo_agent.learn(stage, experiences)
            if self.verbose:
                print(f"    [GRPO] Batch update: {learn_stats.get('num_updates', 0)} updates, "
                      f"avg loss: {learn_stats.get('avg_loss', 0):.4f}")

        return runs

    def _run_to_experience(
        self,
        run: SimulationRun,
        task: Task,
        stage_id: str,
    ) -> Experience:
        """Convert a SimulationRun to an Experience."""
        from tau2.data_model.message import ToolCall

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

    def _run_evaluation_phase(self, stage: LearningStage) -> list[SimulationRun]:
        """Run the evaluation phase for a stage."""
        if self.verbose:
            print(f"  Evaluation Phase: {len(stage.eval_tasks)} tasks")

        runs = []
        for task_id in stage.eval_tasks:
            task = self.curriculum.get_task(task_id)
            if task is None:
                logger.warning(f"Task {task_id} not found, skipping")
                continue

            for trial in range(stage.num_eval_trials):
                run = self._execute_task(task, stage, is_learning=False)
                runs.append(run)

                # Track tool performance
                self.tool_tracker.add_evaluations_from_run(run, task, stage.stage_id)

        return runs

    def _run_retention_phase(self, stage: LearningStage) -> list[SimulationRun]:
        """Run the retention phase for a stage."""
        if self.verbose:
            print(f"  Retention Phase: {len(stage.retention_tasks)} tasks")

        runs = []
        for task_id in stage.retention_tasks:
            task = self.curriculum.get_task(task_id)
            if task is None:
                logger.warning(f"Task {task_id} not found, skipping")
                continue

            for trial in range(stage.num_eval_trials):
                run = self._execute_task(task, stage, is_learning=False)
                runs.append(run)

                # Track tool performance
                self.tool_tracker.add_evaluations_from_run(run, task, stage.stage_id)

        return runs

    def _get_task_domain(self, task: Task) -> str:
        """Get the domain of a task from its user_scenario."""
        # Try to get domain from user_scenario.instructions
        if hasattr(task.user_scenario, 'instructions'):
            instructions = task.user_scenario.instructions
            # Check if instructions is a StructuredUserInstructions object
            if hasattr(instructions, 'domain'):
                return instructions.domain

        # Fallback to self.domain if we can't determine the task's domain
        return self.domain

    def _execute_task(
        self,
        task: Task,
        stage: LearningStage,
        is_learning: bool,
    ) -> SimulationRun:
        """Execute a single task."""
        # Build system prompt with learning materials if using ICL
        llm_args = dict(self.llm_args_agent)

        # Get the domain for this specific task
        task_domain = self._get_task_domain(task)

        # Run the task using tau2's run_task
        run = run_task(
            domain=task_domain,
            task=task,
            agent=self.agent_type,
            user=self.user_type,
            llm_agent=self.agent_llm,
            llm_args_agent=llm_args,
            llm_user=self.user_llm,
            llm_args_user=self.llm_args_user,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
            evaluation_type=EvaluationType.ALL,
            seed=self.seed,
        )

        return run

    def _compute_average_reward(self, runs: list[SimulationRun]) -> float:
        """Compute average reward from runs."""
        if not runs:
            return 0.0
        rewards = [r.reward_info.reward if r.reward_info else 0.0 for r in runs]
        return sum(rewards) / len(rewards)

    def _compute_pass_k(
        self,
        runs: list[SimulationRun],
        num_trials: int,
    ) -> dict[int, float]:
        """Compute Pass@k metrics."""
        # Group runs by task
        task_runs: dict[str, list[SimulationRun]] = defaultdict(list)
        for run in runs:
            task_runs[run.task_id].append(run)

        pass_k = {}
        for k in range(1, num_trials + 1):
            pass_rates = []
            for task_id, task_runs_list in task_runs.items():
                # Count successes
                successes = sum(
                    1 for r in task_runs_list
                    if r.reward_info and r.reward_info.reward > 0
                )
                n = len(task_runs_list)

                # Compute pass@k using combinatorial formula
                if n >= k:
                    # P(at least one success in k trials)
                    # = 1 - P(all failures in k trials)
                    # = 1 - C(n-s, k) / C(n, k)
                    from math import comb
                    if successes >= k:
                        pass_rate = 1.0
                    elif n - successes >= k:
                        pass_rate = 1.0 - comb(n - successes, k) / comb(n, k)
                    else:
                        pass_rate = 1.0
                    pass_rates.append(pass_rate)

            pass_k[k] = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0

        return pass_k

    def _compute_tool_performance(
        self,
        runs: list[SimulationRun],
        stage: LearningStage,
    ) -> dict[str, ToolPerformance]:
        """Compute tool performance metrics."""
        performance = {}

        for tool_name in stage.available_tools:
            accuracy = self.tool_tracker.get_tool_accuracy(tool_name, stage.stage_id)
            performance[tool_name] = ToolPerformance(
                tool_name=tool_name,
                selection_accuracy=accuracy["selection_accuracy"],
                invocation_accuracy=accuracy["invocation_accuracy"],
                output_usage_accuracy=accuracy["output_usage_accuracy"],
                total_calls=accuracy["num_calls"],
                correct_calls=int(accuracy["overall_accuracy"] * accuracy["num_calls"]),
            )

        return performance

    def _compute_new_tool_success(
        self,
        runs: list[SimulationRun],
        stage: LearningStage,
    ) -> float:
        """Compute success rate on new tools."""
        if not stage.new_tools:
            return 0.0

        new_tool_perf = self.tool_tracker.get_new_tool_performance(
            stage.stage_id, stage.new_tools
        )
        return new_tool_perf["overall_accuracy"]

    def _update_training_statistics(self, runs: list[SimulationRun]) -> None:
        """Update training statistics for generalization computation."""
        for run in runs:
            # Track tool combinations
            tools_used = set()
            for msg in run.messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.add(tc.name)
                        # Track parameter values
                        for param_name, param_value in tc.arguments.items():
                            key = f"{tc.name}.{param_name}"
                            self.training_param_values[key].add(str(param_value))

            if tools_used:
                self.training_tool_combinations.add(tuple(sorted(tools_used)))

    def _compute_forgetting_analysis(self) -> None:
        """Compute forgetting analysis for all tools."""
        tool_stats = self.tool_tracker.get_all_tool_stats()

        for tool_name, stats in tool_stats.items():
            forgetting_data = stats["forgetting"]
            self.results.forgetting_analysis[tool_name] = ForgettingAnalysis(
                tool_name=tool_name,
                learned_stage=stats.get("introduction_stage", "unknown"),
                learned_accuracy=forgetting_data["learned_accuracy"],
                max_accuracy=forgetting_data["max_accuracy"],
                max_accuracy_stage=forgetting_data.get("max_stage", "unknown"),
                final_accuracy=forgetting_data["final_accuracy"],
                forgetting=forgetting_data["forgetting"],
                retention=forgetting_data["retention"],
                performance_curve=forgetting_data.get("performance_curve", []),
            )

    def _compute_generalization_metrics(self) -> None:
        """Compute generalization metrics."""
        # Collect all evaluation runs
        all_eval_runs = []
        for sr in self.results.stage_results:
            all_eval_runs.extend(sr.eval_runs)

        if not all_eval_runs:
            return

        # Tool composition generalization
        comp_gen = compute_tool_composition_generalization(
            all_eval_runs, self.training_tool_combinations
        )

        # Parameter generalization
        param_gen = compute_parameter_generalization(
            all_eval_runs, dict(self.training_param_values)
        )

        self.results.generalization_metrics = GeneralizationMetrics(
            tool_composition_seen_accuracy=comp_gen["seen_accuracy"],
            tool_composition_unseen_accuracy=comp_gen["unseen_accuracy"],
            tool_composition_gap=comp_gen["generalization_gap"],
            num_unseen_combinations=comp_gen["num_unseen_combinations"],
            parameter_seen_accuracy=param_gen["seen_accuracy"],
            parameter_unseen_accuracy=param_gen["unseen_accuracy"],
            parameter_gap=param_gen["generalization_gap"],
        )

    def get_metrics(self) -> ContinualMetrics:
        """Get computed metrics."""
        return compute_continual_metrics(self.results)
