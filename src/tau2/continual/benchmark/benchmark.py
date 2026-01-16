"""
Main Benchmark Class

This is the primary entry point for running continual learning evaluations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tau2.data_model.tasks import Task
from tau2.data_model.continual_results import (
    ContinualLearningResults,
    StageResult,
    ToolPerformance,
    ForgettingAnalysis,
    GeneralizationMetrics,
)
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.curriculum.task_selector import TaskSelector
from tau2.continual.benchmark.agent_interface import ContinualAgent, Experience
from tau2.continual.benchmark.protocol import EvaluationProtocol, ProtocolConfig
from tau2.continual.benchmark.metrics import (
    MetricsComputer,
    ContinualMetrics,
)


class ContinualBenchmark:
    """
    Main benchmark class for evaluating continual learning agents.

    This class provides:
    - Standard evaluation protocol
    - Metric computation
    - Result saving and loading

    Usage:
    ```python
    # Load benchmark
    benchmark = ContinualBenchmark.from_curriculum(
        curriculum_path="path/to/curriculum.json",
        domain="airline",
    )

    # Create your agent
    agent = YourContinualAgent(...)

    # Run evaluation
    results = benchmark.evaluate(agent)

    # View metrics
    print(results.metrics.summary())
    ```
    """

    def __init__(
        self,
        curriculum: Curriculum,
        tasks: list[Task],
        tools: list[dict],
        protocol_config: Optional[ProtocolConfig] = None,
        output_dir: str = "./benchmark_results",
    ):
        """
        Initialize the benchmark.

        Args:
            curriculum: The curriculum defining learning stages
            tasks: List of all tasks
            tools: List of tool schemas
            protocol_config: Configuration for evaluation protocol
            output_dir: Directory to save results
        """
        self.curriculum = curriculum
        self.tasks = {task.id: task for task in tasks}
        self.tools = tools
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.task_selector = TaskSelector(tasks)
        self.protocol = EvaluationProtocol(
            config=protocol_config or ProtocolConfig(),
            task_executor=self._execute_task,
        )
        self.metrics_computer = MetricsComputer()

        # Set task cache in curriculum
        self.curriculum.set_task_cache(tasks)

    @classmethod
    def from_curriculum(
        cls,
        curriculum_path: str,
        domain: str,
        protocol_config: Optional[ProtocolConfig] = None,
        output_dir: str = "./benchmark_results",
    ) -> "ContinualBenchmark":
        """
        Create a benchmark from a curriculum file.

        Args:
            curriculum_path: Path to curriculum JSON file
            domain: Domain name (e.g., "airline")
            protocol_config: Optional protocol configuration
            output_dir: Output directory for results

        Returns:
            Configured ContinualBenchmark instance
        """
        # Load curriculum
        curriculum = Curriculum.from_json(curriculum_path)

        # Load tasks for domain
        from tau2.run import load_tasks
        tasks = load_tasks(task_set_name=domain)

        # Load tools for domain
        from tau2.run import get_environment_info
        env_info = get_environment_info(domain, include_tool_info=True)
        tools = env_info.tools if env_info.tools else []

        return cls(
            curriculum=curriculum,
            tasks=tasks,
            tools=tools,
            protocol_config=protocol_config,
            output_dir=output_dir,
        )

    def evaluate(
        self,
        agent: ContinualAgent,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ) -> ContinualLearningResults:
        """
        Run the complete benchmark evaluation.

        Args:
            agent: The continual learning agent to evaluate
            seed: Random seed for reproducibility
            verbose: Whether to print progress

        Returns:
            ContinualLearningResults with all metrics
        """
        if verbose:
            self._print_header(agent)

        # Initialize results
        results = ContinualLearningResults(
            curriculum_id=self.curriculum.curriculum_id,
            curriculum_name=self.curriculum.curriculum_name,
            agent_config=agent.get_config(),
        )

        # Track tool performance across stages
        tool_performance_history = {}

        # Run each stage
        for stage_idx, stage in enumerate(self.curriculum.stages):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Stage {stage_idx + 1}/{len(self.curriculum.stages)}: {stage.stage_name}")
                print(f"  Tools: {len(stage.available_tools)} ({len(stage.new_tools)} new)")
                print(f"{'='*60}")

            # Get tools for this stage
            stage_tools = [t for t in self.tools if t.get("name") in stage.available_tools]

            # Run stage protocol
            stage_results = self.protocol.run_stage(
                stage=stage,
                agent=agent,
                tasks=self.tasks,
                tools=stage_tools,
            )

            # Convert to StageResult
            stage_result = self._create_stage_result(stage, stage_results)
            results.stage_results.append(stage_result)

            # Track tool performance
            self._update_tool_performance_history(
                tool_performance_history, stage, stage_results
            )

            if verbose:
                self._print_stage_summary(stage_result)

            # Check if stage passed
            if not stage_result.passed:
                if verbose:
                    print(f"\n  Stage not passed (reward {stage_result.eval_reward:.3f} < {stage.min_pass_rate})")
                    print("  Stopping evaluation.")
                break

        # Compute final metrics
        results.forgetting_analysis = self._compute_forgetting_analysis(
            tool_performance_history
        )
        results.overall_metrics = self.metrics_computer.compute_all(results)

        # Save results
        self._save_results(results, agent)

        if verbose:
            self._print_final_summary(results)

        return results

    def _execute_task(
        self,
        task: Task,
        agent: ContinualAgent,
        tools: list[dict],
    ):
        """
        Execute a single task with the agent.

        This is the bridge between the benchmark and the agent.
        """
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

        # Create a wrapper that uses the agent
        # This needs to be implemented based on how tau2 orchestrator works
        # For now, we'll use a simplified version

        # TODO: Implement proper integration with tau2 orchestrator
        # This requires creating an adapter that wraps ContinualAgent
        # to work with tau2's agent interface

        raise NotImplementedError(
            "Task execution requires integration with tau2 orchestrator. "
            "See documentation for how to implement this."
        )

    def _create_stage_result(
        self,
        stage,
        stage_results: dict,
    ) -> StageResult:
        """Create a StageResult from protocol results."""
        eval_data = stage_results.get("evaluation", {})
        retention_data = stage_results.get("retention", {})
        learning_data = stage_results.get("learning", {})

        eval_reward = eval_data.get("average_reward", 0.0)
        retention_reward = retention_data.get("average_reward", 0.0)
        learning_reward = 0.0
        if learning_data.get("num_experiences", 0) > 0:
            learning_reward = learning_data.get("successful_experiences", 0) / learning_data["num_experiences"]

        return StageResult(
            stage_id=stage.stage_id,
            stage_name=stage.stage_name,
            learning_runs=learning_data.get("runs", []),
            learning_reward=learning_reward,
            eval_runs=eval_data.get("runs", []),
            eval_reward=eval_reward,
            retention_runs=retention_data.get("runs", []),
            retention_reward=retention_reward,
            passed=eval_reward >= stage.min_pass_rate,
        )

    def _update_tool_performance_history(
        self,
        history: dict,
        stage,
        stage_results: dict,
    ):
        """Update tool performance tracking."""
        # Extract tool usage from runs
        all_runs = (
            stage_results.get("evaluation", {}).get("runs", []) +
            stage_results.get("retention", {}).get("runs", [])
        )

        for tool_name in stage.available_tools:
            if tool_name not in history:
                history[tool_name] = []

            # Compute accuracy for this tool in this stage
            # (simplified - would need actual tool call analysis)
            history[tool_name].append({
                "stage_id": stage.stage_id,
                "accuracy": stage_results.get("evaluation", {}).get("average_reward", 0.0),
            })

    def _compute_forgetting_analysis(
        self,
        tool_performance_history: dict,
    ) -> dict[str, ForgettingAnalysis]:
        """Compute forgetting analysis from tool performance history."""
        analysis = {}

        for tool_name, history in tool_performance_history.items():
            if len(history) < 2:
                continue

            accuracies = [h["accuracy"] for h in history]
            learned_acc = accuracies[0]
            max_acc = max(accuracies)
            final_acc = accuracies[-1]

            analysis[tool_name] = ForgettingAnalysis(
                tool_name=tool_name,
                learned_stage=history[0]["stage_id"],
                learned_accuracy=learned_acc,
                max_accuracy=max_acc,
                max_accuracy_stage=history[accuracies.index(max_acc)]["stage_id"],
                final_accuracy=final_acc,
                forgetting=max(0, max_acc - final_acc),
                retention=final_acc / learned_acc if learned_acc > 0 else 0,
                performance_curve=[(h["stage_id"], h["accuracy"]) for h in history],
            )

        return analysis

    def _save_results(self, results: ContinualLearningResults, agent: ContinualAgent):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.curriculum.curriculum_id}_{timestamp}"

        # Save full results
        results_path = self.output_dir / f"{base_name}_results.json"
        results.save(results_path)

        # Save metrics summary
        metrics_path = self.output_dir / f"{base_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results.overall_metrics, f, indent=2)

        print(f"\nResults saved to: {results_path}")

    def _print_header(self, agent: ContinualAgent):
        """Print benchmark header."""
        print("\n" + "=" * 60)
        print("τ²-Bench Continual Learning Benchmark")
        print("=" * 60)
        print(f"Curriculum: {self.curriculum.curriculum_name}")
        print(f"  Stages: {len(self.curriculum.stages)}")
        print(f"  Total Tools: {self.curriculum.total_tools}")
        print(f"  Total Tasks: {self.curriculum.total_tasks}")
        print(f"Agent Config: {agent.get_config()}")
        print("=" * 60)

    def _print_stage_summary(self, stage_result: StageResult):
        """Print stage summary."""
        status = "✓ PASSED" if stage_result.passed else "✗ FAILED"
        print(f"\n  {status}")
        print(f"  Learning Reward: {stage_result.learning_reward:.4f}")
        print(f"  Eval Reward: {stage_result.eval_reward:.4f}")
        if stage_result.retention_reward > 0:
            print(f"  Retention Reward: {stage_result.retention_reward:.4f}")

    def _print_final_summary(self, results: ContinualLearningResults):
        """Print final summary."""
        metrics = results.overall_metrics

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Stages Completed: {len(results.stage_results)}/{len(self.curriculum.stages)}")
        print(f"\nKey Metrics:")
        if "basic" in metrics:
            print(f"  Average Reward: {metrics['basic'].get('average_reward', 0):.4f}")
        if "continual_learning" in metrics:
            cl = metrics["continual_learning"]
            print(f"  Forward Transfer: {cl.get('forward_transfer', 0):.4f}")
            print(f"  Backward Transfer: {cl.get('backward_transfer', 0):.4f}")
            print(f"  Average Forgetting: {cl.get('average_forgetting', 0):.4f}")
        print("=" * 60)
