"""
Continual Benchmark Runner

This module provides the main entry point for running continual
learning benchmarks on τ²-Bench.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.data_model.continual_results import (
    ContinualLearningResults,
    TrainingMode,
)
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.evaluation.evaluator import ContinualLearningEvaluator
from tau2.continual.evaluation.metrics import ContinualMetrics, compute_continual_metrics
from tau2.run import load_tasks


class ContinualBenchmarkRunner:
    """
    Main runner for continual learning benchmarks.

    Provides a high-level interface for running continual learning
    experiments with various configurations.
    """

    def __init__(
        self,
        curriculum_path: str,
        domain: str,
        agent_llm: str,
        user_llm: Optional[str] = None,
        training_mode: str = "icl",
        output_dir: str = "./continual_results",
        agent_type: str = "llm_agent",
        user_type: str = "user_simulator",
        llm_args_agent: Optional[dict] = None,
        llm_args_user: Optional[dict] = None,
        num_eval_trials: int = 4,
        max_steps: int = 30,
        max_errors: int = 5,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        """
        Initialize the benchmark runner.

        Args:
            curriculum_path: Path to the curriculum JSON file
            domain: Domain name (e.g., "airline", "retail")
            agent_llm: LLM model for the agent
            user_llm: LLM model for the user (defaults to agent_llm)
            training_mode: Training mode ("icl", "vanilla_sft", "none")
            output_dir: Directory to save results
            agent_type: Type of agent to use
            user_type: Type of user simulator to use
            llm_args_agent: Additional LLM arguments for agent
            llm_args_user: Additional LLM arguments for user
            num_eval_trials: Number of evaluation trials per task
            max_steps: Maximum steps per simulation
            max_errors: Maximum errors before termination
            seed: Random seed
            verbose: Whether to print progress
        """
        self.curriculum_path = Path(curriculum_path)
        self.domain = domain
        self.agent_llm = agent_llm
        self.user_llm = user_llm or agent_llm
        self.training_mode = TrainingMode(training_mode)
        self.output_dir = Path(output_dir)
        self.agent_type = agent_type
        self.user_type = user_type
        self.llm_args_agent = llm_args_agent or {"temperature": 0.0}
        self.llm_args_user = llm_args_user or {"temperature": 0.0}
        self.num_eval_trials = num_eval_trials
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.seed = seed
        self.verbose = verbose

        # Load curriculum
        self.curriculum = Curriculum.from_json(self.curriculum_path)

        # Validate domain matches
        if self.curriculum.domain != self.domain:
            logger.warning(
                f"Curriculum domain ({self.curriculum.domain}) does not match "
                f"specified domain ({self.domain})"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: Optional[ContinualLearningResults] = None
        self.metrics: Optional[ContinualMetrics] = None

    def run(self) -> ContinualLearningResults:
        """
        Run the complete continual learning benchmark.

        Returns:
            ContinualLearningResults with all evaluation data
        """
        if self.verbose:
            self._print_config()

        # Load tasks for the domain
        tasks = load_tasks(task_set_name=self.domain)

        if self.verbose:
            print(f"\nLoaded {len(tasks)} tasks from domain '{self.domain}'")

        # Create evaluator
        evaluator = ContinualLearningEvaluator(
            curriculum=self.curriculum,
            domain=self.domain,
            agent_llm=self.agent_llm,
            user_llm=self.user_llm,
            agent_type=self.agent_type,
            user_type=self.user_type,
            llm_args_agent=self.llm_args_agent,
            llm_args_user=self.llm_args_user,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
            seed=self.seed,
            training_mode=self.training_mode,
            verbose=self.verbose,
        )

        # Set tasks
        evaluator.set_tasks(tasks)

        # Run evaluation
        self.results = evaluator.run_full_evaluation()

        # Compute metrics
        self.metrics = compute_continual_metrics(self.results)

        # Save results
        self._save_results()

        if self.verbose:
            self._print_summary()

        return self.results

    def _print_config(self) -> None:
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("Continual Learning Benchmark Configuration")
        print("=" * 60)
        print(f"Curriculum: {self.curriculum.curriculum_name}")
        print(f"  ID: {self.curriculum.curriculum_id}")
        print(f"  Type: {self.curriculum.curriculum_type}")
        print(f"  Stages: {len(self.curriculum.stages)}")
        print(f"  Total Tools: {self.curriculum.total_tools}")
        print(f"  Total Tasks: {self.curriculum.total_tasks}")
        print(f"\nDomain: {self.domain}")
        print(f"Agent LLM: {self.agent_llm}")
        print(f"User LLM: {self.user_llm}")
        print(f"Training Mode: {self.training_mode.value}")
        print(f"Eval Trials: {self.num_eval_trials}")
        print(f"Seed: {self.seed}")
        print(f"Output Dir: {self.output_dir}")
        print("=" * 60)

    def _print_summary(self) -> None:
        """Print results summary."""
        if self.metrics is None:
            return

        print("\n" + self.metrics.summary())

        # Print stage-by-stage summary
        if self.results:
            print("\nStage-by-Stage Results:")
            print("-" * 60)
            for sr in self.results.stage_results:
                status = "✓" if sr.passed else "✗"
                print(
                    f"  {status} {sr.stage_name}: "
                    f"Eval={sr.eval_reward:.3f}, "
                    f"Retention={sr.retention_reward:.3f}, "
                    f"NewTools={sr.new_tool_success_rate:.3f}"
                )

    def _save_results(self) -> None:
        """Save results to files."""
        if self.results is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.curriculum.curriculum_id}_{self.agent_llm.replace('/', '_')}_{timestamp}"

        # Save full results
        results_path = self.output_dir / f"{base_name}_results.json"
        self.results.save(results_path)

        # Save metrics summary
        if self.metrics:
            metrics_path = self.output_dir / f"{base_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics.to_dict(), f, indent=2)

        # Save configuration
        config_path = self.output_dir / f"{base_name}_config.json"
        config = {
            "curriculum_path": str(self.curriculum_path),
            "curriculum_id": self.curriculum.curriculum_id,
            "domain": self.domain,
            "agent_llm": self.agent_llm,
            "user_llm": self.user_llm,
            "training_mode": self.training_mode.value,
            "agent_type": self.agent_type,
            "user_type": self.user_type,
            "llm_args_agent": self.llm_args_agent,
            "llm_args_user": self.llm_args_user,
            "num_eval_trials": self.num_eval_trials,
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "seed": self.seed,
            "timestamp": timestamp,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"\nResults saved to:")
            print(f"  - {results_path}")
            print(f"  - {metrics_path}")
            print(f"  - {config_path}")

    def get_results(self) -> Optional[ContinualLearningResults]:
        """Get the evaluation results."""
        return self.results

    def get_metrics(self) -> Optional[ContinualMetrics]:
        """Get the computed metrics."""
        return self.metrics

    @classmethod
    def from_config(cls, config_path: str) -> "ContinualBenchmarkRunner":
        """
        Create a runner from a configuration file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            Configured ContinualBenchmarkRunner
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(
            curriculum_path=config["curriculum_path"],
            domain=config["domain"],
            agent_llm=config["agent_llm"],
            user_llm=config.get("user_llm"),
            training_mode=config.get("training_mode", "icl"),
            output_dir=config.get("output_dir", "./continual_results"),
            agent_type=config.get("agent_type", "llm_agent"),
            user_type=config.get("user_type", "user_simulator"),
            llm_args_agent=config.get("llm_args_agent"),
            llm_args_user=config.get("llm_args_user"),
            num_eval_trials=config.get("num_eval_trials", 4),
            max_steps=config.get("max_steps", 30),
            max_errors=config.get("max_errors", 5),
            seed=config.get("seed", 42),
            verbose=config.get("verbose", True),
        )


def run_continual_benchmark(
    curriculum_path: str,
    domain: str,
    agent_llm: str,
    user_llm: Optional[str] = None,
    training_mode: str = "icl",
    output_dir: str = "./continual_results",
    seed: int = 42,
    verbose: bool = True,
) -> ContinualLearningResults:
    """
    Convenience function to run a continual learning benchmark.

    Args:
        curriculum_path: Path to curriculum JSON file
        domain: Domain name
        agent_llm: Agent LLM model
        user_llm: User LLM model (defaults to agent_llm)
        training_mode: Training mode
        output_dir: Output directory
        seed: Random seed
        verbose: Print progress

    Returns:
        ContinualLearningResults
    """
    runner = ContinualBenchmarkRunner(
        curriculum_path=curriculum_path,
        domain=domain,
        agent_llm=agent_llm,
        user_llm=user_llm,
        training_mode=training_mode,
        output_dir=output_dir,
        seed=seed,
        verbose=verbose,
    )
    return runner.run()
