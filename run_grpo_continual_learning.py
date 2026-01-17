
import argparse
import json
from pathlib import Path

from tau2.continual.training.grpo_trainer import GRPOContinualTrainer, GRPOTrainingConfig
from tau2.continual.evaluation.evaluator import ContinualLearningEvaluator
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.curriculum.stage import LearningStage
from tau2.data_model.continual_results import TrainingMode
from tau2.data_model.tasks import Task


def create_domain_curriculum(
    domains: list[str],
    tasks_per_domain: int = 10,
    learning_tasks_ratio: float = 0.6,
) -> Curriculum:
    """
    Create a curriculum for sequential domain training.

    Args:
        domains: List of domain names (e.g., ["airline", "retail", "telecom"])
        tasks_per_domain: Number of tasks per domain
        learning_tasks_ratio: Ratio of tasks used for learning vs evaluation

    Returns:
        Curriculum object
    """
    stages = []

    for i, domain in enumerate(domains):
        num_learning = int(tasks_per_domain * learning_tasks_ratio)
        num_eval = tasks_per_domain - num_learning

        # Create stage for this domain
        stage = LearningStage(
            stage_id=f"domain_{domain}",
            stage_name=f"Domain: {domain.capitalize()}",
            learning_tasks=[f"{domain}_task_{j}" for j in range(num_learning)],
            eval_tasks=[f"{domain}_task_{j}" for j in range(num_learning, tasks_per_domain)],
            retention_tasks=[
                f"{prev_domain}_task_0"
                for prev_domain in domains[:i]
            ] if i > 0 else [],
            new_tools=[f"{domain}_tool_1", f"{domain}_tool_2"],
            available_tools=[
                f"{d}_tool_{t}"
                for d in domains[:i+1]
                for t in [1, 2]
            ],
            num_learning_trials=1,
            num_eval_trials=2,
            min_pass_rate=0.5,
        )
        stages.append(stage)

    return Curriculum(
        curriculum_id="grpo_sequential_domains",
        curriculum_name="GRPO Sequential Domain Training",
        domain=domains[0],  # Primary domain
        stages=stages,
        description="Train sequentially across multiple domains using GRPO",
    )


def run_grpo_continual_learning(
    domains: list[str],
    model_name: str = "Qwen/Qwen3-4B-Instruct",
    device: str = "auto",
    learning_rate: float = 1e-6,
    beta: float = 0.1,
    group_size: int = 4,
    online_mode: bool = True,
    output_dir: str = "./grpo_results",
    verbose: bool = True,
):
    """
    Run GRPO-based continual learning across multiple domains.

    Args:
        domains: List of domain names to train on sequentially
        model_name: HuggingFace model name or path
        device: Device to use ('auto', 'cuda', 'cpu')
        learning_rate: Learning rate for GRPO
        beta: KL penalty coefficient
        group_size: Number of samples per prompt for GRPO
        online_mode: If True, update after each experience; if False, batch update
        output_dir: Directory to save results
        verbose: Whether to print progress
    """
    print("=" * 60)
    print("GRPO Continual Learning")
    print("=" * 60)
    print(f"Domains: {domains}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Beta (KL penalty): {beta}")
    print(f"Group size: {group_size}")
    print(f"Online mode: {online_mode}")
    print("=" * 60)

    # Create GRPO config
    grpo_config = GRPOTrainingConfig(
        model_name_or_path=model_name,
        device=device,
        learning_rate=learning_rate,
        beta=beta,
        group_size=group_size,
        output_dir=output_dir,
    )

    # Create GRPO trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOContinualTrainer(config=grpo_config)

    # Load model
    print("Loading model...")
    trainer.load_model()

    # Create curriculum
    print("\nCreating curriculum...")
    curriculum = create_domain_curriculum(domains)

    # Create dummy tasks for demonstration
    print("\nCreating dummy tasks for demonstration...")
    all_tasks = _create_dummy_tasks(domains)

    # Run training
    print("\n" + "=" * 60)
    print("Starting continual learning...")
    print("=" * 60)

    for stage_idx, stage in enumerate(curriculum.stages):
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{len(curriculum.stages)}: {stage.stage_name}")
        print(f"{'='*60}")

        # Create dummy runs for this stage
        dummy_runs = _create_dummy_runs(stage.learning_tasks)

        if online_mode:
            # Online mode: update after each experience
            print(f"\n  Learning Phase (Online): {len(dummy_runs)} experiences")
            for run in dummy_runs:
                stats = trainer.train_on_experience(run, stage.stage_id)
                if verbose and stats.get("status") == "updated":
                    print(f"    Updated: loss={stats.get('loss', 0):.4f}, "
                          f"total={stats.get('total_updates', 0)}")
        else:
            # Batch mode: update after all experiences
            print(f"\n  Learning Phase (Batch): {len(dummy_runs)} experiences")
            stats = trainer.train_stage(stage.stage_id, dummy_runs)
            print(f"    Batch update: {stats.get('num_updates', 0)} updates, "
                  f"avg_loss={stats.get('avg_loss', 0):.4f}")

        # Save checkpoint after each stage
        checkpoint_path = Path(output_dir) / f"stage_{stage.stage_id}"
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Update reference model at end of each domain
        trainer.update_reference_model()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total updates: {trainer.total_updates}")
    print(f"Stage updates: {trainer.stage_updates}")
    print("=" * 60)

    return trainer


def _create_dummy_tasks(domains: list[str]) -> list[Task]:
    """Create dummy tasks for demonstration."""
    from tau2.data_model.tasks import Task, UserScenario, EvaluationCriteria

    tasks = []
    for domain in domains:
        for i in range(10):
            task = Task(
                id=f"{domain}_task_{i}",
                user_scenario=UserScenario(
                    instructions=f"Test task {i} for {domain}",
                ),
                evaluation_criteria=EvaluationCriteria(
                    actions=[],
                ),
            )
            tasks.append(task)
    return tasks


def _create_dummy_runs(task_ids: list[str]) -> list:
    """Create dummy simulation runs for demonstration."""
    from tau2.data_model.simulation import SimulationRun, RewardInfo, TerminationReason
    from tau2.data_model.message import UserMessage, AssistantMessage
    import uuid
    from datetime import datetime

    runs = []
    for task_id in task_ids:
        now = datetime.now().isoformat()
        run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task_id,
            start_time=now,
            end_time=now,
            duration=1.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=[
                UserMessage(role="user", content="Hello, I need help with my booking."),
                AssistantMessage(role="assistant", content="I'd be happy to help you with your booking. Could you please provide your booking reference number?"),
                UserMessage(role="user", content="My reference is ABC123."),
                AssistantMessage(role="assistant", content="Thank you. I found your booking. How can I assist you today?"),
            ],
            reward_info=RewardInfo(reward=1.0, info={}),
        )
        runs.append(run)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Run GRPO-based continual learning"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["airline", "retail", "telecom"],
        help="Domains to train on sequentially",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for GRPO",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty coefficient",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of samples per prompt for GRPO",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use batch updates instead of online updates",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./grpo_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    run_grpo_continual_learning(
        domains=args.domains,
        model_name=args.model,
        device=args.device,
        learning_rate=args.learning_rate,
        beta=args.beta,
        group_size=args.group_size,
        online_mode=not args.batch_mode,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
