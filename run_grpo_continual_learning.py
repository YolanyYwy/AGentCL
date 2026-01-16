import argparse
import json
from pathlib import Path

from tau2.continual.baselines.grpo_agent import GRPOContinualAgent, GRPOConfig
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
        stages=stages,
        description="Train sequentially across multiple domains using GRPO",
    )


def load_domain_tasks(domain: str, data_dir: str = "data/tau2/domains") -> list[Task]:
    """Load tasks for a specific domain."""
    tasks_path = Path(data_dir) / domain / "tasks.json"

    if not tasks_path.exists():
        print(f"Warning: Tasks file not found at {tasks_path}")
        return []

    with open(tasks_path, "r") as f:
        tasks_data = json.load(f)

    tasks = []
    for task_data in tasks_data:
        task = Task.model_validate(task_data)
        tasks.append(task)

    return tasks


def run_grpo_continual_learning(
    domains: list[str],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
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
    print(f"Learning rate: {learning_rate}")
    print(f"Beta (KL penalty): {beta}")
    print(f"Group size: {group_size}")
    print(f"Online mode: {online_mode}")
    print("=" * 60)

    # Create GRPO config
    grpo_config = GRPOConfig(
        model_name_or_path=model_name,
        learning_rate=learning_rate,
        beta=beta,
        group_size=group_size,
        update_after_each_experience=online_mode,
        save_every_n_updates=10,
    )

    # Create GRPO agent
    print("\nInitializing GRPO agent...")
    agent = GRPOContinualAgent(config=grpo_config)

    # Create curriculum
    print("\nCreating curriculum...")
    curriculum = create_domain_curriculum(domains)

    # Load tasks for all domains
    print("\nLoading tasks...")
    all_tasks = []
    for domain in domains:
        tasks = load_domain_tasks(domain)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} tasks from {domain}")

    if not all_tasks:
        print("Warning: No tasks loaded. Using dummy tasks for demonstration.")
        # Create dummy tasks for demonstration
        all_tasks = _create_dummy_tasks(domains)

    # Create evaluator
    training_mode = TrainingMode.GRPO_ONLINE if online_mode else TrainingMode.GRPO
    evaluator = ContinualLearningEvaluator(
        curriculum=curriculum,
        domain=domains[0],  # Primary domain
        agent_llm=model_name,
        user_llm="gpt-4",  # User simulator
        training_mode=training_mode,
        verbose=verbose,
    )

    # Set tasks
    evaluator.set_tasks(all_tasks)

    # Run evaluation
    print("\nStarting continual learning...")
    print("=" * 60)

    for stage_idx, stage in enumerate(curriculum.stages):
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{len(curriculum.stages)}: {stage.stage_name}")
        print(f"{'='*60}")

        # Notify agent of stage start
        agent.on_stage_start(stage)

        # Run learning phase with immediate updates
        print(f"\n  Learning Phase: {len(stage.learning_tasks)} tasks")
        for task_id in stage.learning_tasks:
            task = curriculum.get_task(task_id)
            if task is None:
                continue

            # Execute task (this would normally use run_task)
            # For demonstration, we create a dummy experience
            experience = _create_dummy_experience(task_id, stage.stage_id)

            # Immediate GRPO update
            if online_mode:
                stats = agent.learn_single_experience(experience, stage)
                if verbose and stats.get("status") == "updated":
                    print(f"    Updated after {task_id}: "
                          f"loss={stats.get('loss', 0):.4f}, "
                          f"updates={stats.get('total_updates', 0)}")

        # Batch update if not online mode
        if not online_mode:
            experiences = [
                _create_dummy_experience(tid, stage.stage_id)
                for tid in stage.learning_tasks
            ]
            stats = agent.learn(stage, experiences)
            print(f"    Batch update: {stats.get('num_updates', 0)} updates")

        # Notify agent of stage end
        agent.on_stage_end(stage, {"evaluation": {"average_reward": 0.5}})

        # Save checkpoint
        checkpoint_path = Path(output_dir) / f"checkpoint_stage_{stage_idx}"
        agent.save_checkpoint(str(checkpoint_path))
        print(f"  Checkpoint saved to {checkpoint_path}")

        # Update reference model at end of each domain
        agent.update_reference_model()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total updates: {agent.total_updates}")
    print("=" * 60)

    return agent


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


def _create_dummy_experience(task_id: str, stage_id: str):
    """Create a dummy experience for demonstration."""
    from tau2.continual.benchmark.agent_interface import Experience
    from tau2.data_model.message import UserMessage, AssistantMessage

    return Experience(
        task_id=task_id,
        messages=[
            UserMessage(role="user", content="Hello, I need help."),
            AssistantMessage(role="assistant", content="How can I help you?"),
        ],
        tool_calls=[],
        tool_results=[],
        reward=1.0,  # Successful experience
        success=True,
        expected_actions=[],
        stage_id=stage_id,
    )


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
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or path",
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
        learning_rate=args.learning_rate,
        beta=args.beta,
        group_size=args.group_size,
        online_mode=not args.batch_mode,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
