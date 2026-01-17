#!/usr/bin/env python3
"""
三域持续学习训练脚本
按照 Airline → Retail → Telecom 顺序进行持续学习训练
并计算前向迁移（Forward Transfer）和后向迁移（Backward Transfer）指标
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from tau2.continual.training.grpo_trainer import GRPOContinualTrainer, GRPOTrainingConfig
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.curriculum.stage import LearningStage
from tau2.continual.evaluation.evaluator import ContinualLearningEvaluator
from tau2.continual.evaluation.metrics import compute_continual_metrics
from tau2.data_model.continual_results import TrainingMode, ContinualLearningResults
from tau2.run import load_tasks
from loguru import logger


def create_three_domain_curriculum(
    airline_tasks_path: str,
    retail_tasks_path: str,
    telecom_tasks_path: str,
    tasks_per_domain: int = 100,
    learning_ratio: float = 0.6,
) -> Curriculum:
    """
    Args:
        airline_tasks_path: Airline 任务文件路径
        retail_tasks_path: Retail 任务文件路径
        telecom_tasks_path: Telecom 任务文件路径
        tasks_per_domain: 每个域使用的任务数量
        learning_ratio: 用于学习的任务比例（其余用于评估）
    Returns:
        Curriculum 对象
    """
    # 加载任务
    with open(airline_tasks_path, 'r', encoding='utf-8') as f:
        airline_tasks = json.load(f)[:tasks_per_domain]

    with open(retail_tasks_path, 'r', encoding='utf-8') as f:
        retail_tasks = json.load(f)[:tasks_per_domain]

    with open(telecom_tasks_path, 'r', encoding='utf-8') as f:
        telecom_tasks = json.load(f)[:tasks_per_domain]

    logger.info(f"加载任务数量: Airline={len(airline_tasks)}, Retail={len(retail_tasks)}, Telecom={len(telecom_tasks)}")

    # 计算学习和评估任务数量
    num_learning = int(tasks_per_domain * learning_ratio)
    num_eval = tasks_per_domain - num_learning

    stages = []

    # Stage 1: Airline
    airline_learning_ids = [t['id'] for t in airline_tasks[:num_learning]]
    airline_eval_ids = [t['id'] for t in airline_tasks[num_learning:tasks_per_domain]]

    stage1 = LearningStage(
        stage_id="stage_1_airline",
        stage_name="Stage 1: Airline Domain",
        learning_tasks=airline_learning_ids,
        eval_tasks=airline_eval_ids,
        retention_tasks=[],  # 第一阶段没有保留任务
        new_tools=[],  # 工具信息将从任务中提取
        available_tools=[],
        num_learning_trials=1,
        num_eval_trials=4,
        min_pass_rate=0.5,
    )
    stages.append(stage1)

    # Stage 2: Retail (+ Airline retention)
    retail_learning_ids = [t['id'] for t in retail_tasks[:num_learning]]
    retail_eval_ids = [t['id'] for t in retail_tasks[num_learning:tasks_per_domain]]
    airline_retention_ids = airline_eval_ids[:min(10, len(airline_eval_ids))]  # 保留10个airline任务

    stage2 = LearningStage(
        stage_id="stage_2_retail",
        stage_name="Stage 2: Retail Domain",
        learning_tasks=retail_learning_ids,
        eval_tasks=retail_eval_ids,
        retention_tasks=airline_retention_ids,
        new_tools=[],
        available_tools=[],
        num_learning_trials=1,
        num_eval_trials=4,
        min_pass_rate=0.5,
    )
    stages.append(stage2)

    # Stage 3: Telecom (+ Airline + Retail retention)
    telecom_learning_ids = [t['id'] for t in telecom_tasks[:num_learning]]
    telecom_eval_ids = [t['id'] for t in telecom_tasks[num_learning:tasks_per_domain]]
    retail_retention_ids = retail_eval_ids[:min(10, len(retail_eval_ids))]

    stage3 = LearningStage(
        stage_id="stage_3_telecom",
        stage_name="Stage 3: Telecom Domain",
        learning_tasks=telecom_learning_ids,
        eval_tasks=telecom_eval_ids,
        retention_tasks=airline_retention_ids + retail_retention_ids,  # 保留前两个域的任务
        new_tools=[],
        available_tools=[],
        num_learning_trials=1,
        num_eval_trials=4,
        min_pass_rate=0.5,
    )
    stages.append(stage3)

    curriculum = Curriculum(
        curriculum_id="three_domain_continual",
        curriculum_name="Three Domain Continual Learning (Airline → Retail → Telecom)",
        domain="multi_domain",
        stages=stages,
        description="Sequential training across Airline, Retail, and Telecom domains with GRPO",
    )

    return curriculum


def run_three_domain_training(
    airline_tasks_path: str,
    retail_tasks_path: str,
    telecom_tasks_path: str,
    model_name: str = "Qwen/Qwen3-4B",
    device: str = "cuda",
    learning_rate: float = 1e-6,
    beta: float = 0.1,
    group_size: int = 4,
    tasks_per_domain: int = 100,
    output_dir: str = "./three_domain_results",
    use_grpo: bool = True,
    verbose: bool = True,
):
    """
    运行三域持续学习训练

    Args:
        airline_tasks_path: Airline 任务文件路径
        retail_tasks_path: Retail 任务文件路径
        telecom_tasks_path: Telecom 任务文件路径
        model_name: 模型名称
        device: 设备
        learning_rate: 学习率
        beta: KL 惩罚系数
        group_size: GRPO group size
        tasks_per_domain: 每个域的任务数量
        output_dir: 输出目录
        use_grpo: 是否使用 GRPO 训练
        verbose: 是否打印详细信息
    """
    print("=" * 80)
    print("三域持续学习训练")
    print("=" * 80)
    print(f"训练顺序: Airline → Retail → Telecom")
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"每域任务数: {tasks_per_domain}")
    print(f"使用 GRPO: {use_grpo}")
    if use_grpo:
        print(f"学习率: {learning_rate}")
        print(f"Beta: {beta}")
        print(f"Group Size: {group_size}")
    print("=" * 80)
    print()

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建课程
    print("📚 创建课程...")
    curriculum = create_three_domain_curriculum(
        airline_tasks_path=airline_tasks_path,
        retail_tasks_path=retail_tasks_path,
        telecom_tasks_path=telecom_tasks_path,
        tasks_per_domain=tasks_per_domain,
        learning_ratio=0.6,
    )

    # 保存课程配置
    curriculum_path = output_path / "curriculum.json"
    curriculum.to_json(curriculum_path)
    print(f"✅ 课程已保存: {curriculum_path}")
    print()

    # 加载所有任务
    print("📥 加载任务数据...")
    all_tasks = {}

    with open(airline_tasks_path, 'r', encoding='utf-8') as f:
        airline_tasks = json.load(f)[:tasks_per_domain]
        for task in airline_tasks:
            all_tasks[task['id']] = task

    with open(retail_tasks_path, 'r', encoding='utf-8') as f:
        retail_tasks = json.load(f)[:tasks_per_domain]
        for task in retail_tasks:
            all_tasks[task['id']] = task

    with open(telecom_tasks_path, 'r', encoding='utf-8') as f:
        telecom_tasks = json.load(f)[:tasks_per_domain]
        for task in telecom_tasks:
            all_tasks[task['id']] = task

    print(f"✅ 总任务数: {len(all_tasks)}")
    print()

    # 初始化 GRPO 训练器（如果使用）
    trainer = None
    if use_grpo:
        print("🔧 初始化 GRPO 训练器...")
        grpo_config = GRPOTrainingConfig(
            model_name_or_path=model_name,
            device=device,
            learning_rate=learning_rate,
            beta=beta,
            group_size=group_size,
            output_dir=str(output_path / "grpo_checkpoints"),
        )
        trainer = GRPOContinualTrainer(config=grpo_config)
        trainer.load_model()
        print("✅ GRPO 训练器已初始化")
        print()

    # 创建评估器
    print("🔧 初始化评估器...")
    evaluator = ContinualLearningEvaluator(
        curriculum=curriculum,
        domain="multi_domain",
        agent_llm=None,  # 使用本地模型
        user_llm=None,
        agent_type='hf_agent' if use_grpo else 'llm_agent',
        user_type='hf_user_simulator' if use_grpo else 'user_simulator',
        llm_args_agent={
            'model_name_or_path': model_name,
            'load_in_4bit': False,
        } if use_grpo else {},
        llm_args_user={
            'model_name_or_path': model_name,
            'load_in_4bit': False,
        } if use_grpo else {},
        max_steps=30,
        max_errors=5,
        seed=42,
        training_mode=TrainingMode.NONE,  # 我们手动控制训练
        verbose=verbose,
    )

    # 设置任务
    evaluator.set_tasks(list(all_tasks.values()))
    print("✅ 评估器已初始化")
    print()

    # 运行持续学习训练
    print("=" * 80)
    print("🚀 开始持续学习训练...")
    print("=" * 80)
    print()

    results = ContinualLearningResults(
        curriculum_id=curriculum.curriculum_id,
        curriculum_name=curriculum.curriculum_name,
        domain=curriculum.domain,
        training_mode=TrainingMode.NONE,
        start_time=datetime.now().isoformat(),
        stage_results=[],
    )

    for stage_idx, stage in enumerate(curriculum.stages):
        print(f"\n{'='*80}")
        print(f"📍 {stage.stage_name} ({stage_idx + 1}/{len(curriculum.stages)})")
        print(f"{'='*80}")
        print(f"学习任务: {len(stage.learning_tasks)}")
        print(f"评估任务: {len(stage.eval_tasks)}")
        print(f"保留任务: {len(stage.retention_tasks)}")
        print()

        # 1. 学习阶段
        if use_grpo and trainer and stage.learning_tasks:
            print(f"📖 学习阶段: 在 {len(stage.learning_tasks)} 个任务上训练...")

            # 运行学习任务并收集轨迹
            learning_runs = evaluator._run_tasks(
                task_ids=stage.learning_tasks,
                num_trials=stage.num_learning_trials,
                stage_id=stage.stage_id,
                phase="learning",
            )

            # 使用 GRPO 训练
            for run in learning_runs:
                if run.reward_info and run.reward_info.reward > 0:
                    stats = trainer.train_on_experience(run, stage.stage_id)
                    if verbose and stats.get("status") == "updated":
                        print(f"  ✓ 更新: loss={stats.get('loss', 0):.4f}, total={stats.get('total_updates', 0)}")

            # 保存检查点
            checkpoint_path = output_path / "grpo_checkpoints" / f"stage_{stage.stage_id}"
            trainer.save_checkpoint(str(checkpoint_path))
            print(f"  💾 检查点已保存: {checkpoint_path}")

            # 更新参考模型
            trainer.update_reference_model()
            print(f"  🔄 参考模型已更新")

        # 2. 评估阶段
        print(f"\n📊 评估阶段: 在 {len(stage.eval_tasks)} 个任务上评估...")
        eval_runs = evaluator._run_tasks(
            task_ids=stage.eval_tasks,
            num_trials=stage.num_eval_trials,
            stage_id=stage.stage_id,
            phase="eval",
        )

        eval_reward = sum(r.reward_info.reward for r in eval_runs if r.reward_info) / len(eval_runs) if eval_runs else 0.0
        print(f"  评估奖励: {eval_reward:.4f}")

        # 3. 保留任务评估（测试后向迁移）
        retention_runs = []
        retention_reward = 0.0
        if stage.retention_tasks:
            print(f"\n🔄 保留任务评估: 在 {len(stage.retention_tasks)} 个任务上评估...")
            retention_runs = evaluator._run_tasks(
                task_ids=stage.retention_tasks,
                num_trials=stage.num_eval_trials,
                stage_id=stage.stage_id,
                phase="retention",
            )
            retention_reward = sum(r.reward_info.reward for r in retention_runs if r.reward_info) / len(retention_runs) if retention_runs else 0.0
            print(f"  保留任务奖励: {retention_reward:.4f}")

        # 保存阶段结果
        from tau2.data_model.continual_results import StageResult
        stage_result = StageResult(
            stage_id=stage.stage_id,
            stage_name=stage.stage_name,
            learning_runs=learning_runs if use_grpo else [],
            eval_runs=eval_runs,
            retention_runs=retention_runs,
            eval_reward=eval_reward,
            retention_reward=retention_reward,
            pass_k_rates={1: eval_reward, 4: eval_reward},  # 简化版
            new_tool_success_rate=eval_reward,
            tool_performance={},
        )
        results.stage_results.append(stage_result)

        print(f"\n✅ {stage.stage_name} 完成")

    # 完成训练
    results.end_time = datetime.now().isoformat()

    # 计算持续学习指标
    print("\n" + "=" * 80)
    print("📈 计算持续学习指标...")
    print("=" * 80)

    metrics = compute_continual_metrics(results)

    # 打印指标
    print(metrics.summary())

    # 保存结果
    results_path = output_path / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存: {results_path}")

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"💾 指标已保存: {metrics_path}")

    print("\n" + "=" * 80)
    print("🎉 三域持续学习训练完成！")
    print("=" * 80)

    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="三域持续学习训练 (Airline → Retail → Telecom)"
    )

    # 任务文件路径
    parser.add_argument(
        "--airline-tasks",
        type=str,
        default="data/tau2/domains/airline/tasks.json",
        help="Airline 任务文件路径",
    )
    parser.add_argument(
        "--retail-tasks",
        type=str,
        default="data/tau2/domains/retail/tasks.json",
        help="Retail 任务文件路径",
    )
    parser.add_argument(
        "--telecom-tasks",
        type=str,
        default="data/tau2/domains/telecom/tasks_hard_300.json",
        help="Telecom 任务文件路径",
    )

    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="模型名称",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="设备",
    )

    # GRPO 配置
    parser.add_argument(
        "--use-grpo",
        action="store_true",
        default=True,
        help="使用 GRPO 训练",
    )
    parser.add_argument(
        "--no-grpo",
        action="store_true",
        help="不使用 GRPO 训练（仅评估）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="学习率",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL 惩罚系数",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="GRPO group size",
    )

    # 任务配置
    parser.add_argument(
        "--tasks-per-domain",
        type=int,
        default=100,
        help="每个域使用的任务数量",
    )

    # 输出配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./three_domain_results",
        help="输出目录",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式",
    )

    args = parser.parse_args()

    # 处理 no-grpo 标志
    use_grpo = args.use_grpo and not args.no_grpo

    try:
        results, metrics = run_three_domain_training(
            airline_tasks_path=args.airline_tasks,
            retail_tasks_path=args.retail_tasks,
            telecom_tasks_path=args.telecom_tasks,
            model_name=args.model,
            device=args.device,
            learning_rate=args.learning_rate,
            beta=args.beta,
            group_size=args.group_size,
            tasks_per_domain=args.tasks_per_domain,
            output_dir=args.output_dir,
            use_grpo=use_grpo,
            verbose=not args.quiet,
        )

        print("\n✅ 训练成功完成！")
        print(f"\n关键指标:")
        print(f"  平均奖励: {metrics.average_reward:.4f}")
        print(f"  前向迁移: {metrics.forward_transfer:.4f}")
        print(f"  后向迁移: {metrics.backward_transfer:.4f}")
        print(f"  平均遗忘: {metrics.average_forgetting:.4f}")

        return 0

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
