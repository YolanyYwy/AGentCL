#!/usr/bin/env python3
"""
三域持续学习训练脚本 - 分布式数据并行（DDP）版本
使用 PyTorch DDP 实现真正的多卡联训
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from tau2.data_model.tasks import Task
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.continual_results import TrainingMode, ContinualLearningResults, StageResult
from tau2.continual.curriculum.curriculum import Curriculum
from tau2.continual.curriculum.stage import LearningStage
from tau2.continual.evaluation.metrics import compute_continual_metrics
from tau2.run import run_task
from tau2.evaluator.evaluator import EvaluationType
from loguru import logger


def setup_ddp(rank: int, world_size: int):
    """
    初始化分布式训练环境

    Args:
        rank: 当前进程的 rank（GPU 编号）
        world_size: 总进程数（GPU 数量）
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # NVIDIA GPU 使用 nccl
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"✅ DDP 初始化完成: {world_size} 个 GPU")


def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def load_tasks_from_json(json_path: str, max_tasks: int = None) -> list[Task]:
    """从 JSON 文件加载任务"""
    with open(json_path, 'r', encoding='utf-8') as f:
        tasks_data = json.load(f)

    if max_tasks:
        tasks_data = tasks_data[:max_tasks]

    tasks = []
    for task_dict in tasks_data:
        try:
            task = Task(**task_dict)
            tasks.append(task)
        except Exception as e:
            logger.warning(f"跳过无效任务: {e}")
            continue

    return tasks


class ExperienceDataset(Dataset):
    """
    经验回放数据集
    用于 DDP 训练的数据集包装
    """
    def __init__(self, experiences: List[SimulationRun]):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        return self.experiences[idx]


def collate_experiences(batch):
    """
    自定义 collate 函数
    因为 SimulationRun 对象不能直接 batch，所以返回列表
    """
    return batch


def run_task_single_gpu(task: Task, domain: str, agent_type: str, user_type: str,
                       model_name: str, llm_args_agent: dict, llm_args_user: dict,
                       max_steps: int, max_errors: int, seed: int, rank: int):
    """在单个 GPU 上运行任务"""
    try:
        run = run_task(
            domain=domain,
            task=task,
            agent=agent_type,
            user=user_type,
            llm_agent=None,
            llm_args_agent=llm_args_agent,
            llm_user=None,
            llm_args_user=llm_args_user,
            max_steps=max_steps,
            max_errors=max_errors,
            evaluation_type=EvaluationType.ALL,
            seed=seed,
        )
        return run
    except Exception as e:
        if rank == 0:
            logger.error(f"任务 {task.id} 运行失败: {e}")
        return None


def gather_all_runs(runs: List[SimulationRun], rank: int, world_size: int) -> List[SimulationRun]:
    """
    收集所有 GPU 的运行结果到 rank 0

    Args:
        runs: 当前 GPU 的运行结果
        rank: 当前进程 rank
        world_size: 总进程数

    Returns:
        所有 GPU 的运行结果（仅在 rank 0 返回完整列表）
    """
    # 使用 all_gather 收集所有 GPU 的结果
    gathered_runs = [None] * world_size
    dist.all_gather_object(gathered_runs, runs)

    if rank == 0:
        # 在 rank 0 上合并所有结果
        all_runs = []
        for gpu_runs in gathered_runs:
            if gpu_runs:
                all_runs.extend(gpu_runs)
        return all_runs
    else:
        return []


def train_with_ddp(
    rank: int,
    world_size: int,
    curriculum: Curriculum,
    task_map: dict,
    model_name: str,
    agent_type: str,
    user_type: str,
    llm_args_agent: dict,
    llm_args_user: dict,
    learning_rate: float,
    beta: float,
    group_size: int,
    output_dir: str,
    use_grpo: bool,
    verbose: bool,
):
    """
    DDP 训练主函数
    每个 GPU 运行一个进程

    Args:
        rank: 当前进程的 rank（GPU 编号）
        world_size: 总进程数（GPU 数量）
        其他参数同主函数
    """
    # 1. 初始化 DDP
    setup_ddp(rank, world_size)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"🚀 开始 DDP 训练 (Rank {rank}/{world_size})")
        print(f"{'='*80}\n")

    # 2. 初始化 GRPO 训练器（每个 GPU 一个副本）
    trainer = None
    if use_grpo:
        from tau2.continual.training.grpo_trainer import GRPOContinualTrainer, GRPOTrainingConfig

        grpo_config = GRPOTrainingConfig(
            model_name_or_path=model_name,
            device=f"cuda:{rank}",
            learning_rate=learning_rate,
            beta=beta,
            group_size=group_size,
            output_dir=str(Path(output_dir) / "grpo_checkpoints"),
            torch_dtype="bfloat16",
        )
        trainer = GRPOContinualTrainer(config=grpo_config)
        trainer.load_model()

        # 将模型包装为 DDP
        trainer.model = DDP(
            trainer.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )

        # 参考模型也需要在当前 GPU 上
        if trainer.ref_model is not None:
            trainer.ref_model = trainer.ref_model.to(f"cuda:{rank}")

        if rank == 0:
            print(f"✅ GRPO 训练器已初始化并包装为 DDP")

    # 3. 创建结果存储（仅 rank 0）
    results = None
    if rank == 0:
        results = ContinualLearningResults(
            curriculum_id=curriculum.curriculum_id,
            curriculum_name=curriculum.curriculum_name,
            domain=curriculum.domain,
            training_mode=TrainingMode.NONE,
            start_time=datetime.now().isoformat(),
            stage_results=[],
        )

    # 4. 遍历每个阶段
    for stage_idx, stage in enumerate(curriculum.stages):
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"📍 {stage.stage_name} ({stage_idx + 1}/{len(curriculum.stages)})")
            print(f"{'='*80}")

        # 确定当前阶段的域
        if "airline" in stage.stage_id:
            domain = "airline"
        elif "retail" in stage.stage_id:
            domain = "retail"
        else:
            domain = "telecom"

        # ============================================
        # 阶段 1: 学习阶段（DDP 并行收集经验）
        # ============================================
        learning_runs = []
        if stage.learning_tasks:
            if rank == 0:
                print(f"\n📖 学习阶段: {len(stage.learning_tasks)} 个任务")

            # 获取学习任务
            learning_tasks = [task_map[tid] for tid in stage.learning_tasks if tid in task_map]

            # 使用 DistributedSampler 分配任务到不同 GPU
            # 每个 GPU 处理一部分任务
            tasks_per_gpu = len(learning_tasks) // world_size
            start_idx = rank * tasks_per_gpu
            end_idx = start_idx + tasks_per_gpu if rank < world_size - 1 else len(learning_tasks)
            my_tasks = learning_tasks[start_idx:end_idx]

            if rank == 0:
                print(f"  每个 GPU 处理 ~{tasks_per_gpu} 个任务")

            # 每个 GPU 运行自己的任务
            my_runs = []
            for task in my_tasks:
                run = run_task_single_gpu(
                    task, domain, agent_type, user_type, model_name,
                    llm_args_agent, llm_args_user, 30, 5, 42, rank
                )
                if run:
                    my_runs.append(run)

            # 同步：等待所有 GPU 完成任务收集
            dist.barrier()

            # 收集所有 GPU 的运行结果
            learning_runs = gather_all_runs(my_runs, rank, world_size)

            if rank == 0:
                print(f"  ✅ 收集到 {len(learning_runs)} 个经验")

            # ============================================
            # 阶段 2: GRPO 训练（DDP 同步训练）
            # ============================================
            if use_grpo and trainer and learning_runs:
                if rank == 0:
                    print(f"\n  🔧 GRPO DDP 训练...")

                # 过滤成功的经验
                successful_runs = [r for r in learning_runs if r.reward_info and r.reward_info.reward > 0]

                if len(successful_runs) > 0:
                    # 创建数据集和分布式采样器
                    dataset = ExperienceDataset(successful_runs)
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True
                    )

                    # 创建 DataLoader
                    dataloader = DataLoader(
                        dataset,
                        batch_size=group_size,
                        sampler=sampler,
                        collate_fn=collate_experiences,
                        num_workers=0,  # 避免多进程冲突
                    )

                    # DDP 训练循环
                    total_updates = 0
                    total_loss = 0.0

                    for batch_idx, batch_runs in enumerate(dataloader):
                        # 每个 GPU 处理自己的 batch
                        batch_loss = 0.0

                        for run in batch_runs:
                            # 在当前 GPU 上训练
                            stats = trainer.train_on_experience(run, stage.stage_id)

                            if stats.get("status") == "updated":
                                batch_loss += stats.get("loss", 0.0)
                                total_updates += 1

                        # 计算平均 loss
                        if len(batch_runs) > 0:
                            batch_loss /= len(batch_runs)
                            total_loss += batch_loss

                        # DDP 会自动进行梯度的 All-Reduce
                        # 所有 GPU 的梯度会被平均并同步

                        if rank == 0 and verbose and batch_idx % 5 == 0:
                            print(f"    Batch {batch_idx}: loss={batch_loss:.4f}, updates={total_updates}")

                    # 同步：等待所有 GPU 完成训练
                    dist.barrier()

                    # 计算全局平均 loss（使用 All-Reduce）
                    avg_loss_tensor = torch.tensor([total_loss / max(len(dataloader), 1)], device=f"cuda:{rank}")
                    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                    global_avg_loss = avg_loss_tensor.item()

                    if rank == 0:
                        print(f"  ✅ DDP 训练完成: {total_updates} 次更新, 全局平均 loss={global_avg_loss:.4f}")

                    # 保存检查点（仅 rank 0）
                    if rank == 0:
                        checkpoint_path = Path(output_dir) / "grpo_checkpoints" / f"stage_{stage.stage_id}"
                        # 保存 DDP 模型时需要访问 module
                        original_model = trainer.model
                        trainer.model = trainer.model.module  # 解包 DDP
                        trainer.save_checkpoint(str(checkpoint_path))
                        trainer.model = original_model  # 恢复 DDP
                        print(f"  💾 检查点已保存: {checkpoint_path}")

                    # 同步检查点保存
                    dist.barrier()

                    # 更新参考模型
                    if trainer.ref_model is not None:
                        if rank == 0:
                            # 只在 rank 0 更新参考模型
                            trainer.update_reference_model()
                        # 广播参考模型到所有 GPU
                        dist.barrier()

        # ============================================
        # 阶段 3: 评估阶段（DDP 并行评估）
        # ============================================
        if rank == 0:
            print(f"\n📊 评估阶段: {len(stage.eval_tasks)} 个任务")

        eval_tasks = [task_map[tid] for tid in stage.eval_tasks if tid in task_map]

        # 分配评估任务
        tasks_per_gpu = len(eval_tasks) // world_size
        start_idx = rank * tasks_per_gpu
        end_idx = start_idx + tasks_per_gpu if rank < world_size - 1 else len(eval_tasks)
        my_eval_tasks = eval_tasks[start_idx:end_idx]

        # 每个任务运行多次 trial
        my_eval_runs = []
        for task in my_eval_tasks:
            for trial in range(stage.num_eval_trials):
                run = run_task_single_gpu(
                    task, domain, agent_type, user_type, model_name,
                    llm_args_agent, llm_args_user, 30, 5, 42 + trial, rank
                )
                if run:
                    my_eval_runs.append(run)

        # 同步并收集评估结果
        dist.barrier()
        eval_runs = gather_all_runs(my_eval_runs, rank, world_size)

        eval_reward = 0.0
        if rank == 0 and eval_runs:
            eval_reward = sum(r.reward_info.reward for r in eval_runs if r.reward_info) / len(eval_runs)
            print(f"  评估奖励: {eval_reward:.4f}")

        # ============================================
        # 阶段 4: 保留任务评估（DDP 并行）
        # ============================================
        retention_runs = []
        retention_reward = 0.0

        if stage.retention_tasks:
            if rank == 0:
                print(f"\n🔄 保留任务评估: {len(stage.retention_tasks)} 个任务")

            retention_tasks = [task_map[tid] for tid in stage.retention_tasks if tid in task_map]

            # 分配保留任务
            tasks_per_gpu = len(retention_tasks) // world_size
            start_idx = rank * tasks_per_gpu
            end_idx = start_idx + tasks_per_gpu if rank < world_size - 1 else len(retention_tasks)
            my_retention_tasks = retention_tasks[start_idx:end_idx]

            # 确定保留任务的域
            retention_domain = "airline"
            if retention_tasks:
                tid = retention_tasks[0].id
                if "retail" in tid.lower():
                    retention_domain = "retail"
                elif "telecom" in tid.lower():
                    retention_domain = "telecom"

            my_retention_runs = []
            for task in my_retention_tasks:
                for trial in range(stage.num_eval_trials):
                    run = run_task_single_gpu(
                        task, retention_domain, agent_type, user_type, model_name,
                        llm_args_agent, llm_args_user, 30, 5, 42 + trial, rank
                    )
                    if run:
                        my_retention_runs.append(run)

            dist.barrier()
            retention_runs = gather_all_runs(my_retention_runs, rank, world_size)

            if rank == 0 and retention_runs:
                retention_reward = sum(r.reward_info.reward for r in retention_runs if r.reward_info) / len(retention_runs)
                print(f"  保留任务奖励: {retention_reward:.4f}")

        # ============================================
        # 保存阶段结果（仅 rank 0）
        # ============================================
        if rank == 0:
            stage_result = StageResult(
                stage_id=stage.stage_id,
                stage_name=stage.stage_name,
                learning_runs=learning_runs,
                eval_runs=eval_runs,
                retention_runs=retention_runs,
                eval_reward=eval_reward,
                retention_reward=retention_reward,
                pass_k_rates={1: eval_reward, 4: eval_reward},
                new_tool_success_rate=eval_reward,
                tool_performance={},
            )
            results.stage_results.append(stage_result)
            print(f"\n✅ {stage.stage_name} 完成")

        # 同步所有 GPU
        dist.barrier()

    # ============================================
    # 计算最终指标（仅 rank 0）
    # ============================================
    if rank == 0:
        results.end_time = datetime.now().isoformat()

        print("\n" + "=" * 80)
        print("📈 计算持续学习指标...")
        print("=" * 80)

        metrics = compute_continual_metrics(results)
        print(metrics.summary())

        # 保存结果
        output_path = Path(output_dir)
        results_path = output_path / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存: {results_path}")

        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"💾 指标已保存: {metrics_path}")

        print("\n" + "=" * 80)
        print("🎉 DDP 训练完成！")
        print("=" * 80)

        return results, metrics
    else:
        return None, None


def create_three_domain_curriculum(
    airline_tasks: list[Task],
    retail_tasks: list[Task],
    telecom_tasks: list[Task],
    learning_ratio: float = 0.6,
) -> Curriculum:
    """创建三域持续学习课程"""
    def split_tasks(tasks, ratio):
        num_learning = int(len(tasks) * ratio)
        return tasks[:num_learning], tasks[num_learning:]

    airline_learning, airline_eval = split_tasks(airline_tasks, learning_ratio)
    retail_learning, retail_eval = split_tasks(retail_tasks, learning_ratio)
    telecom_learning, telecom_eval = split_tasks(telecom_tasks, learning_ratio)

    stages = [
        LearningStage(
            stage_id="stage_1_airline",
            stage_name="Stage 1: Airline Domain",
            learning_tasks=[t.id for t in airline_learning],
            eval_tasks=[t.id for t in airline_eval],
            retention_tasks=[],
            new_tools=[], available_tools=[],
            num_learning_trials=1, num_eval_trials=4, min_pass_rate=0.5,
        ),
        LearningStage(
            stage_id="stage_2_retail",
            stage_name="Stage 2: Retail Domain",
            learning_tasks=[t.id for t in retail_learning],
            eval_tasks=[t.id for t in retail_eval],
            retention_tasks=[t.id for t in airline_eval[:min(10, len(airline_eval))]],
            new_tools=[], available_tools=[],
            num_learning_trials=1, num_eval_trials=4, min_pass_rate=0.5,
        ),
        LearningStage(
            stage_id="stage_3_telecom",
            stage_name="Stage 3: Telecom Domain",
            learning_tasks=[t.id for t in telecom_learning],
            eval_tasks=[t.id for t in telecom_eval],
            retention_tasks=[t.id for t in airline_eval[:min(10, len(airline_eval))]] +
                          [t.id for t in retail_eval[:min(10, len(retail_eval))]],
            new_tools=[], available_tools=[],
            num_learning_trials=1, num_eval_trials=4, min_pass_rate=0.5,
        ),
    ]

    return Curriculum(
        curriculum_id="three_domain_ddp",
        curriculum_name="Three Domain Continual Learning - DDP",
        domain="multi_domain",
        stages=stages,
        description="DDP training with gradient synchronization",
    )


def main():
    parser = argparse.ArgumentParser(description="三域持续学习 - DDP 分布式训练")
    parser.add_argument("--airline-tasks", type=str, default="data/tau2/domains/airline/tasks.json")
    parser.add_argument("--retail-tasks", type=str, default="data/tau2/domains/retail/tasks.json")
    parser.add_argument("--telecom-tasks", type=str, default="data/tau2/domains/telecom/tasks_hard_300.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--num-gpus", type=int, default=2, help="使用的 GPU 数量")
    parser.add_argument("--use-grpo", action="store_true", default=True)
    parser.add_argument("--no-grpo", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--tasks-per-domain", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./three_domain_results_ddp")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    use_grpo = args.use_grpo and not args.no_grpo

    # 主进程：加载数据和创建课程
    print("=" * 80)
    print("三域持续学习训练 - DDP 分布式数据并行")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"GPU 数量: {args.num_gpus}")
    print(f"每域任务数: {args.tasks_per_domain}")
    print(f"使用 GRPO: {use_grpo}")
    print("=" * 80)
    print()

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载任务
    print("📥 加载任务数据...")
    airline_tasks = load_tasks_from_json(args.airline_tasks, args.tasks_per_domain)
    retail_tasks = load_tasks_from_json(args.retail_tasks, args.tasks_per_domain)
    telecom_tasks = load_tasks_from_json(args.telecom_tasks, args.tasks_per_domain)

    print(f"✅ Airline: {len(airline_tasks)} 个任务")
    print(f"✅ Retail: {len(retail_tasks)} 个任务")
    print(f"✅ Telecom: {len(telecom_tasks)} 个任务")
    print()

    # 创建课程
    print("📚 创建课程...")
    curriculum = create_three_domain_curriculum(
        airline_tasks, retail_tasks, telecom_tasks, learning_ratio=0.6
    )

    curriculum_path = output_path / "curriculum.json"
    curriculum.to_json(curriculum_path)
    print(f"✅ 课程已保存: {curriculum_path}")
    print()

    # 创建任务映射
    all_tasks = airline_tasks + retail_tasks + telecom_tasks
    task_map = {task.id: task for task in all_tasks}

    # 配置参数
    agent_type = 'hf_agent'
    user_type = 'hf_user_simulator'
    llm_args_agent = {
        'model_name_or_path': args.model,
        'load_in_4bit': True,
        'torch_dtype': 'bfloat16',
    }
    llm_args_user = {
        'model_name_or_path': args.model,
        'load_in_4bit': True,
        'torch_dtype': 'bfloat16',
    }

    # 启动 DDP 训练
    # 使用 mp.spawn 启动多个进程
    try:
        mp.spawn(
            train_with_ddp,
            args=(
                args.num_gpus,
                curriculum,
                task_map,
                args.model,
                agent_type,
                user_type,
                llm_args_agent,
                llm_args_user,
                args.learning_rate,
                args.beta,
                args.group_size,
                args.output_dir,
                use_grpo,
                not args.quiet,
            ),
            nprocs=args.num_gpus,
            join=True
        )

        print("\n✅ DDP 训练成功完成！")
        return 0

    except Exception as e:
        print(f"\n❌ DDP 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
