#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨域持续学习测试

按照 Airline → Retail → Telecom 依次训练和评测
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from tau2.continual.runner import ContinualBenchmarkRunner
from tau2.run import load_tasks


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="跨域持续学习测试")

    # 模型配置
    parser.add_argument(
        "--use-local-model",
        action="store_true",
        help="使用本地 HuggingFace 模型（不需要 API key）"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="本地模型路径或 HuggingFace 模型名称"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="使用 4-bit 量化加载模型（节省显存）"
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="gpt-4",
        help="API 模型名称（当不使用本地模型时）"
    )

    # 评测配置
    parser.add_argument(
        "--num-eval-trials",
        type=int,
        default=1,
        help="每个任务的评测次数"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="每个任务的最大步数"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("="*80)
    print("跨域持续学习测试")
    print("="*80)
    print("\n配置:")
    print("  - 课程: curriculum_simple_test.json")
    print("  - 顺序: Airline → Retail → Telecom")
    print("  - 训练模式: none (不使用持续学习方法)")
    print(f"  - 评测次数: 每个任务{args.num_eval_trials}次")

    if args.use_local_model:
        print(f"\n模型配置: 本地模型")
        print(f"  - 模型: {args.model_path}")
        print(f"  - 4-bit 量化: {'是' if args.load_in_4bit else '否'}")
    else:
        print(f"\n模型配置: API 模型")
        print(f"  - 模型: {args.api_model}")
    print()

    try:
        # 加载所有域的任务
        print("加载任务...")
        all_tasks = []
        for domain_name in ['airline', 'retail', 'telecom']:
            print(f"  - 加载 {domain_name} 域...")
            domain_tasks = load_tasks(task_set_name=domain_name)
            all_tasks.extend(domain_tasks)
            print(f"    已加载 {len(domain_tasks)} 个任务")

        print(f"\n总共加载 {len(all_tasks)} 个任务")

        # 创建 runner
        if args.use_local_model:
            # 使用本地 HuggingFace 模型
            runner = ContinualBenchmarkRunner(
                curriculum_path='curriculum_simple_test.json',
                domain='multi_domain',  # 跨域
                agent_llm=None,
                agent_type='hf_agent',
                user_type='hf_user_simulator',
                llm_args_agent={
                    'model_name_or_path': args.model_path,
                    'load_in_4bit': args.load_in_4bit,
                },
                llm_args_user={
                    'model_name_or_path': args.model_path,
                    'load_in_4bit': args.load_in_4bit,
                },
                training_mode='none',
                output_dir='./results_cross_domain',
                num_eval_trials=args.num_eval_trials,
                max_steps=args.max_steps,
                seed=42,
                verbose=True
            )
        else:
            # 使用 API 模型
            runner = ContinualBenchmarkRunner(
                curriculum_path='curriculum_simple_test.json',
                domain='multi_domain',
                agent_llm=args.api_model,
                training_mode='none',
                output_dir='./results_cross_domain',
                num_eval_trials=args.num_eval_trials,
                max_steps=args.max_steps,
                seed=42,
                verbose=True
            )

        # 设置任务缓存
        runner.curriculum.set_task_cache(all_tasks)

        # 创建 evaluator
        from tau2.continual.evaluation.evaluator import ContinualLearningEvaluator

        evaluator = ContinualLearningEvaluator(
            curriculum=runner.curriculum,
            domain='multi_domain',
            agent_llm=runner.agent_llm,
            user_llm=runner.user_llm,
            agent_type=runner.agent_type,
            user_type=runner.user_type,
            llm_args_agent=runner.llm_args_agent,
            llm_args_user=runner.llm_args_user,
            max_steps=runner.max_steps,
            max_errors=runner.max_errors,
            seed=runner.seed,
            training_mode=runner.training_mode,
            verbose=runner.verbose,
        )

        evaluator.set_tasks(all_tasks)

        print("\n" + "="*80)
        print("开始运行评测...")
        print("="*80)

        # 运行评测
        results = evaluator.run_full_evaluation()

        # 计算指标
        from tau2.continual.evaluation.metrics import compute_continual_metrics
        metrics = compute_continual_metrics(results)

        # 保存结果
        runner.results = results
        runner.metrics = metrics
        runner._save_results()

        if runner.verbose:
            runner._print_summary()

        print("\n" + "="*80)
        print("评测完成！")
        print("="*80)
        print(f"\n结果已保存到: ./results_cross_domain")

        return results

    except Exception as e:
        print(f"\n[ERROR] 运行错误: {e}")
        print("\n详细错误信息:")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

    if results:
        print("\n[SUCCESS] 测试成功完成！")
        sys.exit(0)
    else:
        print("\n[FAILED] 测试失败，请查看错误信息")
        sys.exit(1)

