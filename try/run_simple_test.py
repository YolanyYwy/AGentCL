#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行简单的顺序训练测试

使用现有的 ContinualBenchmarkRunner 框架
支持本地模型和 API 模型
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from tau2.continual.runner import ContinualBenchmarkRunner


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行简单顺序训练测试")

    # 课程配置
    parser.add_argument(
        "--curriculum",
        type=str,
        default="curriculum_simple_test.json",
        help="课程配置文件路径"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="airline",
        help="域名 (airline, retail, telecom)"
    )

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
    print("运行简单顺序训练测试")
    print("="*80)
    print("\n配置:")
    print(f"  - 课程: {args.curriculum}")
    print(f"  - 域: {args.domain}")
    print("  - 训练模式: none (不使用持续学习方法)")
    print(f"  - 评测次数: 每个任务{args.num_eval_trials}次")

    if args.use_local_model:
        print(f"\n模型配置: 本地模型")
        print(f"  - 模型: {args.model_path}")
        print(f"  - 4-bit 量化: {'是' if args.load_in_4bit else '否'}")
        print(f"  - 优点: 不需要 API key，数据隐私")
        print(f"  - 注意: 需要 GPU 和足够的显存")
    else:
        print(f"\n模型配置: API 模型")
        print(f"  - 模型: {args.api_model}")
        print(f"  - 注意: 需要配置 API key (如 OPENAI_API_KEY)")
    print()

    try:
        # 根据配置创建 runner
        if args.use_local_model:
            # 使用本地 HuggingFace 模型
            runner = ContinualBenchmarkRunner(
                curriculum_path=args.curriculum,
                domain=args.domain,
                agent_llm=None,  # 本地模型不需要 API
                agent_type='hf_agent',  # 使用 HuggingFace agent
                user_type='hf_user_simulator',  # 使用 HuggingFace user simulator
                llm_args_agent={
                    'model_name_or_path': args.model_path,
                    'load_in_4bit': args.load_in_4bit,
                },
                llm_args_user={
                    'model_name_or_path': args.model_path,
                    'load_in_4bit': args.load_in_4bit,
                },
                training_mode='none',  # 不使用持续学习方法
                output_dir='./results_simple_test',
                num_eval_trials=args.num_eval_trials,
                max_steps=args.max_steps,
                seed=42,
                verbose=True
            )
        else:
            # 使用 API 模型
            runner = ContinualBenchmarkRunner(
                curriculum_path=args.curriculum,
                domain=args.domain,
                agent_llm=args.api_model,
                training_mode='none',  # 不使用持续学习方法
                output_dir='./results_simple_test',
                num_eval_trials=args.num_eval_trials,
                max_steps=args.max_steps,
                seed=42,
                verbose=True
            )

        print("\n" + "="*80)
        print("开始运行评测...")
        print("="*80)

        # 运行评测
        results = runner.run()

        print("\n" + "="*80)
        print("评测完成！")
        print("="*80)
        print(f"\n结果已保存到: ./results_simple_test")

        return results

    except ImportError as e:
        print(f"\n[ERROR] 导入错误: {e}")
        print("\n可能的解决方案:")
        print("  1. 安装项目依赖: pip install -e .")
        print("  2. 检查 Python 路径是否正确")
        return None

    except FileNotFoundError as e:
        print(f"\n[ERROR] 文件未找到: {e}")
        print("\n可能的解决方案:")
        print("  1. 先运行 simple_sequential_test.py 生成课程配置")
        print("  2. 检查数据文件是否存在")
        return None

    except Exception as e:
        print(f"\n[ERROR] 运行错误: {e}")
        print("\n详细错误信息:")
        import traceback
        traceback.print_exc()
        print("\n可能的解决方案:")
        print("  1. 检查 LLM API key 是否配置 (如 OPENAI_API_KEY)")
        print("  2. 检查网络连接")
        print("  3. 查看上面的详细错误信息")
        return None


if __name__ == "__main__":
    results = main()

    if results:
        print("\n[SUCCESS] 测试成功完成！")
        sys.exit(0)
    else:
        print("\n[FAILED] 测试失败，请查看错误信息")
        sys.exit(1)
