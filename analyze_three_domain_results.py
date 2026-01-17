#!/usr/bin/env python3
"""
三域持续学习结果分析脚本
分析和可视化前向迁移、后向迁移等指标
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_results(results_dir: Path):
    """加载所有实验结果"""
    experiments = {}

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / "metrics.json"
        results_file = exp_dir / "results.json"

        if metrics_file.exists() and results_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            experiments[exp_dir.name] = {
                'metrics': metrics,
                'results': results,
            }

    return experiments


def print_summary_table(experiments):
    """打印汇总表格"""
    print("\n" + "=" * 120)
    print("三域持续学习实验结果汇总")
    print("=" * 120)

    # 准备数据
    data = []
    for exp_name, exp_data in experiments.items():
        metrics = exp_data['metrics']
        basic = metrics.get('basic', {})
        cl = metrics.get('continual_learning', {})
        efficiency = metrics.get('learning_efficiency', {})

        data.append({
            '实验': exp_name,
            '平均奖励': basic.get('average_reward', 0),
            '最终奖励': basic.get('final_reward', 0),
            'Pass@1': basic.get('pass_at_k', {}).get('1', 0),
            '前向迁移': cl.get('forward_transfer', 0),
            '后向迁移': cl.get('backward_transfer', 0),
            '平均遗忘': cl.get('average_forgetting', 0),
            '学习效率': efficiency.get('learning_efficiency', 0),
            'AULC': efficiency.get('aulc', 0),
        })

    df = pd.DataFrame(data)

    # 按平均奖励排序
    df = df.sort_values('平均奖励', ascending=False)

    # 打印表格
    print(df.to_string(index=False))
    print("=" * 120)

    # 找出最佳配置
    best_exp = df.iloc[0]['实验']
    print(f"\n🏆 最佳配置: {best_exp}")
    print(f"   平均奖励: {df.iloc[0]['平均奖励']:.4f}")
    print(f"   前向迁移: {df.iloc[0]['前向迁移']:.4f}")
    print(f"   后向迁移: {df.iloc[0]['后向迁移']:.4f}")

    return df


def plot_learning_curves(experiments, output_dir: Path):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 6))

    for exp_name, exp_data in experiments.items():
        results = exp_data['results']
        stage_results = results.get('stage_results', [])

        if not stage_results:
            continue

        stages = [sr['stage_name'] for sr in stage_results]
        rewards = [sr['eval_reward'] for sr in stage_results]

        plt.plot(stages, rewards, marker='o', label=exp_name, linewidth=2)

    plt.xlabel('训练阶段', fontsize=12)
    plt.ylabel('评估奖励', fontsize=12)
    plt.title('三域持续学习曲线', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 学习曲线已保存: {output_path}")
    plt.close()


def plot_transfer_metrics(experiments, output_dir: Path):
    """绘制迁移指标对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    exp_names = list(experiments.keys())
    forward_transfers = []
    backward_transfers = []

    for exp_name in exp_names:
        metrics = experiments[exp_name]['metrics']
        cl = metrics.get('continual_learning', {})
        forward_transfers.append(cl.get('forward_transfer', 0))
        backward_transfers.append(cl.get('backward_transfer', 0))

    # 前向迁移
    axes[0].barh(exp_names, forward_transfers, color='skyblue')
    axes[0].set_xlabel('前向迁移 (Forward Transfer)', fontsize=12)
    axes[0].set_title('前向迁移对比', fontsize=14, fontweight='bold')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')

    # 后向迁移
    axes[1].barh(exp_names, backward_transfers, color='lightcoral')
    axes[1].set_xlabel('后向迁移 (Backward Transfer)', fontsize=12)
    axes[1].set_title('后向迁移对比', fontsize=14, fontweight='bold')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = output_dir / "transfer_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 迁移指标对比已保存: {output_path}")
    plt.close()


def plot_stage_performance(experiments, output_dir: Path):
    """绘制各阶段性能对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    stage_names = ['Stage 1: Airline', 'Stage 2: Retail', 'Stage 3: Telecom']

    for stage_idx, stage_name in enumerate(stage_names):
        exp_names = []
        rewards = []

        for exp_name, exp_data in experiments.items():
            results = exp_data['results']
            stage_results = results.get('stage_results', [])

            if stage_idx < len(stage_results):
                exp_names.append(exp_name)
                rewards.append(stage_results[stage_idx]['eval_reward'])

        axes[stage_idx].barh(exp_names, rewards, color=f'C{stage_idx}')
        axes[stage_idx].set_xlabel('评估奖励', fontsize=11)
        axes[stage_idx].set_title(stage_name, fontsize=12, fontweight='bold')
        axes[stage_idx].set_xlim(0, 1)
        axes[stage_idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = output_dir / "stage_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 各阶段性能对比已保存: {output_path}")
    plt.close()


def plot_retention_performance(experiments, output_dir: Path):
    """绘制保留任务性能（后向迁移）"""
    plt.figure(figsize=(12, 6))

    for exp_name, exp_data in experiments.items():
        results = exp_data['results']
        stage_results = results.get('stage_results', [])

        if not stage_results:
            continue

        stages = []
        retention_rewards = []

        for sr in stage_results:
            if sr.get('retention_reward', 0) > 0:  # 只显示有保留任务的阶段
                stages.append(sr['stage_name'])
                retention_rewards.append(sr['retention_reward'])

        if stages:
            plt.plot(stages, retention_rewards, marker='s', label=exp_name, linewidth=2)

    plt.xlabel('训练阶段', fontsize=12)
    plt.ylabel('保留任务奖励', fontsize=12)
    plt.title('保留任务性能（后向迁移）', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "retention_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 保留任务性能已保存: {output_path}")
    plt.close()


def generate_report(experiments, output_dir: Path):
    """生成 Markdown 报告"""
    report_path = output_dir / "analysis_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 三域持续学习实验分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

        f.write("## 实验概述\n\n")
        f.write(f"- 训练顺序: Airline → Retail → Telecom\n")
        f.write(f"- 实验数量: {len(experiments)}\n")
        f.write(f"- 评估指标: 前向迁移、后向迁移、平均遗忘、学习效率\n\n")

        f.write("## 实验结果汇总\n\n")

        # 创建表格
        data = []
        for exp_name, exp_data in experiments.items():
            metrics = exp_data['metrics']
            basic = metrics.get('basic', {})
            cl = metrics.get('continual_learning', {})

            data.append({
                '实验': exp_name,
                '平均奖励': f"{basic.get('average_reward', 0):.4f}",
                '前向迁移': f"{cl.get('forward_transfer', 0):.4f}",
                '后向迁移': f"{cl.get('backward_transfer', 0):.4f}",
                '平均遗忘': f"{cl.get('average_forgetting', 0):.4f}",
            })

        df = pd.DataFrame(data)
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 关键发现\n\n")

        # 找出最佳配置
        best_avg_reward = max(experiments.items(), key=lambda x: x[1]['metrics']['basic'].get('average_reward', 0))
        best_fwt = max(experiments.items(), key=lambda x: x[1]['metrics']['continual_learning'].get('forward_transfer', 0))
        best_bwt = max(experiments.items(), key=lambda x: x[1]['metrics']['continual_learning'].get('backward_transfer', 0))

        f.write(f"### 最佳平均奖励\n")
        f.write(f"- 实验: **{best_avg_reward[0]}**\n")
        f.write(f"- 平均奖励: {best_avg_reward[1]['metrics']['basic']['average_reward']:.4f}\n\n")

        f.write(f"### 最佳前向迁移\n")
        f.write(f"- 实验: **{best_fwt[0]}**\n")
        f.write(f"- 前向迁移: {best_fwt[1]['metrics']['continual_learning']['forward_transfer']:.4f}\n\n")

        f.write(f"### 最佳后向迁移\n")
        f.write(f"- 实验: **{best_bwt[0]}**\n")
        f.write(f"- 后向迁移: {best_bwt[1]['metrics']['continual_learning']['backward_transfer']:.4f}\n\n")

        f.write("## 可视化结果\n\n")
        f.write("- [学习曲线](learning_curves.png)\n")
        f.write("- [迁移指标对比](transfer_metrics.png)\n")
        f.write("- [各阶段性能](stage_performance.png)\n")
        f.write("- [保留任务性能](retention_performance.png)\n\n")

    print(f"\n📄 分析报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="分析三域持续学习实验结果")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./three_domain_results",
        help="结果目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认与结果目录相同）",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return 1

    print("=" * 80)
    print("三域持续学习结果分析")
    print("=" * 80)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print()

    # 加载结果
    print("📥 加载实验结果...")
    experiments = load_results(results_dir)

    if not experiments:
        print("❌ 未找到任何实验结果")
        return 1

    print(f"✅ 找到 {len(experiments)} 个实验")
    print()

    # 打印汇总表格
    df = print_summary_table(experiments)

    # 生成可视化
    print("\n📊 生成可视化...")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(experiments, output_dir)
    plot_transfer_metrics(experiments, output_dir)
    plot_stage_performance(experiments, output_dir)
    plot_retention_performance(experiments, output_dir)

    # 生成报告
    print("\n📄 生成分析报告...")
    generate_report(experiments, output_dir)

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print(f"\n查看结果:")
    print(f"  报告: {output_dir / 'analysis_report.md'}")
    print(f"  图表: {output_dir}/*.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
