#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的功能化任务划分策略

解决的问题:
1. 任务纯度：每个类别的任务应该只包含该类别的工具
2. 数量平衡：各类别的任务数量应该相对均衡
3. 类别独立性：不同类别之间的任务差异应该明显
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

random.seed(42)

class ImprovedFunctionalPartitioner:
    """改进的功能化任务划分器"""

    def __init__(self, domain: str, tasks: List[Dict]):
        self.domain = domain
        self.tasks = tasks
        self.tool_categories = self._categorize_tools()

    def _categorize_tools(self) -> Dict[str, str]:
        """工具分类"""
        categories = {}

        # 定义关键词
        query_kw = ['get', 'find', 'search', 'check', 'list', 'lookup', 'view', 'show']
        modify_kw = ['update', 'modify', 'change', 'edit', 'set', 'add', 'remove']
        create_kw = ['create', 'book', 'make', 'place', 'register', 'new']
        delete_kw = ['cancel', 'delete', 'drop']
        process_kw = ['process', 'handle', 'execute', 'calculate']
        transfer_kw = ['transfer', 'send', 'forward', 'escalate']

        # 收集所有工具
        all_tools = set()
        for task in self.tasks:
            actions = task.get('evaluation_criteria', {}).get('actions', [])
            for action in actions:
                tool = action.get('name')
                if tool:
                    all_tools.add(tool)

        # 分类
        for tool in all_tools:
            tool_lower = tool.lower()

            if any(kw in tool_lower for kw in query_kw):
                categories[tool] = 'query'
            elif any(kw in tool_lower for kw in delete_kw):
                categories[tool] = 'delete'
            elif any(kw in tool_lower for kw in create_kw):
                categories[tool] = 'create'
            elif any(kw in tool_lower for kw in modify_kw):
                categories[tool] = 'modify'
            elif any(kw in tool_lower for kw in process_kw):
                categories[tool] = 'process'
            elif any(kw in tool_lower for kw in transfer_kw):
                categories[tool] = 'transfer'
            else:
                categories[tool] = 'other'

        return categories

    def partition_pure_tasks(self, target_tasks_per_category: int = 100) -> Dict[str, Any]:
        """
        划分纯任务（每个任务只包含一个类别的工具）

        Args:
            target_tasks_per_category: 每个类别的目标任务数
        """
        print(f"\n{'='*80}")
        print(f"Partitioning {self.domain.upper()} domain with pure tasks")
        print(f"{'='*80}")

        # 第一步：分析每个任务的工具类别
        task_analysis = []
        for task in self.tasks:
            task_id = task['id']
            actions = task.get('evaluation_criteria', {}).get('actions', [])

            if not actions:
                task_analysis.append({
                    'task_id': task_id,
                    'categories': set(),
                    'tools': [],
                    'is_pure': True,
                    'primary_category': 'no_tool'
                })
                continue

            # 获取所有工具的类别
            tools = [a.get('name') for a in actions if a.get('name')]
            categories = set(self.tool_categories.get(t, 'other') for t in tools)

            # 判断是否为纯任务（只包含一个类别的工具）
            is_pure = len(categories) == 1
            primary_category = list(categories)[0] if categories else 'other'

            task_analysis.append({
                'task_id': task_id,
                'categories': categories,
                'tools': tools,
                'is_pure': is_pure,
                'primary_category': primary_category
            })

        # 第二步：统计纯任务
        pure_tasks_by_category = defaultdict(list)
        mixed_tasks_by_category = defaultdict(list)

        for analysis in task_analysis:
            if analysis['is_pure']:
                pure_tasks_by_category[analysis['primary_category']].append(analysis['task_id'])
            else:
                # 混合任务按主要类别（第一个工具的类别）分类
                mixed_tasks_by_category[analysis['primary_category']].append(analysis['task_id'])

        print(f"\n纯任务统计:")
        for category in sorted(pure_tasks_by_category.keys()):
            count = len(pure_tasks_by_category[category])
            print(f"  {category:12s}: {count:4d} 纯任务")

        print(f"\n混合任务统计:")
        for category in sorted(mixed_tasks_by_category.keys()):
            count = len(mixed_tasks_by_category[category])
            print(f"  {category:12s}: {count:4d} 混合任务")

        # 第三步：平衡各类别的任务数
        balanced_sequences = self._balance_categories(
            pure_tasks_by_category,
            mixed_tasks_by_category,
            target_tasks_per_category
        )

        # 第四步：创建序列
        sequences = []
        seq_id = 0

        for category in sorted(balanced_sequences.keys()):
            task_ids = balanced_sequences[category]

            if not task_ids:
                continue

            # 70/15/15 划分
            n_tasks = len(task_ids)
            n_train = int(n_tasks * 0.7)
            n_val = int(n_tasks * 0.15)

            sequences.append({
                'sequence_id': seq_id,
                'sequence_name': category,
                'function_category': category,
                'metadata': {
                    'category': category,
                    'total_tasks': n_tasks,
                    'is_balanced': True
                },
                'statistics': {
                    'total_tasks': n_tasks,
                    'train_tasks': n_train,
                    'val_tasks': n_val,
                    'test_tasks': n_tasks - n_train - n_val
                },
                'task_ids': {
                    'all': task_ids,
                    'train': task_ids[:n_train],
                    'val': task_ids[n_train:n_train + n_val],
                    'test': task_ids[n_train + n_val:]
                }
            })
            seq_id += 1

        # 统计
        category_stats = {seq['function_category']: seq['statistics']['total_tasks']
                         for seq in sequences}

        print(f"\n最终平衡后的类别分布:")
        for category, count in sorted(category_stats.items()):
            print(f"  {category:12s}: {count:4d} 任务")

        return {
            'domain': self.domain,
            'strategy': 'functional_balanced',
            'num_sequences': len(sequences),
            'total_tasks': sum(category_stats.values()),
            'tool_categories': self.tool_categories,
            'category_statistics': category_stats,
            'sequences': sequences,
            'task_analysis': {
                'pure_tasks': {k: len(v) for k, v in pure_tasks_by_category.items()},
                'mixed_tasks': {k: len(v) for k, v in mixed_tasks_by_category.items()}
            }
        }

    def _balance_categories(self, pure_tasks: Dict, mixed_tasks: Dict,
                           target: int) -> Dict[str, List[str]]:
        """平衡各类别的任务数"""
        balanced = {}

        # 主要类别（排除 no_tool 和 transfer）
        main_categories = ['query', 'create', 'modify', 'delete', 'process', 'other']

        for category in main_categories:
            pure = pure_tasks.get(category, [])
            mixed = mixed_tasks.get(category, [])

            # 优先使用纯任务
            if len(pure) >= target:
                # 如果纯任务足够，随机采样
                balanced[category] = random.sample(pure, target)
            elif len(pure) + len(mixed) >= target:
                # 纯任务不够，补充混合任务
                balanced[category] = pure + random.sample(mixed, target - len(pure))
            else:
                # 总数不够，全部使用
                balanced[category] = pure + mixed

        # 特殊类别（保留所有任务）
        for category in ['no_tool', 'transfer']:
            pure = pure_tasks.get(category, [])
            mixed = mixed_tasks.get(category, [])
            if pure or mixed:
                balanced[category] = pure + mixed

        return balanced

    def partition_by_primary_tool(self, min_tasks_per_category: int = 50) -> Dict[str, Any]:
        """
        按主要工具划分（只看第一个工具）

        Args:
            min_tasks_per_category: 每个类别的最小任务数
        """
        print(f"\n{'='*80}")
        print(f"Partitioning {self.domain.upper()} by primary tool")
        print(f"{'='*80}")

        # 按第一个工具的类别分组
        category_tasks = defaultdict(list)

        for task in self.tasks:
            task_id = task['id']
            actions = task.get('evaluation_criteria', {}).get('actions', [])

            if not actions:
                category_tasks['no_tool'].append(task_id)
                continue

            # 只看第一个工具
            first_tool = actions[0].get('name')
            if first_tool:
                category = self.tool_categories.get(first_tool, 'other')
                category_tasks[category].append(task_id)

        # 过滤掉任务数太少的类别
        filtered_categories = {}
        for category, task_ids in category_tasks.items():
            if len(task_ids) >= min_tasks_per_category or category in ['no_tool', 'transfer']:
                filtered_categories[category] = task_ids

        print(f"\n类别分布（按第一个工具）:")
        for category in sorted(filtered_categories.keys()):
            count = len(filtered_categories[category])
            print(f"  {category:12s}: {count:4d} 任务")

        # 创建序列
        sequences = []
        seq_id = 0

        for category in sorted(filtered_categories.keys()):
            task_ids = filtered_categories[category]

            n_tasks = len(task_ids)
            n_train = int(n_tasks * 0.7)
            n_val = int(n_tasks * 0.15)

            sequences.append({
                'sequence_id': seq_id,
                'sequence_name': category,
                'function_category': category,
                'metadata': {
                    'category': category,
                    'total_tasks': n_tasks,
                    'method': 'primary_tool'
                },
                'statistics': {
                    'total_tasks': n_tasks,
                    'train_tasks': n_train,
                    'val_tasks': n_val,
                    'test_tasks': n_tasks - n_train - n_val
                },
                'task_ids': {
                    'all': task_ids,
                    'train': task_ids[:n_train],
                    'val': task_ids[n_train:n_train + n_val],
                    'test': task_ids[n_train + n_val:]
                }
            })
            seq_id += 1

        category_stats = {seq['function_category']: seq['statistics']['total_tasks']
                         for seq in sequences}

        return {
            'domain': self.domain,
            'strategy': 'functional_primary_tool',
            'num_sequences': len(sequences),
            'total_tasks': sum(category_stats.values()),
            'tool_categories': self.tool_categories,
            'category_statistics': category_stats,
            'sequences': sequences
        }


def main():
    """主函数"""
    print("="*80)
    print("改进的功能化任务划分")
    print("="*80)

    domains_dir = Path('data/tau2/domains')
    output_dir = Path('data/tau2/functional_sequences_v2')
    output_dir.mkdir(exist_ok=True)

    # 目标任务数（每个类别）
    TARGET_PER_CATEGORY = {
        'airline': 60,   # 300 / 5 = 60
        'retail': 60,    # 300 / 5 = 60
        'telecom': 400,  # 2285 / ~5 = 400+
        'mock': 2        # 太少，保持原样
    }

    all_results = {}

    for domain in ['airline', 'retail', 'telecom', 'mock']:
        print(f"\n{'='*80}")
        print(f"Processing domain: {domain.upper()}")
        print(f"{'='*80}")

        # 加载任务（优先使用增强数据）
        augmented_file = domains_dir / domain / 'tasks_augmented.json'
        tasks_file = domains_dir / domain / 'tasks.json'

        if augmented_file.exists():
            with open(augmented_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            print(f"Loaded augmented tasks: {len(tasks)}")
        else:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            print(f"Loaded original tasks: {len(tasks)}")

        # 创建划分器
        partitioner = ImprovedFunctionalPartitioner(domain, tasks)

        # 方法1: 平衡的纯任务划分
        target = TARGET_PER_CATEGORY[domain]
        result_balanced = partitioner.partition_pure_tasks(target)

        # 保存结果
        output_file = output_dir / f'{domain}_functional_balanced.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_balanced, f, indent=2, ensure_ascii=False)

        print(f"\n保存到: {output_file}")

        # 方法2: 按主要工具划分（作为对比）
        result_primary = partitioner.partition_by_primary_tool(min_tasks_per_category=30)

        output_file2 = output_dir / f'{domain}_functional_primary.json'
        with open(output_file2, 'w', encoding='utf-8') as f:
            json.dump(result_primary, f, indent=2, ensure_ascii=False)

        print(f"保存到: {output_file2}")

        all_results[domain] = {
            'balanced': result_balanced,
            'primary': result_primary
        }

    # 生成总结报告
    generate_comparison_report(all_results, output_dir)

    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


def generate_comparison_report(results: Dict, output_dir: Path):
    """生成对比报告"""
    report_file = output_dir / 'COMPARISON_REPORT.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 功能化任务划分对比报告\n\n")

        f.write("## 方法对比\n\n")
        f.write("### 方法 1: 平衡的纯任务划分 (Balanced Pure Tasks)\n\n")
        f.write("**特点**:\n")
        f.write("- 优先使用纯任务（只包含一个类别的工具）\n")
        f.write("- 各类别任务数量相对平衡\n")
        f.write("- 类别间独立性强\n\n")

        f.write("### 方法 2: 按主要工具划分 (Primary Tool)\n\n")
        f.write("**特点**:\n")
        f.write("- 只看第一个工具的类别\n")
        f.write("- 保留所有任务\n")
        f.write("- 可能包含混合任务\n\n")

        f.write("---\n\n")

        for domain, domain_results in sorted(results.items()):
            f.write(f"## {domain.upper()} 域\n\n")

            # 方法1统计
            balanced = domain_results['balanced']
            f.write("### 方法 1: 平衡的纯任务划分\n\n")
            f.write(f"**总任务数**: {balanced['total_tasks']}\n")
            f.write(f"**序列数**: {balanced['num_sequences']}\n\n")

            f.write("| 类别 | 任务数 | 纯任务数 | 混合任务数 |\n")
            f.write("|------|--------|----------|------------|\n")

            pure_tasks = balanced['task_analysis']['pure_tasks']
            mixed_tasks = balanced['task_analysis']['mixed_tasks']

            for category, count in sorted(balanced['category_statistics'].items()):
                pure_count = pure_tasks.get(category, 0)
                mixed_count = mixed_tasks.get(category, 0)
                f.write(f"| {category} | {count} | {pure_count} | {mixed_count} |\n")

            # 方法2统计
            primary = domain_results['primary']
            f.write(f"\n### 方法 2: 按主要工具划分\n\n")
            f.write(f"**总任务数**: {primary['total_tasks']}\n")
            f.write(f"**序列数**: {primary['num_sequences']}\n\n")

            f.write("| 类别 | 任务数 |\n")
            f.write("|------|--------|\n")

            for category, count in sorted(primary['category_statistics'].items()):
                f.write(f"| {category} | {count} |\n")

            f.write("\n---\n\n")

    print(f"\n对比报告保存到: {report_file}")


if __name__ == "__main__":
    main()
