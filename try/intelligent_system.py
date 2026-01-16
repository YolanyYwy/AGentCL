#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能任务生成和划分系统

功能:
1. 为每个域生成大量纯任务（每个类别至少100个）
2. 智能分析 Telecom 域，找到最适合持续学习的划分方式
3. 确保所有类别数量平衡且任务纯度高
"""

import json
import random
import copy
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

random.seed(42)

class IntelligentTaskGenerator:
    """智能任务生成器"""

    def __init__(self, domain: str, tasks: List[Dict], db: Dict = None):
        self.domain = domain
        self.original_tasks = tasks
        self.db = db or {}
        self.tool_templates = self._extract_tool_templates()

    def _extract_tool_templates(self) -> Dict[str, List[Dict]]:
        """提取每个工具的任务模板"""
        templates = defaultdict(list)

        for task in self.original_tasks:
            actions = task.get('evaluation_criteria', {}).get('actions', [])

            # 只保留单工具任务作为模板
            if len(actions) == 1:
                tool = actions[0].get('name')
                if tool:
                    templates[tool].append(task)

        return dict(templates)

    def generate_pure_tasks(self, tool: str, target_count: int) -> List[Dict]:
        """为指定工具生成纯任务"""
        if tool not in self.tool_templates:
            print(f"  警告: 工具 {tool} 没有单工具模板")
            return []

        templates = self.tool_templates[tool]
        if not templates:
            return []

        generated = []
        task_id_start = len(self.original_tasks)

        for i in range(target_count):
            # 随机选择一个模板
            template = random.choice(templates)

            # 深拷贝
            new_task = copy.deepcopy(template)

            # 修改 ID
            new_task['id'] = str(task_id_start + len(generated))

            # 标记为生成任务
            if 'description' in new_task and new_task['description']:
                if 'purpose' in new_task['description']:
                    new_task['description']['purpose'] = f"[GENERATED-PURE-{tool}] {new_task['description'].get('purpose', '')}"

            # 变化参数
            actions = new_task.get('evaluation_criteria', {}).get('actions', [])
            for action in actions:
                self._vary_arguments(action.get('arguments', {}), i)

            generated.append(new_task)

        return generated

    def _vary_arguments(self, arguments: Dict, seed: int):
        """变化参数值"""
        for key, value in arguments.items():
            if isinstance(value, str):
                if 'id' in key.lower() or 'number' in key.lower():
                    # ID 类参数添加变化
                    arguments[key] = f"{value}_gen{seed}_{random.randint(1000, 9999)}"
                elif key in ['zip', 'zipcode']:
                    # 邮编随机生成
                    arguments[key] = f"{random.randint(10000, 99999)}"
            elif isinstance(value, (int, float)):
                # 数值类参数添加随机变化
                arguments[key] = value + random.randint(-10, 10)

class IntelligentPartitioner:
    """智能任务划分器"""

    def __init__(self, domain: str, tasks: List[Dict]):
        self.domain = domain
        self.tasks = tasks

    def analyze_and_partition(self) -> Dict[str, Any]:
        """智能分析并划分任务"""
        print(f"\n{'='*80}")
        print(f"智能分析 {self.domain.upper()} 域")
        print(f"{'='*80}")

        if self.domain == 'telecom':
            return self._partition_telecom()
        else:
            return self._partition_standard()

    def _partition_telecom(self) -> Dict[str, Any]:
        """Telecom 域的智能划分"""
        print("\nTelecom 域特殊处理...")

        # 分析工具的语义相似性
        tool_groups = self._group_telecom_tools()

        print(f"\n识别出 {len(tool_groups)} 个工具组:")
        for group_name, tools in tool_groups.items():
            print(f"  {group_name}: {len(tools)} 个工具")
            for tool in tools[:5]:
                print(f"    - {tool}")
            if len(tools) > 5:
                print(f"    ... 还有 {len(tools)-5} 个")

        # 按工具组划分任务
        sequences = []
        seq_id = 0

        for group_name, tools in sorted(tool_groups.items()):
            # 收集该组的所有任务
            group_tasks = []

            for task in self.tasks:
                actions = task.get('evaluation_criteria', {}).get('actions', [])

                # 只看第一个工具
                if actions:
                    first_tool = actions[0].get('name')
                    if first_tool in tools:
                        group_tasks.append(task['id'])

            if not group_tasks:
                continue

            # 70/15/15 划分
            n_tasks = len(group_tasks)
            n_train = int(n_tasks * 0.7)
            n_val = int(n_tasks * 0.15)

            sequences.append({
                'sequence_id': seq_id,
                'sequence_name': group_name,
                'function_category': group_name,
                'metadata': {
                    'group': group_name,
                    'tools': tools,
                    'total_tasks': n_tasks
                },
                'statistics': {
                    'total_tasks': n_tasks,
                    'train_tasks': n_train,
                    'val_tasks': n_val,
                    'test_tasks': n_tasks - n_train - n_val
                },
                'task_ids': {
                    'all': group_tasks,
                    'train': group_tasks[:n_train],
                    'val': group_tasks[n_train:n_train + n_val],
                    'test': group_tasks[n_train + n_val:]
                }
            })
            seq_id += 1

        category_stats = {seq['function_category']: seq['statistics']['total_tasks']
                         for seq in sequences}

        return {
            'domain': self.domain,
            'strategy': 'intelligent_telecom',
            'num_sequences': len(sequences),
            'total_tasks': len(self.tasks),
            'tool_groups': tool_groups,
            'category_statistics': category_stats,
            'sequences': sequences
        }

    def _group_telecom_tools(self) -> Dict[str, List[str]]:
        """将 Telecom 工具按功能分组"""
        # 收集所有工具
        all_tools = set()
        for task in self.tasks:
            actions = task.get('evaluation_criteria', {}).get('actions', [])
            for action in actions:
                tool = action.get('name')
                if tool:
                    all_tools.add(tool)

        # 按语义分组
        groups = {
            'network_mode': [],      # 网络模式相关
            'data_control': [],      # 数据控制相关
            'roaming': [],           # 漫游相关
            'airplane_mode': [],     # 飞行模式相关
            'permission': [],        # 权限相关
            'apn_settings': [],      # APN 设置相关
            'device_control': [],    # 设备控制相关
            'wifi_calling': [],      # WiFi 通话相关
            'vpn': [],               # VPN 相关
            'sim_card': [],          # SIM 卡相关
            'payment': [],           # 支付相关
            'other': []              # 其他
        }

        for tool in all_tools:
            tool_lower = tool.lower()

            if 'network' in tool_lower and 'mode' in tool_lower:
                groups['network_mode'].append(tool)
            elif 'data' in tool_lower and ('toggle' in tool_lower or 'saver' in tool_lower or 'refuel' in tool_lower):
                groups['data_control'].append(tool)
            elif 'roaming' in tool_lower:
                groups['roaming'].append(tool)
            elif 'airplane' in tool_lower:
                groups['airplane_mode'].append(tool)
            elif 'permission' in tool_lower or 'grant' in tool_lower:
                groups['permission'].append(tool)
            elif 'apn' in tool_lower:
                groups['apn_settings'].append(tool)
            elif 'reboot' in tool_lower or 'device' in tool_lower:
                groups['device_control'].append(tool)
            elif 'wifi' in tool_lower and 'calling' in tool_lower:
                groups['wifi_calling'].append(tool)
            elif 'vpn' in tool_lower:
                groups['vpn'].append(tool)
            elif 'sim' in tool_lower:
                groups['sim_card'].append(tool)
            elif 'payment' in tool_lower or 'resume' in tool_lower:
                groups['payment'].append(tool)
            elif 'transfer' in tool_lower:
                groups['other'].append(tool)
            else:
                groups['other'].append(tool)

        # 移除空组
        return {k: v for k, v in groups.items() if v}

    def _partition_standard(self) -> Dict[str, Any]:
        """标准域的划分（Airline, Retail）"""
        # 按工具分类
        tool_categories = self._categorize_tools()

        # 按类别收集纯任务
        category_tasks = defaultdict(list)

        for task in self.tasks:
            actions = task.get('evaluation_criteria', {}).get('actions', [])

            if not actions:
                category_tasks['no_tool'].append(task['id'])
                continue

            # 检查是否为纯任务
            tools = [a.get('name') for a in actions if a.get('name')]
            categories = set(tool_categories.get(t, 'other') for t in tools)

            if len(categories) == 1:
                # 纯任务
                category = list(categories)[0]
                category_tasks[category].append(task['id'])

        print(f"\n纯任务统计:")
        for category in sorted(category_tasks.keys()):
            count = len(category_tasks[category])
            print(f"  {category}: {count} 个纯任务")

        # 创建序列
        sequences = []
        seq_id = 0

        for category in sorted(category_tasks.keys()):
            task_ids = category_tasks[category]

            if not task_ids:
                continue

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
                    'is_pure': True
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
            'strategy': 'intelligent_pure',
            'num_sequences': len(sequences),
            'total_tasks': len(self.tasks),
            'tool_categories': tool_categories,
            'category_statistics': category_stats,
            'sequences': sequences
        }

    def _categorize_tools(self) -> Dict[str, str]:
        """工具分类"""
        categories = {}

        all_tools = set()
        for task in self.tasks:
            actions = task.get('evaluation_criteria', {}).get('actions', [])
            for action in actions:
                tool = action.get('name')
                if tool:
                    all_tools.add(tool)

        query_kw = ['get', 'find', 'search', 'check', 'list', 'lookup']
        modify_kw = ['update', 'modify', 'change', 'edit', 'set']
        create_kw = ['create', 'book', 'make', 'place', 'register']
        delete_kw = ['cancel', 'delete', 'remove', 'drop']
        process_kw = ['process', 'handle', 'calculate', 'exchange', 'return']
        transfer_kw = ['transfer', 'send', 'forward']

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


def main():
    """主函数"""
    print("="*80)
    print("智能任务生成和划分系统")
    print("="*80)

    domains_dir = Path('data/tau2/domains')
    output_dir = Path('data/tau2/intelligent_sequences')
    output_dir.mkdir(exist_ok=True)

    # 目标：每个类别至少 100 个纯任务
    TARGET_PER_CATEGORY = 100

    for domain in ['airline', 'retail', 'telecom', 'mock']:
        print(f"\n{'='*80}")
        print(f"处理域: {domain.upper()}")
        print(f"{'='*80}")

        # 加载原始任务
        tasks_file = domains_dir / domain / 'tasks.json'
        with open(tasks_file, 'r', encoding='utf-8') as f:
            original_tasks = json.load(f)

        print(f"原始任务数: {len(original_tasks)}")

        # 加载数据库
        db_file = domains_dir / domain / 'db.json'
        db = None
        if db_file.exists():
            with open(db_file, 'r', encoding='utf-8') as f:
                db = json.load(f)

        # 第一步：生成大量纯任务
        if domain != 'telecom':  # Telecom 已经有足够多的任务
            print(f"\n生成纯任务...")
            generator = IntelligentTaskGenerator(domain, original_tasks, db)

            # 识别所有单工具任务的工具
            single_tool_tools = list(generator.tool_templates.keys())
            print(f"发现 {len(single_tool_tools)} 个有单工具模板的工具")

            # 为每个工具生成任务
            all_generated = []
            for tool in single_tool_tools:
                generated = generator.generate_pure_tasks(tool, TARGET_PER_CATEGORY)
                if generated:
                    print(f"  {tool}: 生成 {len(generated)} 个纯任务")
                    all_generated.extend(generated)

            # 合并任务
            all_tasks = original_tasks + all_generated
            print(f"\n总任务数: {len(all_tasks)} (原始 {len(original_tasks)} + 生成 {len(all_generated)})")

            # 保存生成的任务
            generated_file = domains_dir / domain / 'tasks_generated_pure.json'
            with open(generated_file, 'w', encoding='utf-8') as f:
                json.dump(all_tasks, f, indent=2, ensure_ascii=False)
            print(f"保存到: {generated_file}")
        else:
            all_tasks = original_tasks

        # 第二步：智能划分
        print(f"\n智能划分任务...")
        partitioner = IntelligentPartitioner(domain, all_tasks)
        result = partitioner.analyze_and_partition()

        # 保存结果
        output_file = output_dir / f'{domain}_intelligent.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n保存到: {output_file}")
        print(f"序列数: {result['num_sequences']}")
        print(f"类别分布:")
        for category, count in sorted(result['category_statistics'].items()):
            print(f"  {category}: {count} 任务")

    # 生成总结报告
    generate_summary_report(output_dir)

    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


def generate_summary_report(output_dir: Path):
    """生成总结报告"""
    report_file = output_dir / 'INTELLIGENT_REPORT.md'

    # 读取所有结果
    results = {}
    for domain in ['airline', 'retail', 'telecom', 'mock']:
        result_file = output_dir / f'{domain}_intelligent.json'
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                results[domain] = json.load(f)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 智能任务生成和划分报告\n\n")

        f.write("## 方案特点\n\n")
        f.write("1. **大量纯任务生成**: 每个类别至少 100 个纯任务\n")
        f.write("2. **Telecom 智能分组**: 按工具语义相似性分组\n")
        f.write("3. **数量平衡**: 各类别任务数量相对均衡\n")
        f.write("4. **高纯度**: 优先使用纯任务\n\n")

        f.write("---\n\n")

        for domain, result in sorted(results.items()):
            f.write(f"## {domain.upper()} 域\n\n")
            f.write(f"**总任务数**: {result['total_tasks']}\n")
            f.write(f"**序列数**: {result['num_sequences']}\n")
            f.write(f"**策略**: {result['strategy']}\n\n")

            f.write("### 类别分布\n\n")
            f.write("| 类别 | 任务数 | 占比 |\n")
            f.write("|------|--------|------|\n")

            total = result['total_tasks']
            for category, count in sorted(result['category_statistics'].items()):
                percentage = count / total * 100 if total > 0 else 0
                f.write(f"| {category} | {count} | {percentage:.1f}% |\n")

            f.write("\n---\n\n")

    print(f"\n报告保存到: {report_file}")


if __name__ == "__main__":
    main()
