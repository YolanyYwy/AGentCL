#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能化持续学习训练示例

展示如何使用功能化任务序列进行持续学习训练
"""

import json
from pathlib import Path
from typing import Dict, List

class FunctionalContinualLearner:
    """基于功能类别的持续学习训练器"""

    def __init__(self, domain: str):
        self.domain = domain
        self.tasks = self._load_tasks()
        self.sequences = self._load_sequences()
        self.task_map = {task['id']: task for task in self.tasks}

    def _load_tasks(self) -> List[Dict]:
        """加载任务数据"""
        # 优先加载增强数据
        augmented_file = Path(f'data/tau2/domains/{self.domain}/tasks_augmented.json')
        if augmented_file.exists():
            with open(augmented_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 否则加载原始数据
        tasks_file = Path(f'data/tau2/domains/{self.domain}/tasks.json')
        with open(tasks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_sequences(self) -> Dict:
        """加载功能序列"""
        seq_file = Path(f'data/tau2/functional_sequences/{self.domain}_functional_sequences.json')
        with open(seq_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def train_by_function_category(self, category_order: List[str] = None):
        """按功能类别顺序训练"""
        if category_order is None:
            # 默认顺序：查询 -> 创建 -> 修改 -> 删除
            category_order = ['query', 'create', 'modify', 'delete', 'process', 'transfer', 'other', 'no_tool']

        print(f"\n{'='*80}")
        print(f"Training on {self.domain.upper()} domain by function category")
        print(f"{'='*80}")

        # 按类别组织序列
        category_sequences = {}
        for sequence in self.sequences['sequences']:
            category = sequence['function_category']
            if category not in category_sequences:
                category_sequences[category] = []
            category_sequences[category].append(sequence)

        # 按指定顺序训练
        for category in category_order:
            if category not in category_sequences:
                continue

            print(f"\n{'-'*80}")
            print(f"Category: {category.upper()}")
            print(f"{'-'*80}")

            sequences = category_sequences[category]
            print(f"Number of sequences in this category: {len(sequences)}")

            for seq_idx, sequence in enumerate(sequences):
                self._train_on_sequence(sequence, seq_idx)

    def train_by_tool(self):
        """按工具逐个训练"""
        print(f"\n{'='*80}")
        print(f"Training on {self.domain.upper()} domain by tool")
        print(f"{'='*80}")

        # 按任务数排序（从多到少）
        sequences_sorted = sorted(
            self.sequences['sequences'],
            key=lambda s: s['statistics']['total_tasks'],
            reverse=True
        )

        for seq_idx, sequence in enumerate(sequences_sorted):
            self._train_on_sequence(sequence, seq_idx)

    def _train_on_sequence(self, sequence: Dict, seq_idx: int):
        """在单个序列上训练"""
        tool_name = sequence['sequence_name']
        category = sequence['function_category']
        stats = sequence['statistics']

        print(f"\n  Sequence {seq_idx}: {tool_name} ({category})")
        print(f"    Total tasks: {stats['total_tasks']}")
        print(f"    Train: {stats['train_tasks']}, Val: {stats['val_tasks']}, Test: {stats['test_tasks']}")

        # 训练阶段
        train_ids = sequence['task_ids']['train']
        print(f"    Training on {len(train_ids)} tasks...")
        for task_id in train_ids:
            if task_id in self.task_map:
                task = self.task_map[task_id]
                # 这里添加你的训练代码
                # model.train(task)
                pass

        # 验证阶段
        val_ids = sequence['task_ids']['val']
        if val_ids:
            print(f"    Validating on {len(val_ids)} tasks...")
            for task_id in val_ids:
                if task_id in self.task_map:
                    task = self.task_map[task_id]
                    # 这里添加你的验证代码
                    # model.validate(task)
                    pass

        # 测试阶段
        test_ids = sequence['task_ids']['test']
        if test_ids:
            print(f"    Testing on {len(test_ids)} tasks...")
            for task_id in test_ids:
                if task_id in self.task_map:
                    task = self.task_map[task_id]
                    # 这里添加你的测试代码
                    # model.test(task)
                    pass

    def evaluate_transfer(self):
        """评估前向和后向迁移"""
        print(f"\n{'='*80}")
        print(f"Evaluating transfer effects on {self.domain.upper()} domain")
        print(f"{'='*80}")

        sequences = self.sequences['sequences']
        performance = {}

        # 逐序列训练并评估
        for train_idx, train_seq in enumerate(sequences):
            print(f"\n{'-'*80}")
            print(f"After training sequence {train_idx}: {train_seq['sequence_name']}")
            print(f"{'-'*80}")

            # 训练当前序列
            train_ids = train_seq['task_ids']['train']
            # model.train(train_ids)

            # 在所有序列上评估
            for eval_idx, eval_seq in enumerate(sequences[:train_idx + 1]):
                test_ids = eval_seq['task_ids']['test']

                # 计算准确率（这里是模拟）
                # accuracy = model.evaluate(test_ids)
                accuracy = 0.85  # 模拟值

                performance[(train_idx, eval_idx)] = accuracy
                print(f"  Sequence {eval_idx} ({eval_seq['sequence_name']}): {accuracy:.2%}")

        # 计算前向迁移
        forward_transfer = self._calculate_forward_transfer(performance, len(sequences))
        print(f"\nForward Transfer: {forward_transfer:.2%}")

        # 计算后向迁移
        backward_transfer = self._calculate_backward_transfer(performance, len(sequences))
        print(f"Backward Transfer: {backward_transfer:.2%}")
        if backward_transfer < 0:
            print("  (Negative value indicates catastrophic forgetting)")

    def _calculate_forward_transfer(self, performance: Dict, num_sequences: int) -> float:
        """计算前向迁移"""
        total = 0
        count = 0

        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                if (i, j) in performance:
                    total += performance[(i, j)]
                    count += 1

        return total / count if count > 0 else 0

    def _calculate_backward_transfer(self, performance: Dict, num_sequences: int) -> float:
        """计算后向迁移"""
        total = 0
        count = 0

        for i in range(num_sequences):
            for j in range(i):
                if (i, j) in performance and (j, j) in performance:
                    total += performance[(i, j)] - performance[(j, j)]
                    count += 1

        return total / count if count > 0 else 0

    def print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"Statistics for {self.domain.upper()} domain")
        print(f"{'='*80}")

        print(f"\nTotal tasks: {len(self.tasks)}")
        print(f"Total sequences: {self.sequences['num_sequences']}")

        print(f"\nFunction category distribution:")
        for category, count in sorted(self.sequences['category_statistics'].items()):
            percentage = count / self.sequences['total_tasks'] * 100
            print(f"  {category:12s}: {count:4d} tasks ({percentage:5.1f}%)")

        print(f"\nTool categories:")
        tool_categories = self.sequences['tool_categories']
        category_count = {}
        for tool, category in tool_categories.items():
            category_count[category] = category_count.get(category, 0) + 1

        for category, count in sorted(category_count.items()):
            print(f"  {category:12s}: {count:2d} tools")


def example_1_train_by_category():
    """示例 1: 按功能类别训练"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Training by Function Category")
    print("="*80)

    learner = FunctionalContinualLearner('airline')
    learner.print_statistics()

    # 按功能类别顺序训练
    category_order = ['query', 'create', 'modify', 'delete']
    learner.train_by_function_category(category_order)


def example_2_train_by_tool():
    """示例 2: 按工具训练"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Training by Tool")
    print("="*80)

    learner = FunctionalContinualLearner('retail')
    learner.print_statistics()

    # 按工具训练（按任务数从多到少）
    learner.train_by_tool()


def example_3_evaluate_transfer():
    """示例 3: 评估迁移效果"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Evaluating Transfer Effects")
    print("="*80)

    learner = FunctionalContinualLearner('airline')
    learner.evaluate_transfer()


def example_4_cross_domain():
    """示例 4: 跨域迁移"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Cross-Domain Transfer")
    print("="*80)

    # 源域: Airline
    source_learner = FunctionalContinualLearner('airline')
    print("\nSource domain: AIRLINE")
    source_learner.print_statistics()

    # 在源域的 query 类别上预训练
    print("\nPre-training on source domain (query category)...")
    query_sequences = [
        seq for seq in source_learner.sequences['sequences']
        if seq['function_category'] == 'query'
    ]
    for seq in query_sequences:
        source_learner._train_on_sequence(seq, 0)

    # 目标域: Retail
    target_learner = FunctionalContinualLearner('retail')
    print("\n\nTarget domain: RETAIL")
    target_learner.print_statistics()

    # 在目标域的 query 类别上微调
    print("\nFine-tuning on target domain (query category)...")
    query_sequences = [
        seq for seq in target_learner.sequences['sequences']
        if seq['function_category'] == 'query'
    ]
    for seq in query_sequences:
        target_learner._train_on_sequence(seq, 0)


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Functional Continual Learning Examples")
    print("="*80)

    # 运行示例
    example_1_train_by_category()
    # example_2_train_by_tool()
    # example_3_evaluate_transfer()
    # example_4_cross_domain()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nTo run other examples, uncomment them in the main() function.")


if __name__ == "__main__":
    main()
