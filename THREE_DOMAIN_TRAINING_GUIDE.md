# 三域持续学习训练指南

## 📋 概述

本指南介绍如何在 Airline、Retail 和 Telecom 三个域上进行持续学习训练，并计算前向迁移（Forward Transfer）和后向迁移（Backward Transfer）指标。

## 🎯 训练流程

训练按照以下顺序进行：
1. **Stage 1: Airline Domain** - 学习航空领域任务
2. **Stage 2: Retail Domain** - 学习零售领域任务 + 保留 Airline 任务测试
3. **Stage 3: Telecom Domain** - 学习电信领域任务 + 保留 Airline + Retail 任务测试

## 📊 评估指标

### 前向迁移（Forward Transfer, FWT）
- **定义**: 在新任务上的性能 - 基线性能
- **含义**: 正值表示之前的学习有助于新任务，负值表示负迁移
- **计算**: `FWT = Acc(new_tasks | current_stage) - Acc(new_tasks | baseline)`

### 后向迁移（Backward Transfer, BWT）
- **定义**: 在旧任务上的最终性能 - 学习时的性能
- **含义**: 正值表示改进，负值表示遗忘
- **计算**: `BWT = Acc(old_tasks | final_stage) - Acc(old_tasks | learned_stage)`

### 其他指标
- **平均遗忘（Average Forgetting）**: 工具级别的遗忘程度
- **学习效率（Learning Efficiency）**: 单位样本的学习效果
- **AULC（Area Under Learning Curve）**: 学习曲线下面积

## 🚀 快速开始

### 1. 单 GPU 训练

```bash
# 基础训练
python run_three_domain_continual_learning.py \
    --model Qwen/Qwen3-4B \
    --device cuda \
    --tasks-per-domain 100 \
    --output-dir ./results

# 不使用 GRPO（仅评估）
python run_three_domain_continual_learning.py \
    --model Qwen/Qwen3-4B \
    --device cuda \
    --no-grpo \
    --tasks-per-domain 100 \
    --output-dir ./results_no_grpo
```

### 2. 多 GPU 并行训练（推荐）

```bash
# 在 8 张 GPU 上并行运行不同配置
chmod +x run_three_domain_multi_gpu.sh
./run_three_domain_multi_gpu.sh
```

这会自动在 8 张 GPU 上运行以下配置：
- GPU 0: Baseline (lr=1e-6, beta=0.1, group=4)
- GPU 1: 更小学习率 (lr=5e-7)
- GPU 2: 更大学习率 (lr=2e-6)
- GPU 3: 更小 KL 惩罚 (beta=0.05)
- GPU 4: 更大 KL 惩罚 (beta=0.2)
- GPU 5: 更大 group size (group=8)
- GPU 6: 更小 group size (group=2)
- GPU 7: 不使用 GRPO (仅评估)

### 3. 监控训练进度

```bash
# 查看所有日志
tail -f logs/gpu*.log

# 查看特定 GPU
tail -f logs/gpu0_baseline.log

# 监控 GPU 使用
watch -n 1 nvidia-smi

# 使用监控脚本
chmod +x monitor_training.sh
watch -n 5 ./monitor_training.sh
```

### 4. 分析结果

```bash
# 分析所有实验结果
python analyze_three_domain_results.py \
    --results-dir ./three_domain_results

# 查看报告
cat ./three_domain_results/analysis_report.md

# 查看可视化
ls ./three_domain_results/*.png
```

## 📁 输出文件结构

```
three_domain_results/
├── gpu0_baseline/
│   ├── curriculum.json          # 课程配置
│   ├── results.json              # 详细结果
│   ├── metrics.json              # 评估指标
│   └── grpo_checkpoints/         # 模型检查点
│       ├── stage_stage_1_airline/
│       ├── stage_stage_2_retail/
│       └── stage_stage_3_telecom/
├── gpu1_lr5e7/
│   └── ...
├── learning_curves.png           # 学习曲线
├── transfer_metrics.png          # 迁移指标对比
├── stage_performance.png         # 各阶段性能
├── retention_performance.png     # 保留任务性能
└── analysis_report.md            # 分析报告
```

## 🔧 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-4B` | 模型名称 |
| `--device` | `cuda` | 设备（cuda/cpu/auto） |
| `--tasks-per-domain` | `100` | 每个域的任务数量 |
| `--learning-rate` | `1e-6` | 学习率 |
| `--beta` | `0.1` | KL 惩罚系数 |
| `--group-size` | `4` | GRPO group size |
| `--use-grpo` | `True` | 使用 GRPO 训练 |
| `--no-grpo` | `False` | 不使用 GRPO（仅评估） |

### 任务文件路径

| 参数 | 默认值 |
|------|--------|
| `--airline-tasks` | `data/tau2/domains/airline/tasks.json` |
| `--retail-tasks` | `data/tau2/domains/retail/tasks.json` |
| `--telecom-tasks` | `data/tau2/domains/telecom/tasks_hard_300.json` |

## 📈 结果解读

### 理想的持续学习系统应该：

✅ **高前向迁移（FWT > 0）**
- 表示之前的学习有助于新任务
- 说明模型能够迁移知识

✅ **高后向迁移（BWT > 0）**
- 表示在新任务上学习后，旧任务性能提升
- 说明模型没有遗忘，甚至有改进

✅ **低平均遗忘（Forgetting < 0.1）**
- 表示模型保持了对旧任务的记忆
- 说明持续学习策略有效

✅ **高学习效率（Efficiency > 0.01）**
- 表示模型能够快速学习
- 说明训练效率高

### 示例结果解读

```
平均奖励: 0.7500
前向迁移: 0.1200  ✅ 正值，说明有正向迁移
后向迁移: 0.0500  ✅ 正值，说明没有遗忘
平均遗忘: 0.0300  ✅ 低值，说明遗忘少
学习效率: 0.0125  ✅ 较高，说明学习快
```

## 🛠️ 故障排除

### 问题1: 模型加载失败
```
OSError: Qwen/Qwen3-4B is not a local folder...
```
**解决**: 确认模型名称正确，或使用本地路径

### 问题2: GPU 内存不足
```
CUDA out of memory
```
**解决**:
- 减少 `--tasks-per-domain`
- 减少 `--group-size`
- 使用 4-bit 量化（需要修改代码添加 `load_in_4bit=True`）

### 问题3: 任务文件不存在
```
FileNotFoundError: data/tau2/domains/telecom/tasks_hard_300.json
```
**解决**: 确保已经运行了任务筛选脚本生成 `tasks_hard_300.json`

## 📚 相关文档

- [GRPO 训练原理](CONTINUAL_LEARNING_BENCHMARK.md)
- [持续学习评估指标](src/tau2/continual/evaluation/metrics.py)
- [课程设计指南](src/tau2/continual/curriculum/README.md)

## 💡 最佳实践

1. **先运行小规模实验**
   - 使用 `--tasks-per-domain 10` 快速验证
   - 确认流程正常后再增加任务数

2. **使用多 GPU 并行**
   - 同时测试多个超参数配置
   - 节省时间，找到最佳配置

3. **定期保存检查点**
   - GRPO 训练器会自动保存
   - 可以从检查点恢复训练

4. **分析结果对比**
   - 使用 `analyze_three_domain_results.py` 对比所有实验
   - 关注前向迁移和后向迁移指标

## 🎓 引用

如果使用本代码进行研究，请引用：

```bibtex
@article{tau2bench2024,
  title={τ²-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
  author={...},
  journal={...},
  year={2024}
}
```

## 📧 联系方式

如有问题，请提交 Issue 或联系维护者。
