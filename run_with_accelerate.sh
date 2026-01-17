#!/bin/bash
# 使用 Accelerate 启动多 GPU 训练

# 配置
MODEL="Qwen/Qwen3-4B"
TASKS_PER_DOMAIN=10
OUTPUT_DIR="./results_accelerate"

echo "=========================================="
echo "使用 Accelerate 进行多 GPU 训练"
echo "=========================================="
echo "模型: $MODEL"
echo "每域任务数: $TASKS_PER_DOMAIN"
echo ""

# 方法 1: 使用 accelerate launch (推荐)
# 自动检测所有可用 GPU 并使用
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    run_three_domain_continual_learning.py \
    --model $MODEL \
    --device cuda \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir $OUTPUT_DIR \
    --use-grpo

# 方法 2: 使用配置文件
# accelerate launch --config_file accelerate_config.yaml run_three_domain_continual_learning.py ...

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
