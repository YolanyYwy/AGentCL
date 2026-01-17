#!/bin/bash
# DDP 分布式训练启动脚本

MODEL="Qwen/Qwen3-4B"
NUM_GPUS=2  # 使用的 GPU 数量
TASKS_PER_DOMAIN=10

echo "=========================================="
echo "三域持续学习 - DDP 分布式数据并行"
echo "=========================================="
echo "模型: $MODEL"
echo "GPU 数量: $NUM_GPUS"
echo "每域任务数: $TASKS_PER_DOMAIN"
echo ""
echo "DDP 特性:"
echo "  ✓ 梯度 All-Reduce 同步"
echo "  ✓ 参数全局一致"
echo "  ✓ 真正的多卡联训"
echo ""

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO  # 调试 NCCL 通信

# 运行 DDP 训练
python run_three_domain_ddp.py \
    --model $MODEL \
    --num-gpus $NUM_GPUS \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --use-grpo \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 2 \
    --output-dir ./results_ddp_${NUM_GPUS}gpu

echo ""
echo "=========================================="
echo "DDP 训练完成！"
echo "=========================================="
