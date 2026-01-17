#!/bin/bash
# 在多 GPU 上训练 telecom 300 个困难任务
# 使用不同的超参数进行实验

MODEL="Qwen/Qwen3-4B"
TASKS_FILE="data/tau2/domains/telecom/tasks_hard_300.json"
OUTPUT_BASE="./grpo_results_telecom_300"

# 创建日志目录
mkdir -p logs
mkdir -p $OUTPUT_BASE

echo "开始在 8 张 GPU 上并行训练..."
echo "任务文件: $TASKS_FILE"
echo "模型: $MODEL"
echo ""

# GPU 0: 学习率 1e-6, beta 0.1 (baseline)
echo "GPU 0: lr=1e-6, beta=0.1 (baseline)"
CUDA_VISIBLE_DEVICES=0 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu0_lr1e6_beta0.1 \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 4 \
    > logs/gpu0_baseline.log 2>&1 &

# GPU 1: 学习率 5e-7, beta 0.1
echo "GPU 1: lr=5e-7, beta=0.1"
CUDA_VISIBLE_DEVICES=1 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu1_lr5e7_beta0.1 \
    --learning-rate 5e-7 \
    --beta 0.1 \
    --group-size 4 \
    > logs/gpu1_lr5e7.log 2>&1 &

# GPU 2: 学习率 2e-6, beta 0.1
echo "GPU 2: lr=2e-6, beta=0.1"
CUDA_VISIBLE_DEVICES=2 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu2_lr2e6_beta0.1 \
    --learning-rate 2e-6 \
    --beta 0.1 \
    --group-size 4 \
    > logs/gpu2_lr2e6.log 2>&1 &

# GPU 3: 学习率 1e-6, beta 0.05
echo "GPU 3: lr=1e-6, beta=0.05"
CUDA_VISIBLE_DEVICES=3 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu3_lr1e6_beta0.05 \
    --learning-rate 1e-6 \
    --beta 0.05 \
    --group-size 4 \
    > logs/gpu3_beta0.05.log 2>&1 &

# GPU 4: 学习率 1e-6, beta 0.2
echo "GPU 4: lr=1e-6, beta=0.2"
CUDA_VISIBLE_DEVICES=4 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu4_lr1e6_beta0.2 \
    --learning-rate 1e-6 \
    --beta 0.2 \
    --group-size 4 \
    > logs/gpu4_beta0.2.log 2>&1 &

# GPU 5: 学习率 1e-6, beta 0.1, group_size 8
echo "GPU 5: lr=1e-6, beta=0.1, group_size=8"
CUDA_VISIBLE_DEVICES=5 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu5_lr1e6_group8 \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 8 \
    > logs/gpu5_group8.log 2>&1 &

# GPU 6: 学习率 1e-6, beta 0.1, group_size 2
echo "GPU 6: lr=1e-6, beta=0.1, group_size=2"
CUDA_VISIBLE_DEVICES=6 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu6_lr1e6_group2 \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 2 \
    > logs/gpu6_group2.log 2>&1 &

# GPU 7: batch mode (非在线学习)
echo "GPU 7: batch mode"
CUDA_VISIBLE_DEVICES=7 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu7_batch_mode \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 4 \
    --batch-mode \
    > logs/gpu7_batch.log 2>&1 &

echo ""
echo "✅ 所有 8 个训练任务已启动！"
echo ""
echo "监控命令:"
echo "  查看所有日志: tail -f logs/gpu*.log"
echo "  查看特定 GPU: tail -f logs/gpu0_baseline.log"
echo "  查看 GPU 使用: watch -n 1 nvidia-smi"
echo "  查看进程: ps aux | grep python"
echo ""
echo "实验配置:"
echo "  GPU 0: baseline (lr=1e-6, beta=0.1, group=4)"
echo "  GPU 1: 更小学习率 (lr=5e-7)"
echo "  GPU 2: 更大学习率 (lr=2e-6)"
echo "  GPU 3: 更小 KL 惩罚 (beta=0.05)"
echo "  GPU 4: 更大 KL 惩罚 (beta=0.2)"
echo "  GPU 5: 更大 group size (group=8)"
echo "  GPU 6: 更小 group size (group=2)"
echo "  GPU 7: batch mode (非在线学习)"
echo ""

# 等待所有后台任务完成
wait

echo ""
echo "🎉 所有训练任务已完成！"
echo "结果保存在: $OUTPUT_BASE"
