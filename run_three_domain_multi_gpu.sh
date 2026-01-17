#!/bin/bash
# 三域持续学习多 GPU 并行训练脚本
# 在多张 GPU 上同时运行不同超参数配置的实验

MODEL="Qwen/Qwen3-4B"
AIRLINE_TASKS="data/tau2/domains/airline/tasks.json"
RETAIL_TASKS="data/tau2/domains/retail/tasks.json"
TELECOM_TASKS="data/tau2/domains/telecom/tasks_hard_300.json"
OUTPUT_BASE="./three_domain_results"
TASKS_PER_DOMAIN=100

# 创建日志目录
mkdir -p logs
mkdir -p $OUTPUT_BASE

echo "=========================================="
echo "三域持续学习多 GPU 并行训练"
echo "=========================================="
echo "训练顺序: Airline → Retail → Telecom"
echo "模型: $MODEL"
echo "每域任务数: $TASKS_PER_DOMAIN"
echo ""

# GPU 0: Baseline (lr=1e-6, beta=0.1, group=4)
echo "🚀 GPU 0: Baseline 配置"
CUDA_VISIBLE_DEVICES=0 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 4 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu0_baseline \
    > logs/gpu0_baseline.log 2>&1 &

# GPU 1: 更小学习率 (lr=5e-7)
echo "🚀 GPU 1: 更小学习率 (lr=5e-7)"
CUDA_VISIBLE_DEVICES=1 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 5e-7 \
    --beta 0.1 \
    --group-size 4 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu1_lr5e7 \
    > logs/gpu1_lr5e7.log 2>&1 &

# GPU 2: 更大学习率 (lr=2e-6)
echo "🚀 GPU 2: 更大学习率 (lr=2e-6)"
CUDA_VISIBLE_DEVICES=2 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 2e-6 \
    --beta 0.1 \
    --group-size 4 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu2_lr2e6 \
    > logs/gpu2_lr2e6.log 2>&1 &

# GPU 3: 更小 KL 惩罚 (beta=0.05)
echo "🚀 GPU 3: 更小 KL 惩罚 (beta=0.05)"
CUDA_VISIBLE_DEVICES=3 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 1e-6 \
    --beta 0.05 \
    --group-size 4 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu3_beta0.05 \
    > logs/gpu3_beta0.05.log 2>&1 &

# GPU 4: 更大 KL 惩罚 (beta=0.2)
echo "🚀 GPU 4: 更大 KL 惩罚 (beta=0.2)"
CUDA_VISIBLE_DEVICES=4 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 1e-6 \
    --beta 0.2 \
    --group-size 4 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu4_beta0.2 \
    > logs/gpu4_beta0.2.log 2>&1 &

# GPU 5: 更大 group size (group=8)
echo "🚀 GPU 5: 更大 group size (group=8)"
CUDA_VISIBLE_DEVICES=5 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 8 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu5_group8 \
    > logs/gpu5_group8.log 2>&1 &

# GPU 6: 更小 group size (group=2)
echo "🚀 GPU 6: 更小 group size (group=2)"
CUDA_VISIBLE_DEVICES=6 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --use-grpo \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --group-size 2 \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu6_group2 \
    > logs/gpu6_group2.log 2>&1 &

# GPU 7: 不使用 GRPO (仅评估)
echo "🚀 GPU 7: 不使用 GRPO (仅评估)"
CUDA_VISIBLE_DEVICES=7 python run_three_domain_continual_learning.py \
    --airline-tasks $AIRLINE_TASKS \
    --retail-tasks $RETAIL_TASKS \
    --telecom-tasks $TELECOM_TASKS \
    --model $MODEL \
    --device cuda \
    --no-grpo \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ${OUTPUT_BASE}/gpu7_no_grpo \
    > logs/gpu7_no_grpo.log 2>&1 &

echo ""
echo "=========================================="
echo "✅ 所有 8 个训练任务已启动！"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  GPU 0: Baseline (lr=1e-6, beta=0.1, group=4)"
echo "  GPU 1: 更小学习率 (lr=5e-7)"
echo "  GPU 2: 更大学习率 (lr=2e-6)"
echo "  GPU 3: 更小 KL 惩罚 (beta=0.05)"
echo "  GPU 4: 更大 KL 惩罚 (beta=0.2)"
echo "  GPU 5: 更大 group size (group=8)"
echo "  GPU 6: 更小 group size (group=2)"
echo "  GPU 7: 不使用 GRPO (仅评估)"
echo ""
echo "监控命令:"
echo "  查看所有日志: tail -f logs/gpu*.log"
echo "  查看特定 GPU: tail -f logs/gpu0_baseline.log"
echo "  查看 GPU 使用: watch -n 1 nvidia-smi"
echo "  查看进程: ps aux | grep python"
echo ""
echo "停止所有训练: pkill -f run_three_domain_continual_learning.py"
echo "=========================================="

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "🎉 所有训练任务已完成！"
echo "=========================================="
echo "结果保存在: $OUTPUT_BASE"
echo ""
echo "查看结果:"
echo "  cat ${OUTPUT_BASE}/gpu0_baseline/metrics.json"
echo "  python analyze_three_domain_results.py --results-dir $OUTPUT_BASE"
