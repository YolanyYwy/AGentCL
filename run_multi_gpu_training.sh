#!/bin/bash
# 多 GPU 并行训练脚本
# 在 8 张 GPU 上同时运行不同的实验

# 设置基础参数
MODEL="Qwen/Qwen3-4B"
OUTPUT_BASE="./grpo_results"

# GPU 0: 训练 airline 域
CUDA_VISIBLE_DEVICES=0 python run_grpo_continual_learning.py \
    --domains airline \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu0_airline \
    --learning-rate 1e-6 \
    --beta 0.1 \
    > logs/gpu0_airline.log 2>&1 &

# GPU 1: 训练 retail 域
CUDA_VISIBLE_DEVICES=1 python run_grpo_continual_learning.py \
    --domains retail \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu1_retail \
    --learning-rate 1e-6 \
    --beta 0.1 \
    > logs/gpu1_retail.log 2>&1 &

# GPU 2: 训练 telecom 域
CUDA_VISIBLE_DEVICES=2 python run_grpo_continual_learning.py \
    --domains telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu2_telecom \
    --learning-rate 1e-6 \
    --beta 0.1 \
    > logs/gpu2_telecom.log 2>&1 &

# GPU 3: 顺序训练 airline -> retail
CUDA_VISIBLE_DEVICES=3 python run_grpo_continual_learning.py \
    --domains airline retail \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu3_airline_retail \
    --learning-rate 1e-6 \
    --beta 0.1 \
    > logs/gpu3_airline_retail.log 2>&1 &

# GPU 4: 顺序训练 retail -> telecom
CUDA_VISIBLE_DEVICES=4 python run_grpo_continual_learning.py \
    --domains retail telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu4_retail_telecom \
    --learning-rate 1e-6 \
    --beta 0.1 \
    > logs/gpu4_retail_telecom.log 2>&1 &

# GPU 5: 顺序训练 airline -> retail -> telecom
CUDA_VISIBLE_DEVICES=5 python run_grpo_continual_learning.py \
    --domains airline retail telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu5_all_domains \
    --learning-rate 1e-6 \
    --beta 0.1 \
    > logs/gpu5_all_domains.log 2>&1 &

# GPU 6: 测试不同学习率 (5e-7)
CUDA_VISIBLE_DEVICES=6 python run_grpo_continual_learning.py \
    --domains airline retail telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu6_lr5e7 \
    --learning-rate 5e-7 \
    --beta 0.1 \
    > logs/gpu6_lr5e7.log 2>&1 &

# GPU 7: 测试不同学习率 (2e-6)
CUDA_VISIBLE_DEVICES=7 python run_grpo_continual_learning.py \
    --domains airline retail telecom \
    --model $MODEL \
    --device cuda \
    --output-dir ${OUTPUT_BASE}/gpu7_lr2e6 \
    --learning-rate 2e-6 \
    --beta 0.1 \
    > logs/gpu7_lr2e6.log 2>&1 &

echo "所有训练任务已启动！"
echo "查看日志: tail -f logs/gpu*.log"
echo "查看 GPU 使用情况: watch -n 1 nvidia-smi"

# 等待所有后台任务完成
wait

echo "所有训练任务已完成！"
