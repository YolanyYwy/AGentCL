#!/bin/bash
# 快速开始：使用 Accelerate 进行多 GPU 训练

echo "=========================================="
echo "Accelerate 多 GPU 训练 - 快速开始"
echo "=========================================="
echo ""

# 检查 accelerate 是否安装
if ! command -v accelerate &> /dev/null; then
    echo "❌ Accelerate 未安装"
    echo "请运行: pip install accelerate"
    exit 1
fi

echo "✅ Accelerate 已安装"
echo ""

# 检查 GPU
echo "检查可用 GPU..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# 询问用户
read -p "使用多少个 GPU? (默认: 2): " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-2}

read -p "每个域多少任务? (默认: 10): " TASKS_PER_DOMAIN
TASKS_PER_DOMAIN=${TASKS_PER_DOMAIN:-10}

echo ""
echo "配置:"
echo "  GPU 数量: $NUM_GPUS"
echo "  每域任务数: $TASKS_PER_DOMAIN"
echo ""

# 修改 run.py 导入
echo "📝 修改 run.py 使用 Accelerate..."
sed -i.bak 's/from tau2.continual.training.grpo_trainer import/from tau2.continual.training.grpo_trainer_accelerate import/' run.py

if [ $? -eq 0 ]; then
    echo "✅ run.py 已修改（备份保存为 run.py.bak）"
else
    echo "⚠️  自动修改失败，请手动修改 run.py"
    echo "   将: from tau2.continual.training.grpo_trainer import GRPOContinualTrainer"
    echo "   改为: from tau2.continual.training.grpo_trainer_accelerate import GRPOContinualTrainer"
fi

echo ""
echo "🚀 启动训练..."
echo ""

# 启动训练
accelerate launch \
    --mixed_precision bf16 \
    --num_processes $NUM_GPUS \
    run.py \
    --model Qwen/Qwen3-4B \
    --device cuda \
    --tasks-per-domain $TASKS_PER_DOMAIN \
    --output-dir ./results_accelerate_${NUM_GPUS}gpu \
    --use-grpo

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "结果保存在: ./results_accelerate_${NUM_GPUS}gpu"
