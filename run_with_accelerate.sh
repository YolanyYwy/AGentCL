#!/bin/bash
# 使用 Accelerate 启动多卡训练
#
# 使用说明:
# 1. 多卡训练: bash run_with_accelerate.sh
# 2. 单卡训练: bash run_with_accelerate.sh single
# 3. 自定义配置: bash run_with_accelerate.sh custom path/to/config.yaml

# ============================================
# 配置区域 - 请根据你的实际情况修改
# ============================================

# 模型配置
MODEL="Qwen/Qwen3-4B"
TASKS_PER_DOMAIN=10  # 增加任务数量以充分利用多卡
OUTPUT_DIR="./results_accelerate"

# OpenAI API 配置（中转 API）
# 如果使用中转 API，请取消注释并填写以下配置
OPENAI_API_BASE="https://api.lingleap.com/v1"  # 中转 API 地址
OPENAI_API_KEY="sk-st5yh98uy1h854ngLMmiDkruIe8pPJPuLtZzCjM2a0qufOO4"  # 你的 API Key
OPENAI_MODEL="gpt-4"  # 模型名称

# 如果不使用中转 API，注释掉上面三行，使用环境变量
# export OPENAI_API_KEY="your-api-key-here"

# ============================================
# 以下代码无需修改
# ============================================

# 检查命令行参数
MODE=${1:-multi}  # 默认使用多卡模式

if [ "$MODE" == "single" ]; then
    CONFIG_FILE="accelerate_config_single_gpu.yaml"
    echo "=========================================="
    echo "使用 Accelerate 进行单 GPU 训练"
    echo "=========================================="
elif [ "$MODE" == "custom" ]; then
    CONFIG_FILE=${2:-"accelerate_config.yaml"}
    echo "=========================================="
    echo "使用自定义配置文件: $CONFIG_FILE"
    echo "=========================================="
else
    CONFIG_FILE="accelerate_config.yaml"
    echo "=========================================="
    echo "使用 Accelerate 进行多 GPU 训练"
    echo "=========================================="
fi

echo "模型: $MODEL"
echo "每域任务数: $TASKS_PER_DOMAIN"
echo "配置文件: $CONFIG_FILE"
echo "OpenAI 模型: $OPENAI_MODEL"
if [ ! -z "$OPENAI_API_BASE" ]; then
    echo "API Base: $OPENAI_API_BASE"
fi
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 构建命令参数
CMD_ARGS="--model $MODEL --device cuda --tasks-per-domain $TASKS_PER_DOMAIN --output-dir $OUTPUT_DIR --use-grpo"

# 添加 OpenAI API 配置（如果设置了）
if [ ! -z "$OPENAI_API_BASE" ]; then
    CMD_ARGS="$CMD_ARGS --openai-api-base $OPENAI_API_BASE"
fi

if [ ! -z "$OPENAI_API_KEY" ]; then
    CMD_ARGS="$CMD_ARGS --openai-api-key $OPENAI_API_KEY"
fi

if [ ! -z "$OPENAI_MODEL" ]; then
    CMD_ARGS="$CMD_ARGS --openai-model $OPENAI_MODEL"
fi

# 使用配置文件启动
echo "执行命令: accelerate launch --config_file $CONFIG_FILE run.py $CMD_ARGS"
echo ""

accelerate launch \
    --config_file $CONFIG_FILE \
    run.py \
    $CMD_ARGS

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "提示:"
echo "- 查看训练日志以了解多卡训练的详细信息"
echo "- 如果遇到 NCCL 错误，请检查 GPU 之间的通信"
echo "- 如果显存不足，可以减少 tasks-per-domain 或启用梯度检查点"
echo "- 如果使用中转 API，请确保在脚本开头正确配置了 API 地址和密钥"


