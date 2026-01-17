#!/bin/bash
# GPU 训练监控脚本

echo "=========================================="
echo "GPU 训练监控面板"
echo "=========================================="
echo ""

# 显示 GPU 使用情况
echo "📊 GPU 使用情况:"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s | Temp: %s°C | GPU: %s%% | Mem: %s%% (%s/%s MB)\n", $1, $2, $3, $4, $5, $6, $7}'

echo ""
echo "🔄 运行中的训练进程:"
ps aux | grep "run_grpo_continual_learning.py" | grep -v grep | \
    awk '{printf "PID: %s | GPU: %s | CPU: %s%% | Mem: %s%% | Time: %s\n", $2, "N/A", $3, $4, $10}'

echo ""
echo "📝 最新训练日志 (最后 5 行):"
echo ""

for log in logs/gpu*.log; do
    if [ -f "$log" ]; then
        echo "--- $(basename $log) ---"
        tail -n 3 "$log" 2>/dev/null || echo "  (日志为空或不存在)"
        echo ""
    fi
done

echo "=========================================="
echo "刷新: watch -n 5 bash monitor_training.sh"
echo "停止所有训练: pkill -f run_grpo_continual_learning.py"
echo "=========================================="
