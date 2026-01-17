# 显存优化指南

## 问题诊断

你遇到的 CUDA OOM 错误是由以下原因导致的:

### 1. Accelerate 配置问题
- **问题**: `distributed_type: MULTI_GPU` 但 `num_processes: 1` 配置矛盾
- **结果**: Accelerate 可能启动了多个进程
- **已修复**: 改为 `distributed_type: NO` (单 GPU 模式)

### 2. device_map="auto" 问题
- **问题**: `device_map="auto"` 会让 HuggingFace 自动将模型分布到所有可用 GPU
- **结果**: 模型被分散到多个 GPU,每个 GPU 都占用显存
- **已修复**: 强制使用 `device_map={"": 0}` 将所有层放在 GPU 0

### 3. 多模型实例
- **问题**: 每个任务同时加载 Agent 和 User 两个模型实例
- **结果**: 显存占用翻倍
- **当前状态**: Agent 使用独立实例,User 使用共享缓存实例

## 已应用的修复

### 1. accelerate_config.yaml
```yaml
distributed_type: NO  # 单 GPU 模式
num_processes: 1
gpu_ids: [0]
gradient_accumulation_steps: 4  # 增加梯度累积
```

### 2. hf_model_cache.py
```python
# 强制使用单个 GPU
if device == "auto":
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}  # 所有层在 GPU 0
```

## 进一步优化建议

### 选项 1: 启用 4-bit 量化 (推荐)
在 `run.py` 中修改:

```python
llm_args_agent={
    'model_name_or_path': model_name,
    'load_in_4bit': True,  # 启用 4-bit 量化
}
llm_args_user={
    'model_name_or_path': model_name,
    'load_in_4bit': True,  # 启用 4-bit 量化
}
```

**效果**: 显存占用减少约 75%

### 选项 2: 使用更小的模型
```bash
python run.py --model Qwen/Qwen3-1.8B  # 使用 1.8B 模型
```

### 选项 3: 减少 batch size 和序列长度
在 `grpo_trainer_accelerate.py` 中:
```python
batch_size = 1  # 减小 batch size
max_new_tokens = 256  # 减少生成长度
```

### 选项 4: 启用梯度检查点
在模型加载时:
```python
model.gradient_checkpointing_enable()
```

## 运行命令

### 使用配置文件运行 (推荐)
```bash
accelerate launch --config_file accelerate_config.yaml run.py \
    --model Qwen/Qwen3-4B \
    --device cuda \
    --tasks-per-domain 1 \
    --output-dir ./results \
    --use-grpo
```

### 直接运行 (不使用 accelerate)
```bash
python run.py \
    --model Qwen/Qwen3-4B \
    --device cuda \
    --tasks-per-domain 1 \
    --output-dir ./results \
    --use-grpo
```

## 显存使用估算

### Qwen3-4B 模型 (不同配置)
- **FP32**: ~16 GB
- **BF16**: ~8 GB
- **8-bit**: ~4 GB
- **4-bit**: ~2 GB

### 当前配置 (2 个模型实例, BF16)
- Agent 模型: ~8 GB
- User 模型 (共享): ~8 GB
- 激活值和梯度: ~4-8 GB
- **总计**: ~20-24 GB

### 优化后 (4-bit 量化)
- Agent 模型: ~2 GB
- User 模型 (共享): ~2 GB
- 激活值和梯度: ~2-4 GB
- **总计**: ~6-8 GB

## 监控显存使用

```bash
# 实时监控 GPU 显存
watch -n 1 nvidia-smi

# 或使用 Python
python -c "import torch; print(f'GPU 0: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')"
```

## 故障排除

### 如果仍然 OOM:
1. 检查是否有其他进程占用 GPU: `nvidia-smi`
2. 清理 GPU 缓存: `torch.cuda.empty_cache()`
3. 减少 `tasks_per_domain` 参数
4. 使用更小的模型或启用量化
5. 设置环境变量: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
