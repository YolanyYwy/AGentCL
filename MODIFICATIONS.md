# 代码修改说明

## 修改概述

本次修改主要实现了以下功能:

1. **使用 OpenAI API 替代 HuggingFace 模型作为 User Simulator** - 节省 GPU 资源
2. **支持中转 API** - 可以使用自定义的 API 地址和密钥（如 api.lingleap.com）
3. **支持多卡训练** - 使用 Accelerate 进行分布式训练

## 主要修改文件

### 1. 新增文件

#### `src/tau2/user/openai_user_simulator.py`
- 新增基于 OpenAI API 的 User Simulator
- 支持自定义 `api_base` 和 `api_key` 参数
- 使用 `gpt-4o-mini` 作为默认模型(成本低廉)
- 完全兼容现有的 UserSimulator 接口

#### `accelerate_config_single_gpu.yaml`
- 单卡训练配置文件
- 用于只有一张 GPU 或调试时使用

### 2. 修改文件

#### `run.py`
- 添加命令行参数支持中转 API 配置:
  - `--openai-api-base`: 自定义 API 地址
  - `--openai-api-key`: 自定义 API 密钥
  - `--openai-model`: 模型名称
- 修改评估器初始化部分
- User 改为使用 `openai_user_simulator` 类型
- User 参数改为 OpenAI API 参数 (temperature, max_tokens, api_base, api_key)

**修改位置**:
- 第 488-506 行: 添加命令行参数
- 第 133-149 行: 修改函数签名
- 第 256-291 行: 修改评估器初始化

#### `src/tau2/registry.py`
- 导入 `OpenAIUserSimulator`
- 注册 `openai_user_simulator` 类型

**修改位置**:
- 第 47 行: 添加 import
- 第 213 行: 注册新的 user 类型

#### `accelerate_config.yaml`
- 从单卡配置改为多卡配置
- `distributed_type: MULTI_GPU`
- `num_processes: 2` (根据实际 GPU 数量调整)
- `gpu_ids: all` (使用所有可用 GPU)

#### `run_with_accelerate.sh`
- 添加中转 API 配置区域
- 支持多种运行模式:
  - 默认: 多卡训练
  - `bash run_with_accelerate.sh single`: 单卡训练
  - `bash run_with_accelerate.sh custom config.yaml`: 自定义配置
- 增加任务数量到 10 (充分利用多卡)
- 添加使用提示和错误检查

## 使用方法

### 方式一: 使用中转 API（推荐）

#### 1. 修改 `run_with_accelerate.sh` 配置

编辑脚本开头的配置区域:

```bash
# OpenAI API 配置（中转 API）
OPENAI_API_BASE="https://api.lingleap.com/v1"  # 你的中转 API 地址
OPENAI_API_KEY="sk-xxx"  # 你的 API Key
OPENAI_MODEL="gpt-5"  # 模型名称
```

#### 2. 运行训练

```bash
# 多卡训练
bash run_with_accelerate.sh

# 单卡训练
bash run_with_accelerate.sh single
```

### 方式二: 使用命令行参数

```bash
accelerate launch --config_file accelerate_config.yaml run.py \
    --model Qwen/Qwen3-4B \
    --device cuda \
    --tasks-per-domain 10 \
    --output-dir ./results \
    --use-grpo \
    --openai-api-base https://api.lingleap.com/v1 \
    --openai-api-key sk-xxx \
    --openai-model gpt-5
```

### 方式三: 使用环境变量（标准 OpenAI API）

如果使用标准的 OpenAI API（不是中转）:

```bash
export OPENAI_API_KEY="your-api-key-here"

# 注释掉 run_with_accelerate.sh 中的中转 API 配置
# 或者不传递 --openai-api-base 参数

bash run_with_accelerate.sh
```

## 中转 API 配置示例

### Lingleap API

```bash
OPENAI_API_BASE="https://api.lingleap.com/v1"
OPENAI_API_KEY="sk-xxx"
OPENAI_MODEL="gpt-5"  # 或 gpt-4o, gpt-4o-mini 等
```

### 其他中转服务

```bash
# 示例 1: 自建中转
OPENAI_API_BASE="https://your-proxy.com/v1"
OPENAI_API_KEY="your-key"
OPENAI_MODEL="gpt-4o-mini"

# 示例 2: 其他中转服务
OPENAI_API_BASE="https://api.another-service.com/v1"
OPENAI_API_KEY="your-key"
OPENAI_MODEL="gpt-4"
```

## 资源使用对比

### 修改前 (使用 HuggingFace User)
- Agent: Qwen3-4B (本地模型) - 需要 GPU
- User: Qwen3-4B (本地模型) - 需要 GPU
- **总 GPU 需求**: 需要加载两个模型实例

### 修改后 (使用 OpenAI User)
- Agent: Qwen3-4B (本地模型) - 需要 GPU
- User: gpt-4o-mini/gpt-5 (OpenAI API) - 不需要 GPU
- **总 GPU 需求**: 只需要加载一个模型实例
- **成本**: OpenAI API 调用费用 (中转 API 通常更便宜)

## 多卡训练优势

1. **更快的训练速度**: 多个 GPU 并行训练
2. **更大的批次**: 可以处理更多任务
3. **更好的资源利用**: 充分利用多卡服务器

## 调整配置

### 调整 GPU 数量

编辑 `accelerate_config.yaml`:

```yaml
# 使用 2 张 GPU
num_processes: 2
gpu_ids: all

# 或者指定特定的 GPU
num_processes: 2
gpu_ids: [0, 1]

# 使用 4 张 GPU
num_processes: 4
gpu_ids: [0, 1, 2, 3]
```

### 调整任务数量

编辑 `run_with_accelerate.sh`:

```bash
TASKS_PER_DOMAIN=10  # 修改这个值
```

或者使用命令行参数:

```bash
accelerate launch --config_file accelerate_config.yaml run.py \
    --tasks-per-domain 20 \
    --other-args...
```

### 更换模型

```bash
# 在 run_with_accelerate.sh 中
OPENAI_MODEL="gpt-4o"  # 更强大的模型
OPENAI_MODEL="gpt-4o-mini"  # 更便宜的模型
OPENAI_MODEL="gpt-5"  # 最新模型（如果中转支持）
```

## 常见问题

### 1. 中转 API 连接失败

**问题**: 无法连接到中转 API

**解决方案**:
- 检查 `OPENAI_API_BASE` 是否正确（注意要包含 `/v1`）
- 检查 `OPENAI_API_KEY` 是否有效
- 检查网络连接
- 尝试使用 curl 测试:

```bash
curl https://api.lingleap.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 2. 模型名称不匹配

**问题**: 中转 API 返回模型不存在错误

**解决方案**:
- 检查中转服务支持的模型列表
- 修改 `OPENAI_MODEL` 为支持的模型名称
- 常见模型名称: `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-3.5-turbo`

### 3. NCCL 错误（多卡训练）

**问题**: 多卡训练时出现 NCCL 通信错误

**解决方案**:
- 检查 GPU 之间的通信是否正常
- 增加超时时间: 在 `accelerate_config.yaml` 中取消注释 `ddp_timeout`
- 确保所有 GPU 在同一台机器上

### 4. 显存不足

**解决方案**:
- 减少 `TASKS_PER_DOMAIN`
- 启用梯度检查点
- 使用更小的模型
- 增加 `gradient_accumulation_steps`

### 5. API 配额限制

**问题**: 中转 API 返回配额不足错误

**解决方案**:
- 检查 API 账户余额
- 降低请求频率（减少 `TASKS_PER_DOMAIN`）
- 联系中转服务提供商

## 测试建议

1. **先测试 API 连接**: 使用 curl 或简单脚本测试中转 API 是否正常

2. **先用单卡测试**: 确保代码正常运行
```bash
bash run_with_accelerate.sh single
```

3. **再用多卡训练**: 确认多卡配置正确
```bash
bash run_with_accelerate.sh
```

4. **监控资源使用**: 使用 `nvidia-smi` 监控 GPU 使用情况
```bash
watch -n 1 nvidia-smi
```

## 进一步优化

如果需要进一步优化,可以考虑:

1. **使用 DeepSpeed**: 取消注释 `accelerate_config.yaml` 中的 DeepSpeed 配置
2. **使用更便宜的模型**: 如 `gpt-3.5-turbo` 或 `gpt-4o-mini`
3. **批量处理**: 增加批次大小以提高吞吐量
4. **混合精度训练**: 已启用 bf16,可以尝试 fp16

## 回滚方法

如果需要回滚到原来的配置:

1. 在 `run.py` 中恢复使用 `hf_user_simulator`:
```python
user_type='hf_user_simulator' if use_grpo else 'user_simulator',
```

2. 恢复 User LLM 参数:
```python
user_llm=None,
llm_args_user={
    'model_name_or_path': model_name,
    'load_in_4bit': True,
} if use_grpo else {},
```

3. 使用单卡配置:
```bash
bash run_with_accelerate.sh single
```

## 技术细节

### litellm 支持

代码使用 `litellm` 库来调用 OpenAI API，它支持:
- 自定义 `api_base` 参数
- 自定义 `api_key` 参数
- 多种 API 提供商（OpenAI, Azure, 自定义中转等）

### 参数传递流程

```
run_with_accelerate.sh (配置)
    ↓
run.py (命令行参数)
    ↓
ContinualLearningEvaluator (llm_args_user)
    ↓
OpenAIUserSimulator (api_base, api_key)
    ↓
generate() 函数 (litellm)
    ↓
中转 API / OpenAI API
```

### 兼容性

- 兼容标准 OpenAI API
- 兼容各种中转服务（只要遵循 OpenAI API 格式）
- 支持自定义模型名称
