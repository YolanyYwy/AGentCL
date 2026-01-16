# τ²-Bench Continual Learning Benchmark

## 架构设计原则

本 Benchmark 框架与评测算法**严格分离**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Continual Learning Benchmark Framework                │
│                         (本项目提供)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   Curriculum    │    │   Evaluation    │    │    Metrics      │     │
│  │   Definition    │    │   Protocol      │    │   Computation   │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│           │                     │                      │                │
│           └─────────────────────┼──────────────────────┘                │
│                                 │                                        │
│                                 ▼                                        │
│                    ┌─────────────────────────┐                          │
│                    │    Agent Interface      │                          │
│                    │  (Abstract Protocol)    │                          │
│                    └─────────────────────────┘                          │
│                                 ▲                                        │
└─────────────────────────────────┼────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
         ┌──────────▼──────────┐     ┌─────────▼──────────┐
         │  Your Algorithm     │     │  Baseline Methods  │
         │  (用户自己实现)       │     │  (示例实现)         │
         │                     │     │                    │
         │  - 本地模型训练      │     │  - Vanilla SFT     │
         │  - 持续学习算法      │     │  - ICL Baseline    │
         │  - 自定义策略        │     │  - Replay Buffer   │
         └─────────────────────┘     └────────────────────┘
```

## 核心概念

### 1. Benchmark 框架提供什么

- **课程定义** (Curriculum): 定义学习阶段、工具序列、任务分配
- **评测协议** (Evaluation Protocol): 标准化的学习-评测-保持测试流程
- **指标计算** (Metrics): FWT、BWT、遗忘率、泛化性等
- **Agent 接口** (Agent Interface): 抽象接口，用户实现具体算法

### 2. 用户需要实现什么

- **ContinualAgent**: 实现 `learn()` 和 `act()` 方法
- **训练逻辑**: 如何更新模型参数
- **持续学习策略**: EWC、Replay、正则化等

## Agent 接口定义

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from tau2.data_model.message import Message
from tau2.continual.curriculum.stage import LearningStage

class ContinualAgent(ABC):
    """
    持续学习 Agent 的抽象接口。

    用户需要继承此类并实现具体的学习和推理逻辑。
    """

    @abstractmethod
    def learn(
        self,
        stage: LearningStage,
        experiences: list[dict],
    ) -> dict[str, Any]:
        """
        在一个学习阶段上学习。

        Args:
            stage: 当前学习阶段，包含学习材料和工具信息
            experiences: 学习经验列表，每个经验包含:
                - messages: 对话历史
                - tool_calls: 工具调用
                - reward: 奖励信号
                - task_id: 任务ID

        Returns:
            学习统计信息（如 loss、更新步数等）
        """
        pass

    @abstractmethod
    def act(
        self,
        messages: list[Message],
        available_tools: list[dict],
    ) -> Message:
        """
        根据对话历史和可用工具生成响应。

        Args:
            messages: 对话历史
            available_tools: 当前可用的工具 schema 列表

        Returns:
            Agent 的响应消息（文本或工具调用）
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存模型检查点。"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """加载模型检查点。"""
        pass

    def on_stage_start(self, stage: LearningStage) -> None:
        """阶段开始时的回调（可选实现）。"""
        pass

    def on_stage_end(self, stage: LearningStage, metrics: dict) -> None:
        """阶段结束时的回调（可选实现）。"""
        pass
```

## 使用流程

### 1. 实现你的 Agent

```python
from tau2.continual.benchmark.agent_interface import ContinualAgent
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyContinualAgent(ContinualAgent):
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 你的持续学习算法组件
        self.replay_buffer = []
        self.ewc_fisher = None

    def learn(self, stage, experiences):
        # 实现你的学习逻辑
        # 例如：EWC、Replay、正则化等
        pass

    def act(self, messages, available_tools):
        # 实现你的推理逻辑
        pass
```

### 2. 运行 Benchmark

```python
from tau2.continual.benchmark import ContinualBenchmark

# 加载 benchmark
benchmark = ContinualBenchmark.from_curriculum(
    curriculum_path="data/tau2/curricula/airline_progressive.json",
    domain="airline",
)

# 创建你的 agent
agent = MyContinualAgent(model_path="your-model-path")

# 运行评测
results = benchmark.evaluate(agent)

# 查看结果
print(results.metrics.summary())
```

## 目录结构

```
src/tau2/continual/
├── benchmark/                    # Benchmark 框架（核心）
│   ├── __init__.py
│   ├── agent_interface.py       # Agent 抽象接口
│   ├── benchmark.py             # 主 Benchmark 类
│   ├── protocol.py              # 评测协议
│   └── metrics.py               # 指标计算
├── curriculum/                   # 课程定义
│   ├── curriculum.py
│   ├── stage.py
│   └── task_selector.py
├── baselines/                    # 示例 baseline（供参考）
│   ├── vanilla_finetune.py
│   ├── icl_baseline.py
│   └── replay_baseline.py
└── utils/
    ├── data_utils.py
    └── tool_utils.py
```
