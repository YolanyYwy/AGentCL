# τ²-Bench Continual Learning: Agent 工具使用持续学习能力评测

## 核心立论：为什么需要这个 Benchmark

### 1. 能力重要性论证

**现实场景中工具环境是动态变化的：**

| 变化类型 | 现实例子 | 对 Agent 的要求 |
|---------|---------|----------------|
| **工具新增** | 航空公司上线新的改签系统 | 快速学习新工具，不影响已有能力 |
| **工具更新** | API 参数变更、返回格式修改 | 适应变化，避免调用失败 |
| **工具废弃** | 旧系统下线，新系统替代 | 迁移知识，停止使用废弃工具 |
| **跨域迁移** | 从航空客服扩展到酒店客服 | 复用工具使用模式 |

**核心论点**：当前 benchmark（包括 τ²-Bench）只评估静态工具集上的**一次性能力**，无法回答：
- Agent 能否持续学习新工具？
- 学习新工具后，旧工具能力是否退化？
- 工具更新后，Agent 能否快速适应？

### 2. 当前能力不足的证据

基于 τ²-Bench 数据集分析，我们发现：

```
工具使用频率分布（Airline 领域）：
┌─────────────────────────────────┬───────┬────────┐
│ 工具名称                         │ 使用次数 │ 占比    │
├─────────────────────────────────┼───────┼────────┤
│ get_reservation_details         │   57  │ 38.3%  │
│ update_reservation_flights      │   21  │ 14.1%  │
│ search_direct_flight            │   20  │ 13.4%  │
│ get_user_details                │   14  │  9.4%  │
│ cancel_reservation              │   13  │  8.7%  │
│ book_reservation                │    9  │  6.0%  │
│ update_reservation_baggages     │    6  │  4.0%  │
│ send_certificate                │    3  │  2.0%  │
│ update_reservation_passengers   │    3  │  2.0%  │
│ calculate                       │    1  │  0.7%  │
│ transfer_to_human_agents        │    1  │  0.7%  │
└─────────────────────────────────┴───────┴────────┘
```

**问题**：
1. 高频工具（如 `get_reservation_details`）被过度训练
2. 低频工具（如 `calculate`）学习不充分
3. 工具间存在**能力不平衡**，持续学习场景下可能加剧

---

## 第一部分：任务序列设计

### 1.1 工具依赖图分析

基于 Airline 领域的工具，构建工具依赖关系：

```
                    ┌─────────────────┐
                    │ list_all_airports│ (独立)
                    └─────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                          查询层 (READ)                            │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │ get_user_details│───▶│get_reservation_details│               │
│  └─────────────────┘    └──────────────────────┘                │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │search_direct_   │    │  search_onestop_     │                │
│  │    flight       │    │      flight          │                │
│  └─────────────────┘    └──────────────────────┘                │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                          操作层 (WRITE)                           │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │ book_reservation│    │  cancel_reservation  │                │
│  └─────────────────┘    └──────────────────────┘                │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │update_flights   │    │  update_baggages     │                │
│  └─────────────────┘    └──────────────────────┘                │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │update_passengers│    │   send_certificate   │                │
│  └─────────────────┘    └──────────────────────┘                │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                          辅助层 (GENERIC)                         │
│  ┌─────────────────┐    ┌──────────────────────┐                │
│  │    calculate    │    │transfer_to_human     │                │
│  └─────────────────┘    └──────────────────────┘                │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 阶段划分（4 阶段渐进式）

基于工具依赖和任务复杂度，设计以下任务序列：

#### **Stage 0: 基础查询阶段（Foundation）**

| 属性 | 值 |
|-----|---|
| **可用工具** | `get_user_details`, `get_reservation_details`, `list_all_airports` |
| **工具数量** | 3 |
| **任务类型** | 信息查询、策略拒绝 |
| **任务复杂度** | Level 1 (0-2 actions) |

**任务示例**（从现有数据集筛选）：
```json
{
  "learning_tasks": ["task_1", "task_5", "task_6"],
  "eval_tasks": ["task_0", "task_3", "task_4"],
  "task_characteristics": {
    "task_0": "策略拒绝 - 拒绝无效取消请求",
    "task_1": "基础查询 - 获取用户和预订信息",
    "task_3": "会员验证 - 检查会员等级",
    "task_5": "会员查询 - 验证会员状态"
  }
}
```

**评测重点**：
- 基础工具调用准确率
- 参数传递正确性
- 策略遵守能力

---

#### **Stage 1: 搜索扩展阶段（Search Expansion）**

| 属性 | 值 |
|-----|---|
| **新增工具** | `search_direct_flight`, `search_onestop_flight` |
| **累计工具** | 5 |
| **任务类型** | 航班搜索、信息筛选 |
| **任务复杂度** | Level 1-2 (1-4 actions) |

**新增工具学习材料**：
```markdown
## 新工具：search_direct_flight

搜索两个城市之间的直飞航班。

**参数**：
- origin (str): 出发机场代码，如 'JFK'
- destination (str): 目的机场代码，如 'LAX'
- date (str): 日期，格式 'YYYY-MM-DD'

**返回**：可用航班列表，包含航班号、时间、价格、座位信息

**使用示例**：
User: 我想查一下明天从纽约到洛杉矶的航班
Agent: [调用 search_direct_flight(origin='JFK', destination='LAX', date='2024-05-16')]
```

**任务分配**：
```json
{
  "learning_tasks": ["task_8", "task_9"],
  "eval_tasks": ["task_7", "task_10", "task_11"],
  "retention_tasks": ["task_0", "task_1", "task_3"]
}
```

**评测重点**：
- **新工具学习**：search 类工具使用准确率
- **知识保持**：Stage 0 工具能力是否下降
- **组合能力**：查询 + 搜索的工具链

---

#### **Stage 2: 预订操作阶段（Booking Operations）**

| 属性 | 值 |
|-----|---|
| **新增工具** | `book_reservation`, `cancel_reservation` |
| **累计工具** | 7 |
| **任务类型** | 预订创建、预订取消 |
| **任务复杂度** | Level 2-3 (3-6 actions) |

**新增工具学习材料**：
```markdown
## 新工具：book_reservation

创建新的航班预订。

**参数**：
- user_id (str): 用户ID
- origin (str): 出发机场
- destination (str): 目的机场
- flight_type (str): 'one_way' 或 'round_trip'
- cabin (str): 'basic_economy', 'economy', 'business'
- flights (list): 航班信息列表
- passengers (list): 乘客信息列表
- payment_methods (list): 支付方式列表
- total_baggages (int): 行李总数
- nonfree_baggages (int): 付费行李数
- insurance (str): 'yes' 或 'no'

**关键约束**：
- 支付金额必须等于总价
- 需要验证座位可用性
- Certificate 可用于支付，Gift Card 需检查余额

**使用示例**：
[完整的预订流程示例]
```

**任务分配**：
```json
{
  "learning_tasks": ["task_14", "task_20"],
  "eval_tasks": ["task_23", "task_24", "task_25", "task_35"],
  "retention_tasks": ["task_1", "task_7", "task_10"]
}
```

**评测重点**：
- **复杂参数处理**：多参数工具的正确调用
- **支付逻辑**：金额计算和支付方式选择
- **工具链执行**：查询 → 搜索 → 预订的完整流程

---

#### **Stage 3: 高级修改阶段（Advanced Modifications）**

| 属性 | 值 |
|-----|---|
| **新增工具** | `update_reservation_flights`, `update_reservation_baggages`, `update_reservation_passengers`, `send_certificate`, `calculate` |
| **累计工具** | 12 |
| **任务类型** | 预订修改、补偿处理 |
| **任务复杂度** | Level 3-4 (4-18 actions) |

**任务分配**：
```json
{
  "learning_tasks": ["task_2", "task_12", "task_15"],
  "eval_tasks": ["task_42", "task_44"],
  "retention_tasks": ["task_0", "task_7", "task_14", "task_23"]
}
```

**评测重点**：
- **复杂推理**：多实体协调（Task 44 涉及 5+ 预订）
- **支付优化**：Certificate/Gift Card 最优分配
- **全工具链**：完整的客服场景处理

---

### 1.3 任务序列汇总

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         任务序列时间轴                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 0          Stage 1           Stage 2           Stage 3           │
│  ────────         ────────          ────────          ────────          │
│  [3 tools]        [+2 tools]        [+2 tools]        [+5 tools]        │
│                                                                          │
│  get_user         search_direct     book_reservation  update_flights    │
│  get_reservation  search_onestop    cancel_reservation update_baggages  │
│  list_airports                                        update_passengers │
│                                                       send_certificate  │
│                                                       calculate         │
│                                                                          │
│  Level 1          Level 1-2         Level 2-3         Level 3-4         │
│  (0-2 actions)    (1-4 actions)     (3-6 actions)     (4-18 actions)    │
│                                                                          │
│  ┌─────┐          ┌─────┐           ┌─────┐           ┌─────┐           │
│  │Learn│ ───────▶ │Learn│ ───────▶  │Learn│ ───────▶  │Learn│           │
│  │Eval │          │Eval │           │Eval │           │Eval │           │
│  └─────┘          │Retain│          │Retain│          │Retain│          │
│                   └─────┘           └─────┘           └─────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 第二部分：训练与评测流程

### 2.1 训练方式：Vanilla SFT（无特殊算法）

为了建立公平的 baseline，我们首先使用**最简单的训练方式**：

```python
class VanillaContinualTraining:
    """
    Vanilla 持续学习训练
    - 无 replay buffer
    - 无正则化
    - 无特殊持续学习算法
    """

    def __init__(self, base_model: str, learning_rate: float = 2e-5):
        self.model = load_model(base_model)
        self.lr = learning_rate
        self.stage_checkpoints = {}

    def train_stage(self, stage_id: str, training_data: list[dict]):
        """
        在单个阶段上训练

        Args:
            stage_id: 阶段标识
            training_data: 该阶段的训练数据（来自 learning_tasks）
        """
        # 转换为 SFT 格式
        sft_data = self._convert_to_sft_format(training_data)

        # 标准 SFT 训练
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=sft_data,
            learning_rate=self.lr,
            num_epochs=3,
            # 无任何持续学习技巧
        )
        trainer.train()

        # 保存阶段检查点
        self.stage_checkpoints[stage_id] = self.model.save_checkpoint()

    def _convert_to_sft_format(self, runs: list[SimulationRun]) -> Dataset:
        """
        将 SimulationRun 转换为 SFT 训练格式

        格式：
        - input: system prompt + conversation history
        - output: agent's tool calls or responses
        """
        examples = []
        for run in runs:
            for i, msg in enumerate(run.messages):
                if isinstance(msg, AssistantMessage):
                    context = self._build_context(run.messages[:i])
                    target = self._format_assistant_message(msg)
                    examples.append({
                        "input": context,
                        "output": target
                    })
        return Dataset.from_list(examples)
```

### 2.2 ICL Baseline（无训练）

同时提供纯 ICL 的 baseline：

```python
class ICLContinualBaseline:
    """
    In-Context Learning Baseline
    - 通过 prompt 中的示例学习
    - 无参数更新
    """

    def __init__(self, base_model: str):
        self.model = load_model(base_model)
        self.stage_examples = {}  # 每阶段的示例缓存

    def learn_stage(self, stage_id: str, learning_materials: list[dict]):
        """
        将学习材料转换为 few-shot examples
        """
        examples = []
        for material in learning_materials:
            if material["type"] == "example":
                examples.append(self._parse_example(material["content"]))
        self.stage_examples[stage_id] = examples

    def generate(self, stage_id: str, conversation: list[Message]) -> Message:
        """
        使用累积的 few-shot examples 生成响应
        """
        # 构建包含所有阶段示例的 prompt
        prompt = self._build_prompt_with_examples(
            current_stage=stage_id,
            conversation=conversation
        )
        return self.model.generate(prompt)

    def _build_prompt_with_examples(self, current_stage: str, conversation: list) -> str:
        """
        累积所有已学阶段的示例（测试 ICL 的持续学习能力）
        """
        all_examples = []
        for stage_id, examples in self.stage_examples.items():
            if stage_id <= current_stage:
                all_examples.extend(examples)

        # 限制示例数量（避免上下文溢出）
        max_examples = 5
        selected = all_examples[-max_examples:] if len(all_examples) > max_examples else all_examples

        return self._format_few_shot_prompt(selected, conversation)
```

### 2.3 完整评测流程

```python
class ContinualLearningEvaluator:
    """
    持续学习完整评测流程
    """

    def __init__(self, curriculum: Curriculum, agent: BaseAgent):
        self.curriculum = curriculum
        self.agent = agent
        self.results = ContinualLearningResults()

        # 用于计算指标的历史数据
        self.tool_performance_history = defaultdict(list)  # tool_name -> [(stage, accuracy)]
        self.stage_performance_history = []  # [(stage, eval_acc, retention_acc)]

    def run_full_evaluation(self) -> ContinualLearningResults:
        """
        运行完整的持续学习评测
        """
        for stage_idx, stage in enumerate(self.curriculum.stages):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx}: {stage.stage_name}")
            print(f"{'='*60}")

            # 1. 学习阶段
            if stage.learning_tasks:
                self._run_learning_phase(stage)

            # 2. 评测新任务
            eval_results = self._run_evaluation_phase(stage)

            # 3. 评测知识保持（retention）
            retention_results = self._run_retention_phase(stage) if stage.retention_tasks else None

            # 4. 记录工具级别的性能
            self._record_tool_performance(stage, eval_results, retention_results)

            # 5. 保存阶段结果
            self.results.add_stage_result(StageResult(
                stage_id=stage.stage_id,
                eval_results=eval_results,
                retention_results=retention_results,
                tool_performance=self._get_current_tool_performance()
            ))

        # 6. 计算最终指标
        self.results.metrics = self._compute_final_metrics()

        return self.results

    def _run_learning_phase(self, stage: LearningStage):
        """学习阶段：让 Agent 在学习任务上练习"""
        print(f"  Learning Phase: {len(stage.learning_tasks)} tasks")

        for task_id in stage.learning_tasks:
            task = self.curriculum.get_task(task_id)
            for trial in range(stage.num_learning_trials):
                run = self._execute_task(task, is_learning=True)
                # 学习阶段的结果用于训练/更新 Agent
                self.agent.learn_from_experience(run)

    def _run_evaluation_phase(self, stage: LearningStage) -> list[SimulationRun]:
        """评测阶段：测试新任务的完成能力"""
        print(f"  Evaluation Phase: {len(stage.eval_tasks)} tasks")

        eval_runs = []
        for task_id in stage.eval_tasks:
            task = self.curriculum.get_task(task_id)
            for trial in range(stage.num_eval_trials):
                run = self._execute_task(task, is_learning=False)
                eval_runs.append(run)

        return eval_runs

    def _run_retention_phase(self, stage: LearningStage) -> list[SimulationRun]:
        """保持阶段：测试旧任务的能力是否退化"""
        print(f"  Retention Phase: {len(stage.retention_tasks)} tasks")

        retention_runs = []
        for task_id in stage.retention_tasks:
            task = self.curriculum.get_task(task_id)
            for trial in range(stage.num_eval_trials):
                run = self._execute_task(task, is_learning=False)
                retention_runs.append(run)

        return retention_runs

    def _record_tool_performance(self, stage, eval_results, retention_results):
        """记录每个工具在当前阶段的表现"""
        all_runs = eval_results + (retention_results or [])

        # 按工具统计
        tool_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for run in all_runs:
            for tool_call in self._extract_tool_calls(run):
                tool_name = tool_call.name
                is_correct = self._check_tool_call_correctness(tool_call, run)
                tool_stats[tool_name]["total"] += 1
                if is_correct:
                    tool_stats[tool_name]["correct"] += 1

        # 记录到历史
        for tool_name, stats in tool_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            self.tool_performance_history[tool_name].append((stage.stage_id, accuracy))
```

---

## 第三部分：评估指标设计

### 3.1 基础指标（与 τ²-Bench 一致）

| 指标 | 公式 | 说明 |
|------|------|------|
| **Average Reward** | $\frac{1}{N}\sum_{i=1}^{N} r_i$ | 所有任务的平均奖励 |
| **Pass@k** | $\frac{C(s,k)}{C(n,k)}$ | k 次尝试中至少成功一次的概率 |

### 3.2 持续学习核心指标（改进版）

#### 3.2.1 前向迁移 (Forward Transfer) - **修正版**

**问题**：原始定义 `FWT = mean(new_tool_success_rate)` 只是绝对性能，不是迁移增益。

**修正定义**：

$$FWT_i = Acc(T_{new}^i | \text{Stage}_i) - Acc(T_{new}^i | \text{Baseline})$$

其中 Baseline 可以是：
- **Zero-doc Baseline**：不提供新工具文档，只有 schema
- **Random Baseline**：随机工具选择
- **Frozen Baseline**：使用 Stage 0 的模型直接测试

```python
def compute_forward_transfer(
    stage_results: list[StageResult],
    baseline_results: list[StageResult]  # 使用 baseline agent 的结果
) -> dict[str, float]:
    """
    计算前向迁移指标

    Returns:
        per_stage_fwt: 每阶段的 FWT
        average_fwt: 平均 FWT
    """
    fwt_per_stage = {}

    for stage_result, baseline_result in zip(stage_results, baseline_results):
        stage_id = stage_result.stage_id

        # 只考虑涉及新工具的任务
        new_tool_tasks = stage_result.get_new_tool_tasks()

        stage_acc = compute_accuracy(stage_result, new_tool_tasks)
        baseline_acc = compute_accuracy(baseline_result, new_tool_tasks)

        fwt_per_stage[stage_id] = stage_acc - baseline_acc

    return {
        "per_stage": fwt_per_stage,
        "average": np.mean(list(fwt_per_stage.values()))
    }
```

#### 3.2.2 后向迁移 / 遗忘 (Backward Transfer / Forgetting) - **工具级**

**问题**：Stage-level 的遗忘容易被任务难度变化污染。

**修正定义**：工具级遗忘

$$Forgetting(tool_j) = \max_{k \in \{0,...,i-1\}} Acc(tool_j, k) - Acc(tool_j, i)$$

$$KnowledgeRetention(tool_j, i) = \frac{Acc(tool_j, i)}{Acc(tool_j, k_{learned})}$$

```python
def compute_tool_level_forgetting(
    tool_performance_history: dict[str, list[tuple[str, float]]]
) -> dict[str, dict]:
    """
    计算工具级别的遗忘指标

    Args:
        tool_performance_history: {tool_name: [(stage_id, accuracy), ...]}

    Returns:
        per_tool_forgetting: 每个工具的遗忘率
        per_tool_retention: 每个工具的知识保持率
    """
    results = {}

    for tool_name, history in tool_performance_history.items():
        if len(history) < 2:
            continue

        # 找到该工具被学习的阶段（首次出现）
        learned_stage_idx = 0
        learned_acc = history[0][1]

        # 找到最高性能点
        max_acc = max(h[1] for h in history)
        max_stage_idx = [i for i, h in enumerate(history) if h[1] == max_acc][0]

        # 当前性能（最后一个阶段）
        final_acc = history[-1][1]

        # 计算遗忘
        forgetting = max(0, max_acc - final_acc)

        # 计算知识保持率
        retention = final_acc / learned_acc if learned_acc > 0 else 0

        results[tool_name] = {
            "forgetting": forgetting,
            "retention": retention,
            "max_accuracy": max_acc,
            "final_accuracy": final_acc,
            "performance_curve": [(h[0], h[1]) for h in history]
        }

    return results
```

#### 3.2.3 学习效率 (Learning Efficiency) - **改进版**

**问题**：原定义 `LE = 1 / (stage index where acc >= 0.5)` 过于粗糙。

**改进定义**：Area Under Learning Curve (AULC) + 归一化

$$LE = \frac{AULC}{AULC_{ideal}} \times \frac{1}{\#examples}$$

```python
def compute_learning_efficiency(
    learning_curve: list[float],  # 每个学习步骤后的准确率
    num_examples: int,
    ideal_curve: list[float] = None  # 理想学习曲线（可选）
) -> dict:
    """
    计算学习效率

    Args:
        learning_curve: 学习过程中的准确率序列
        num_examples: 学习示例数量
        ideal_curve: 理想曲线（默认为立即达到 1.0）

    Returns:
        aulc: Area Under Learning Curve
        normalized_efficiency: 归一化后的学习效率
        samples_to_threshold: 达到阈值所需样本数
    """
    # 计算 AULC
    aulc = np.trapz(learning_curve) / len(learning_curve)

    # 理想 AULC（立即达到最终性能）
    if ideal_curve is None:
        ideal_aulc = 1.0
    else:
        ideal_aulc = np.trapz(ideal_curve) / len(ideal_curve)

    # 归一化效率
    normalized_efficiency = (aulc / ideal_aulc) / num_examples if num_examples > 0 else 0

    # 达到不同阈值所需的样本数
    thresholds = [0.5, 0.7, 0.9]
    samples_to_threshold = {}
    for threshold in thresholds:
        for i, acc in enumerate(learning_curve):
            if acc >= threshold:
                samples_to_threshold[threshold] = i + 1
                break
        else:
            samples_to_threshold[threshold] = -1  # 未达到

    return {
        "aulc": aulc,
        "normalized_efficiency": normalized_efficiency,
        "samples_to_threshold": samples_to_threshold
    }
```

### 3.3 泛化性指标（新增）

#### 3.3.1 工具组合泛化 (Tool Composition Generalization)

评估 Agent 能否将学到的单个工具能力组合起来解决新问题。

$$TCG = Acc(\text{unseen tool combinations}) - Acc_{random}$$

```python
def compute_tool_composition_generalization(
    eval_results: list[SimulationRun],
    training_tool_combinations: set[tuple[str, ...]],
) -> float:
    """
    计算工具组合泛化能力

    Args:
        eval_results: 评测结果
        training_tool_combinations: 训练时见过的工具组合

    Returns:
        泛化准确率：在未见过的工具组合上的表现
    """
    unseen_results = []
    seen_results = []

    for run in eval_results:
        tool_combo = tuple(sorted(set(tc.name for tc in run.tool_calls)))

        if tool_combo in training_tool_combinations:
            seen_results.append(run)
        else:
            unseen_results.append(run)

    if not unseen_results:
        return None  # 没有未见过的组合

    unseen_acc = np.mean([r.reward_info.reward for r in unseen_results])
    seen_acc = np.mean([r.reward_info.reward for r in seen_results]) if seen_results else 0

    return {
        "unseen_accuracy": unseen_acc,
        "seen_accuracy": seen_acc,
        "generalization_gap": seen_acc - unseen_acc,
        "num_unseen_combinations": len(set(
            tuple(sorted(set(tc.name for tc in r.tool_calls)))
            for r in unseen_results
        ))
    }
```

#### 3.3.2 参数泛化 (Parameter Generalization)

评估 Agent 能否正确处理未见过的参数值。

```python
def compute_parameter_generalization(
    eval_results: list[SimulationRun],
    training_param_values: dict[str, set],  # tool_name -> seen parameter values
) -> dict:
    """
    计算参数泛化能力

    评估在未见过的参数值上的表现
    """
    results = {
        "seen_params_accuracy": [],
        "unseen_params_accuracy": [],
    }

    for run in eval_results:
        for tool_call in run.tool_calls:
            tool_name = tool_call.name

            # 检查参数值是否在训练时见过
            has_unseen_param = False
            for param_name, param_value in tool_call.arguments.items():
                key = f"{tool_name}.{param_name}"
                if key in training_param_values:
                    if str(param_value) not in training_param_values[key]:
                        has_unseen_param = True
                        break

            is_correct = check_tool_call_correctness(tool_call, run)

            if has_unseen_param:
                results["unseen_params_accuracy"].append(is_correct)
            else:
                results["seen_params_accuracy"].append(is_correct)

    return {
        "seen_params_accuracy": np.mean(results["seen_params_accuracy"]) if results["seen_params_accuracy"] else 0,
        "unseen_params_accuracy": np.mean(results["unseen_params_accuracy"]) if results["unseen_params_accuracy"] else 0,
        "generalization_gap": (
            np.mean(results["seen_params_accuracy"]) - np.mean(results["unseen_params_accuracy"])
        ) if results["unseen_params_accuracy"] else None
    }
```

#### 3.3.3 跨域泛化 (Cross-Domain Generalization)

评估在一个领域学到的工具使用模式能否迁移到其他领域。

```python
def compute_cross_domain_generalization(
    source_domain_results: dict[str, list[SimulationRun]],  # 源域结果
    target_domain_results: dict[str, list[SimulationRun]],  # 目标域结果
    tool_mapping: dict[str, str],  # 源域工具 -> 目标域相似工具
) -> dict:
    """
    计算跨域泛化能力

    Args:
        source_domain_results: 在源域（如 airline）的评测结果
        target_domain_results: 在目标域（如 retail）的评测结果
        tool_mapping: 工具映射关系，如 {'get_user_details': 'get_user_details'}
    """
    # 计算源域性能
    source_tool_acc = compute_per_tool_accuracy(source_domain_results)

    # 计算目标域性能（只看有映射关系的工具）
    target_tool_acc = compute_per_tool_accuracy(target_domain_results)

    # 计算迁移效果
    transfer_results = {}
    for source_tool, target_tool in tool_mapping.items():
        if source_tool in source_tool_acc and target_tool in target_tool_acc:
            transfer_results[f"{source_tool}->{target_tool}"] = {
                "source_accuracy": source_tool_acc[source_tool],
                "target_accuracy": target_tool_acc[target_tool],
                "transfer_gain": target_tool_acc[target_tool] - 0.5  # vs random baseline
            }

    return {
        "per_tool_transfer": transfer_results,
        "average_transfer_gain": np.mean([
            v["transfer_gain"] for v in transfer_results.values()
        ]) if transfer_results else 0
    }
```

### 3.4 工具调用细粒度指标

为了避免 `new_tool_success_rate` 过于 noisy，拆分为三元指标：

```python
@dataclass
class ToolCallEvaluation:
    """工具调用的细粒度评估"""

    # 1. 工具选择正确性：是否选择了正确的工具
    tool_selection_correct: bool

    # 2. 工具调用有效性：参数是否正确、调用是否成功
    tool_invocation_valid: bool

    # 3. 工具结果使用正确性：是否正确使用了工具返回的结果
    tool_output_used_correctly: bool

    @property
    def fully_correct(self) -> bool:
        return (
            self.tool_selection_correct and
            self.tool_invocation_valid and
            self.tool_output_used_correctly
        )

def evaluate_tool_call(
    tool_call: ToolCall,
    expected_actions: list[dict],
    run: SimulationRun
) -> ToolCallEvaluation:
    """
    对单个工具调用进行细粒度评估
    """
    # 1. 检查工具选择
    expected_tool_names = [a["name"] for a in expected_actions]
    selection_correct = tool_call.name in expected_tool_names

    # 2. 检查调用有效性
    invocation_valid = False
    if selection_correct:
        # 找到对应的期望 action
        for expected in expected_actions:
            if expected["name"] == tool_call.name:
                # 检查参数
                if "compare_args" in expected:
                    invocation_valid = compare_arguments(
                        tool_call.arguments,
                        expected["arguments"],
                        expected["compare_args"]
                    )
                else:
                    invocation_valid = tool_call.arguments == expected["arguments"]
                break

    # 3. 检查结果使用
    output_used_correctly = False
    if invocation_valid:
        # 检查后续消息是否正确使用了工具结果
        output_used_correctly = check_output_usage(tool_call, run)

    return ToolCallEvaluation(
        tool_selection_correct=selection_correct,
        tool_invocation_valid=invocation_valid,
        tool_output_used_correctly=output_used_correctly
    )
```

### 3.5 指标汇总表

| 类别 | 指标名称 | 公式/定义 | 评估目标 |
|-----|---------|----------|---------|
| **基础** | Average Reward | $\frac{1}{N}\sum r_i$ | 整体任务完成能力 |
| **基础** | Pass@k | $\frac{C(s,k)}{C(n,k)}$ | 多次尝试成功率 |
| **持续学习** | Forward Transfer (FWT) | $Acc_{new} - Acc_{baseline}$ | 新工具学习增益 |
| **持续学习** | Backward Transfer (BWT) | $Acc_{old}^{final} - Acc_{old}^{learned}$ | 旧知识保持/提升 |
| **持续学习** | Tool-level Forgetting | $\max_k Acc_k - Acc_{final}$ | 工具级遗忘程度 |
| **持续学习** | Learning Efficiency | $\frac{AULC}{\#examples}$ | 学习效率 |
| **泛化性** | Tool Composition Gen. | $Acc_{unseen\_combo}$ | 工具组合泛化 |
| **泛化性** | Parameter Gen. | $Acc_{unseen\_params}$ | 参数值泛化 |
| **泛化性** | Cross-Domain Gen. | $Acc_{target} - Acc_{random}$ | 跨域迁移能力 |
| **细粒度** | Tool Selection Acc. | 选择正确工具的比例 | 工具选择能力 |
| **细粒度** | Tool Invocation Acc. | 参数正确的比例 | 工具调用能力 |
| **细粒度** | Output Usage Acc. | 正确使用结果的比例 | 结果利用能力 |

---

## 第四部分：实现代码框架

### 4.1 目录结构

```
src/tau2/
├── continual/                          # 持续学习模块
│   ├── __init__.py
│   │
│   ├── benchmark/                      # 核心 Benchmark 框架（算法无关）
│   │   ├── __init__.py
│   │   ├── agent_interface.py         # ContinualAgent 抽象接口
│   │   ├── protocol.py                # 标准评测协议
│   │   ├── benchmark.py               # 主 Benchmark 类
│   │   └── metrics.py                 # 指标计算
│   │
│   ├── baselines/                      # 示例 Baseline 实现
│   │   ├── __init__.py
│   │   ├── vanilla_finetune.py        # Vanilla SFT Baseline
│   │   └── icl_baseline.py            # ICL Baseline
│   │
│   ├── curriculum/                     # 课程定义
│   │   ├── __init__.py
│   │   ├── curriculum.py              # 课程定义
│   │   ├── stage.py                   # 阶段定义
│   │   └── task_selector.py           # 任务选择器
│   │
│   ├── training/                       # 训练工具（可选）
│   │   ├── __init__.py
│   │   ├── vanilla_trainer.py         # Vanilla SFT 训练器
│   │   ├── icl_baseline.py            # ICL 工具类
│   │   └── data_converter.py          # 数据格式转换
│   │
│   ├── evaluation/                     # 评测工具
│   │   ├── __init__.py
│   │   ├── evaluator.py               # 主评估器
│   │   ├── metrics.py                 # 指标计算
│   │   └── tool_analysis.py           # 工具级分析
│   │
│   └── runner.py                       # 主运行器
│
├── data_model/
│   ├── continual_results.py           # 持续学习结果数据模型
│   └── ...
└── ...

data/tau2/
├── curricula/                          # 课程定义
│   ├── airline_progressive.json       # 渐进式学习课程
│   ├── airline_tool_expansion.json    # 工具扩展课程
│   └── cross_domain_transfer.json     # 跨域迁移课程
└── ...
```

### 4.2 架构设计：Benchmark 与算法分离

本框架采用**Benchmark-Algorithm 分离**的设计原则：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Framework                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ ContinualAgent  │  │ EvaluationProto │  │ MetricsComputer │ │
│  │   (Interface)   │  │     col         │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                 │
│                    ┌───────────▼───────────┐                    │
│                    │  ContinualBenchmark   │                    │
│                    │   (Main Entry Point)  │                    │
│                    └───────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ implements
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Your Algorithm                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ VanillaFinetune │  │  ICLBaseline    │  │  YourCustom     │ │
│  │     Agent       │  │     Agent       │  │     Agent       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计原则**：

1. **Benchmark 定义 WHAT**：评测什么（任务、指标、协议）
2. **Agent 定义 HOW**：如何学习和行动（你的算法）
3. **分离的好处**：
   - 公平比较：所有算法使用相同的评测协议
   - 易于扩展：添加新算法只需实现 `ContinualAgent` 接口
   - 可复现性：标准化的评测流程

### 4.3 实现自定义 Agent

要在本 Benchmark 上评测你的持续学习算法，只需实现 `ContinualAgent` 接口：

```python
from tau2.continual import ContinualAgent, Experience, AgentResponse, LearningStage

class MyCustomAgent(ContinualAgent):
    """你的持续学习算法实现"""

    def __init__(self, model_path: str, **kwargs):
        # 初始化你的模型
        self.model = load_your_model(model_path)
        self.replay_buffer = []  # 例如：经验回放
        self.fisher_info = {}    # 例如：EWC 的 Fisher 信息

    def learn(
        self,
        stage: LearningStage,
        experiences: list[Experience],
    ) -> dict[str, Any]:
        """
        实现你的持续学习算法

        Args:
            stage: 当前学习阶段，包含：
                - stage.learning_materials: 学习材料
                - stage.new_tools: 新增工具
                - stage.available_tools: 所有可用工具
            experiences: 学习经验列表

        Returns:
            学习统计信息
        """
        # 1. 提取成功的经验
        successful = [e for e in experiences if e.success]

        # 2. 应用你的持续学习算法
        # 例如：EWC、经验回放、渐进式网络等
        train_data = self.prepare_data(successful)

        # 3. 训练
        stats = self.train_with_your_algorithm(train_data)

        return stats

    def act(
        self,
        messages: list[Message],
        available_tools: list[dict],
        stage_context: Optional[str] = None,
    ) -> AgentResponse:
        """
        生成响应

        Args:
            messages: 对话历史
            available_tools: 可用工具的 schema
            stage_context: 可选的阶段上下文

        Returns:
            AgentResponse（文本或工具调用）
        """
        # 使用你的模型生成响应
        output = self.model.generate(messages, available_tools)

        if self.is_tool_call(output):
            return AgentResponse(tool_calls=self.parse_tool_calls(output))
        else:
            return AgentResponse(content=output)

    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        self.model.save(path)
        # 保存其他状态（replay buffer, fisher info 等）

    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        self.model.load(path)

    def on_stage_end(self, stage: LearningStage, metrics: dict) -> None:
        """
        阶段结束时的回调

        可用于：
        - 更新 EWC 的 Fisher 信息
        - 整理 replay buffer
        - 记录日志
        """
        # 例如：计算 Fisher 信息用于 EWC
        self.update_fisher_information(stage)

    def get_config(self) -> dict[str, Any]:
        """返回配置信息用于日志"""
        return {
            "type": "MyCustomAgent",
            "algorithm": "EWC",
            "lambda": 0.5,
            # ...
        }
```

### 4.4 运行评测

```python
from tau2.continual import ContinualBenchmark, ProtocolConfig

# 1. 加载 Benchmark
benchmark = ContinualBenchmark.from_curriculum(
    curriculum_path="data/tau2/curricula/airline_progressive.json",
    domain="airline",
)

# 2. 创建你的 Agent
agent = MyCustomAgent(model_path="path/to/model")

# 3. 运行评测
results = benchmark.evaluate(agent)

# 4. 查看结果
print(results.overall_metrics)
```

### 4.5 核心类实现

```python
# src/tau2/continual/curriculum/curriculum.py

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class LearningStage:
    """学习阶段定义"""
    stage_id: str
    stage_name: str

    # 工具配置
    available_tools: list[str]           # 该阶段可用的工具
    new_tools: list[str]                 # 本阶段新增的工具

    # 任务配置
    learning_tasks: list[str]            # 学习任务
    eval_tasks: list[str]                # 评测任务
    retention_tasks: list[str] = field(default_factory=list)  # 保持性测试任务

    # 学习材料
    learning_materials: list[dict] = field(default_factory=list)

    # 评测配置
    num_learning_trials: int = 3
    num_eval_trials: int = 4

    # 阶段门控
    min_pass_rate: float = 0.5

@dataclass
class Curriculum:
    """完整课程定义"""
    curriculum_id: str
    curriculum_name: str
    domain: str
    curriculum_type: Literal["progressive", "tool_expansion", "cross_domain"]

    stages: list[LearningStage]

    # 元信息
    total_tools: int = 0
    total_tasks: int = 0

    def __post_init__(self):
        self.total_tools = len(set(
            tool for stage in self.stages for tool in stage.available_tools
        ))
        self.total_tasks = len(set(
            task for stage in self.stages
            for task in stage.learning_tasks + stage.eval_tasks + stage.retention_tasks
        ))
```

```python
# src/tau2/continual/evaluation/metrics.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ContinualMetrics:
    """持续学习评测指标"""

    # 基础指标
    average_reward: float
    pass_at_k: dict[int, float]

    # 持续学习指标
    forward_transfer: float
    forward_transfer_per_stage: dict[str, float]

    backward_transfer: float
    tool_level_forgetting: dict[str, float]
    tool_level_retention: dict[str, float]

    learning_efficiency: float
    learning_efficiency_per_stage: dict[str, float]

    # 泛化性指标
    tool_composition_generalization: float
    parameter_generalization: float
    cross_domain_generalization: Optional[float]

    # 细粒度指标
    tool_selection_accuracy: float
    tool_invocation_accuracy: float
    tool_output_usage_accuracy: float

    def to_dict(self) -> dict:
        """转换为字典，便于保存"""
        return {
            "basic": {
                "average_reward": self.average_reward,
                "pass_at_k": self.pass_at_k,
            },
            "continual_learning": {
                "forward_transfer": self.forward_transfer,
                "backward_transfer": self.backward_transfer,
                "tool_level_forgetting": self.tool_level_forgetting,
                "learning_efficiency": self.learning_efficiency,
            },
            "generalization": {
                "tool_composition": self.tool_composition_generalization,
                "parameter": self.parameter_generalization,
                "cross_domain": self.cross_domain_generalization,
            },
            "fine_grained": {
                "tool_selection": self.tool_selection_accuracy,
                "tool_invocation": self.tool_invocation_accuracy,
                "tool_output_usage": self.tool_output_usage_accuracy,
            }
        }

    def summary(self) -> str:
        """生成摘要报告"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              Continual Learning Evaluation Summary            ║
╠══════════════════════════════════════════════════════════════╣
║ Basic Metrics                                                 ║
║   Average Reward:        {self.average_reward:>8.4f}                         ║
║   Pass@1:                {self.pass_at_k.get(1, 0):>8.4f}                         ║
║   Pass@4:                {self.pass_at_k.get(4, 0):>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Continual Learning Metrics                                    ║
║   Forward Transfer:      {self.forward_transfer:>8.4f}                         ║
║   Backward Transfer:     {self.backward_transfer:>8.4f}                         ║
║   Avg Tool Forgetting:   {np.mean(list(self.tool_level_forgetting.values())):>8.4f}                         ║
║   Learning Efficiency:   {self.learning_efficiency:>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Generalization Metrics                                        ║
║   Tool Composition:      {self.tool_composition_generalization:>8.4f}                         ║
║   Parameter Gen.:        {self.parameter_generalization:>8.4f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Fine-Grained Tool Metrics                                     ║
║   Tool Selection:        {self.tool_selection_accuracy:>8.4f}                         ║
║   Tool Invocation:       {self.tool_invocation_accuracy:>8.4f}                         ║
║   Output Usage:          {self.tool_output_usage_accuracy:>8.4f}                         ║
╚══════════════════════════════════════════════════════════════╝
        """
```

### 4.3 CLI 命令

```python
# 在 src/tau2/cli.py 中添加

@cli.command()
@click.option("--curriculum", required=True, type=click.Path(exists=True))
@click.option("--domain", required=True, type=click.Choice(["airline", "retail", "telecom"]))
@click.option("--agent-llm", required=True)
@click.option("--training-mode", type=click.Choice(["vanilla_sft", "icl", "none"]), default="icl")
@click.option("--output-dir", default="./continual_results")
@click.option("--seed", type=int, default=42)
def continual(curriculum, domain, agent_llm, training_mode, output_dir, seed):
    """
    Run continual learning benchmark.

    Example:
        tau2 continual \\
            --curriculum data/tau2/curricula/airline_progressive.json \\
            --domain airline \\
            --agent-llm gpt-4 \\
            --training-mode icl
    """
    from tau2.continual.runner import ContinualBenchmarkRunner

    runner = ContinualBenchmarkRunner(
        curriculum_path=curriculum,
        domain=domain,
        agent_llm=agent_llm,
        training_mode=training_mode,
        output_dir=output_dir,
        seed=seed
    )

    results = runner.run()

    # 打印摘要
    click.echo(results.metrics.summary())

    # 保存结果
    results.save(output_dir)
    click.echo(f"\nResults saved to: {output_dir}")
```

---

## 第五部分：实验设计

### 5.1 主实验：持续学习能力评测

**目标**：评估不同 LLM 在工具使用持续学习任务上的表现

**实验设置**：

| 设置项 | 值 |
|-------|---|
| 领域 | Airline |
| 课程 | Progressive (4 stages) |
| 训练模式 | ICL (baseline), Vanilla SFT |
| 评测模型 | GPT-4, GPT-3.5, Claude-3, Llama-3 |
| 每任务尝试次数 | 4 |
| 随机种子 | 42, 43, 44 (3 runs) |

**预期结果表格**：

| Model | Avg Reward | FWT | BWT | Forgetting | Tool Composition Gen. |
|-------|------------|-----|-----|------------|----------------------|
| GPT-4 (ICL) | - | - | - | - | - |
| GPT-4 (SFT) | - | - | - | - | - |
| GPT-3.5 (ICL) | - | - | - | - | - |
| Claude-3 (ICL) | - | - | - | - | - |
| Llama-3 (ICL) | - | - | - | - | - |

### 5.2 消融实验

#### 实验 A1: 学习材料的影响

| 条件 | 描述 |
|-----|------|
| Full | 完整学习材料（文档 + 示例 + 演示） |
| Doc-only | 仅工具文档 |
| Example-only | 仅使用示例 |
| Zero-doc | 仅有 schema，无任何说明 |

#### 实验 A2: 阶段划分粒度的影响

| 条件 | 阶段数 | 每阶段新增工具数 |
|-----|-------|---------------|
| Fine-grained | 6 | 1-2 |
| Medium | 4 | 2-3 |
| Coarse | 2 | 5-6 |
| All-at-once | 1 | 12 |

#### 实验 A3: 保持任务数量的影响

| 条件 | 每阶段 retention_tasks 数量 |
|-----|---------------------------|
| No retention | 0 |
| Light | 1 |
| Medium | 3 |
| Heavy | 5 |

### 5.3 跨域迁移实验

**设置**：
1. 在 Airline 域完成 4 阶段学习
2. 直接在 Retail 域评测（zero-shot transfer）
3. 在 Retail 域进行 1 阶段适应学习后再评测

**工具映射**：

| Airline | Retail | 功能 |
|---------|--------|------|
| get_user_details | get_user_details | 用户查询 |
| get_reservation_details | get_order_details | 订单查询 |
| cancel_reservation | cancel_pending_order | 取消操作 |
| book_reservation | - | 无直接对应 |

---

## 第六部分：预期贡献与局限性

### 6.1 预期贡献

1. **首个 Agent 工具使用持续学习 Benchmark**
   - 填补了动态工具环境下 Agent 评测的空白

2. **细粒度的工具级评测指标**
   - Tool-level Forgetting
   - 三元工具调用评估

3. **完整的泛化性评测框架**
   - 工具组合泛化
   - 参数泛化
   - 跨域泛化

4. **可复现的实验框架**
   - 基于成熟的 τ²-Bench 基础设施
   - 提供 Vanilla SFT 和 ICL baseline

### 6.2 局限性与未来工作

1. **当前局限**：
   - 仅支持 τ²-Bench 的三个领域
   - 未考虑多轮对话中的工具演进
   - 训练方式较为基础（无专门的持续学习算法）

2. **未来工作**：
   - 引入 EWC、ER 等持续学习算法
   - 扩展到更多领域和工具类型
   - 研究工具 schema 变化的适应机制

---

## 附录 A：课程配置示例

```json
{
  "curriculum_id": "airline_progressive_v1",
  "curriculum_name": "Airline Progressive Learning",
  "domain": "airline",
  "curriculum_type": "progressive",

  "stages": [
    {
      "stage_id": "stage_0_foundation",
      "stage_name": "Foundation - Basic Queries",
      "available_tools": ["get_user_details", "get_reservation_details", "list_all_airports"],
      "new_tools": ["get_user_details", "get_reservation_details", "list_all_airports"],
      "learning_tasks": ["task_1", "task_5", "task_6"],
      "eval_tasks": ["task_0", "task_3", "task_4"],
      "retention_tasks": [],
      "learning_materials": [
        {
          "type": "documentation",
          "tool": "get_user_details",
          "content": "获取用户详情，包括预订列表、支付方式等..."
        }
      ],
      "num_learning_trials": 3,
      "num_eval_trials": 4,
      "min_pass_rate": 0.5
    },
    {
      "stage_id": "stage_1_search",
      "stage_name": "Search Expansion",
      "available_tools": ["get_user_details", "get_reservation_details", "list_all_airports", "search_direct_flight", "search_onestop_flight"],
      "new_tools": ["search_direct_flight", "search_onestop_flight"],
      "learning_tasks": ["task_8", "task_9"],
      "eval_tasks": ["task_7", "task_10", "task_11"],
      "retention_tasks": ["task_0", "task_1", "task_3"],
      "learning_materials": [
        {
          "type": "documentation",
          "tool": "search_direct_flight",
          "content": "搜索直飞航班..."
        },
        {
          "type": "example",
          "tool": "search_direct_flight",
          "content": "User: 查询明天JFK到LAX的航班\nAgent: [search_direct_flight(...)]"
        }
      ],
      "num_learning_trials": 3,
      "num_eval_trials": 4,
      "min_pass_rate": 0.5
    }
  ]
}
```

---

## 附录 B：评测结果示例

```json
{
  "curriculum_id": "airline_progressive_v1",
  "agent": "gpt-4",
  "training_mode": "icl",
  "timestamp": "2024-01-15T10:30:00Z",

  "stage_results": [
    {
      "stage_id": "stage_0_foundation",
      "eval_reward": 0.85,
      "retention_reward": null,
      "new_tool_performance": {
        "get_user_details": {"selection": 0.95, "invocation": 0.90, "usage": 0.88},
        "get_reservation_details": {"selection": 0.92, "invocation": 0.88, "usage": 0.85}
      }
    },
    {
      "stage_id": "stage_1_search",
      "eval_reward": 0.78,
      "retention_reward": 0.82,
      "new_tool_performance": {
        "search_direct_flight": {"selection": 0.85, "invocation": 0.80, "usage": 0.75}
      }
    }
  ],

  "metrics": {
    "basic": {
      "average_reward": 0.72,
      "pass_at_k": {"1": 0.65, "2": 0.78, "4": 0.88}
    },
    "continual_learning": {
      "forward_transfer": 0.15,
      "backward_transfer": -0.03,
      "tool_level_forgetting": {
        "get_user_details": 0.02,
        "get_reservation_details": 0.05
      },
      "learning_efficiency": 0.42
    },
    "generalization": {
      "tool_composition": 0.68,
      "parameter": 0.72,
      "cross_domain": null
    }
  }
}
```
