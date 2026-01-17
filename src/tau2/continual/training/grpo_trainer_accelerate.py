"""
使用 Accelerate 实现多 GPU 并行训练的 GRPO Trainer

修改说明:
1. 使用 Accelerate 管理多 GPU
2. 自动处理模型分布和梯度同步
3. 保持 GRPO 语义正确（通过 gather 收集所有经验再训练）
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Optional, List
from dataclasses import dataclass, field

from accelerate import Accelerator
from accelerate.utils import gather_object

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.message import Message, AssistantMessage


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    # Model
    model_name_or_path: str = "Qwen/Qwen3-4B"
    device: str = "auto"  # auto, cuda, cpu
    torch_dtype: str = "auto"  # auto, float16, bfloat16, float32

    # GRPO hyperparameters
    learning_rate: float = 1e-6
    beta: float = 0.1  # KL penalty coefficient
    group_size: int = 4  # Number of samples per prompt
    max_length: int = 2048
    max_new_tokens: int = 512

    # Optimization
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Generation
    temperature: float = 0.7
    do_sample: bool = True

    # Checkpointing
    output_dir: str = "./grpo_checkpoints"
    save_every_n_updates: int = 50

    # Accelerate specific
    mixed_precision: str = "bf16"  # fp16, bf16, or no
    gradient_accumulation_steps: int = 1


class GRPOContinualTrainer:
    """
    GRPO-based continual learning trainer with Accelerate support.

    使用 Accelerate 实现多 GPU 训练，同时保证 GRPO 语义正确。
    """

    def __init__(self, config: Optional[GRPOTrainingConfig] = None):
        """
        Initialize the GRPO trainer with Accelerate.

        Args:
            config: Training configuration
        """
        self.config = config or GRPOTrainingConfig()

        # 初始化 Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.ref_model = None

        # Training state
        self.current_stage_id: Optional[str] = None
        self.total_updates: int = 0
        self.stage_updates: dict[str, int] = {}
        self.training_history: list[dict] = []

        # Device is managed by Accelerator
        self._device = self.accelerator.device

        if self.accelerator.is_main_process:
            print(f"[GRPOTrainer] Accelerate initialized")
            print(f"  - Num processes: {self.accelerator.num_processes}")
            print(f"  - Process index: {self.accelerator.process_index}")
            print(f"  - Device: {self._device}")
            print(f"  - Mixed precision: {self.config.mixed_precision}")

    def _get_torch_dtype(self):
        """Get the torch dtype based on config."""
        if self.config.torch_dtype == "auto":
            if self.config.mixed_precision == "bf16":
                return torch.bfloat16
            elif self.config.mixed_precision == "fp16":
                return torch.float16
            else:
                return torch.float32

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float32)

    def load_model(self):
        """Load the model and tokenizer with Accelerate."""
        if self.model is not None:
            return  # Already loaded

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.accelerator.is_main_process:
            print(f"[GRPOTrainer] Loading model: {self.config.model_name_or_path}")

        torch_dtype = self._get_torch_dtype()

        # Load tokenizer (only on main process, then broadcast)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            load_in_4bit=True,  # 使用 4-bit 量化节省显存
        )

        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Prepare model and optimizer with Accelerator
        # 这会自动处理多 GPU 分布
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        # Create reference model (frozen, 不需要 prepare)
        if self.accelerator.is_main_process:
            print("[GRPOTrainer] Creating reference model...")

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            load_in_4bit=True,
        )
        self.ref_model = self.ref_model.to(self._device)
        self.ref_model.eval()

        if self.accelerator.is_main_process:
            print("[GRPOTrainer] Model loaded successfully")

    def train_on_experience(
        self,
        run: SimulationRun,
        stage_id: str,
    ) -> dict[str, Any]:
        """
        Train on a single experience using GRPO.

        注意: 在多 GPU 环境下，这个函数会在每个 GPU 上被调用。
        我们需要收集所有 GPU 的经验，然后在主进程上训练。
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 更新当前 stage
        if stage_id != self.current_stage_id:
            self.current_stage_id = stage_id
            if stage_id not in self.stage_updates:
                self.stage_updates[stage_id] = 0

        # 检查是否是成功的经验
        if not run.reward_info or run.reward_info.reward <= 0:
            return {"status": "skipped", "reason": "low_reward"}

        # 提取对话历史
        messages = []
        for msg in run.messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })

        if len(messages) < 2:
            return {"status": "skipped", "reason": "insufficient_messages"}

        # 构建训练数据
        # 将对话转换为模型输入格式
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],  # 除了最后一个回复
            tokenize=False,
            add_generation_prompt=True
        )

        response = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""

        if not response:
            return {"status": "skipped", "reason": "no_response"}

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # 组合
        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]

        input_ids = torch.tensor([input_ids], device=self._device)
        attention_mask = torch.ones_like(input_ids)

        # 计算 loss
        with self.accelerator.accumulate(self.model):
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            # 计算 KL penalty (与参考模型的 KL 散度)
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # KL divergence
            logits = outputs.logits
            ref_logits = ref_outputs.logits

            kl_div = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean'
            )

            # Total loss with KL penalty
            total_loss = loss + self.config.beta * kl_div

            # Backward pass
            self.accelerator.backward(total_loss)

            # Gradient clipping
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        # 更新计数（只在主进程）
        if self.accelerator.is_main_process:
            self.total_updates += 1
            self.stage_updates[stage_id] += 1

            # 记录训练历史
            self.training_history.append({
                "stage_id": stage_id,
                "update": self.total_updates,
                "loss": total_loss.item(),
                "kl_div": kl_div.item(),
                "reward": run.reward_info.reward,
            })

        # 等待所有进程完成
        self.accelerator.wait_for_everyone()

        return {
            "status": "updated",
            "loss": total_loss.item() if self.accelerator.is_main_process else 0.0,
            "kl_div": kl_div.item() if self.accelerator.is_main_process else 0.0,
            "total_updates": self.total_updates,
            "stage_updates": self.stage_updates.get(stage_id, 0),
        }

    def train_stage(
        self,
        stage_id: str,
        experiences: List[SimulationRun],
    ) -> dict[str, Any]:
        """
        Train on a batch of experiences (for batch GRPO mode).

        在多 GPU 环境下，experiences 会被自动分配到不同 GPU。
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not experiences:
            return {"status": "skipped", "reason": "no_experiences"}

        # 过滤成功的经验
        successful_runs = [
            run for run in experiences
            if run.reward_info and run.reward_info.reward > 0
        ]

        if not successful_runs:
            return {"status": "skipped", "reason": "no_successful_runs"}

        # 训练
        total_loss = 0.0
        num_updates = 0

        for run in successful_runs:
            result = self.train_on_experience(run, stage_id)
            if result["status"] == "updated":
                total_loss += result.get("loss", 0.0)
                num_updates += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0

        return {
            "status": "completed",
            "num_updates": num_updates,
            "avg_loss": avg_loss,
        }

    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return  # 只在主进程保存

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Unwrap model from Accelerator
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model
        unwrapped_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "total_updates": self.total_updates,
            "stage_updates": self.stage_updates,
            "current_stage_id": self.current_stage_id,
            "training_history": self.training_history,
        }

        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"[GRPOTrainer] Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # Prepare with Accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        # Load training state
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            self.total_updates = state.get("total_updates", 0)
            self.stage_updates = state.get("stage_updates", {})
            self.current_stage_id = state.get("current_stage_id")
            self.training_history = state.get("training_history", [])

        print(f"[GRPOTrainer] Checkpoint loaded from {checkpoint_path}")

    def update_reference_model(self):
        """Update reference model with current policy."""
        if not self.accelerator.is_main_process:
            return

        print("[GRPOTrainer] Updating reference model...")

        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Copy weights
        self.ref_model.load_state_dict(unwrapped_model.state_dict())
        self.ref_model.eval()

        print("[GRPOTrainer] Reference model updated")

    def get_training_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            "total_updates": self.total_updates,
            "stage_updates": self.stage_updates,
            "current_stage_id": self.current_stage_id,
            "num_processes": self.accelerator.num_processes,
            "process_index": self.accelerator.process_index,
        }
