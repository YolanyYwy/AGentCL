import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Optional, List
from dataclasses import dataclass, field

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.message import Message, AssistantMessage
#from tau2.continual.training.data_converter import DataConverter


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    # Model
    model_name_or_path: str = "Qwen/Qwen3-4B-Instruct"
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


class GRPOContinualTrainer:
    """
    GRPO-based continual learning trainer.

    This trainer performs online learning with GRPO updates after each
    successful experience. It's designed as a simple baseline for
    continual learning evaluation.
    """

    def __init__(self, config: Optional[GRPOTrainingConfig] = None):
        """
        Initialize the GRPO trainer.

        Args:
            config: Training configuration
        """
        self.config = config or GRPOTrainingConfig()
        #self.data_converter = DataConverter()

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

        # Determine device
        self._device = self._get_device()

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_torch_dtype(self):
        """Get the torch dtype based on config and device."""
        if self.config.torch_dtype == "auto":
            if self._device == "cuda":
                return torch.bfloat16
            else:
                return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float32)

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return  # Already loaded

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[GRPOTrainer] Loading model: {self.config.model_name_or_path}")
        print(f"[GRPOTrainer] Device: {self._device}")

        torch_dtype = self._get_torch_dtype()

        # Load tokenizer
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
            device_map=self._device if self._device in ["cuda", "mps"] else None,
            trust_remote_code=True,
        )

        if self._device == "cpu":
            self.model = self.model.to("cpu")

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Create reference model (frozen copy for KL penalty)
        print("[GRPOTrainer] Creating reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=self._device if self._device in ["cuda", "mps"] else None,
            trust_remote_code=True,
        )
        if self._device == "cpu":
            self.ref_model = self.ref_model.to("cpu")

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print("[GRPOTrainer] Model loaded successfully")

    def train_on_experience(
        self,
        run: SimulationRun,
        stage_id: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Train on a single experience using GRPO.

        This is the core method for online learning - call this after
        each task execution to update the model immediately.

        Args:
            run: The simulation run (experience)
            stage_id: Current stage identifier
            system_prompt: Optional system prompt

        Returns:
            Training statistics
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        self.current_stage_id = stage_id

        # Skip unsuccessful runs
        if run.reward_info is None or run.reward_info.reward <= 0:
            return {"status": "skipped", "reason": "unsuccessful_run"}

        # Extract prompt and response
        prompt_text, response_text = self._extract_prompt_response(
            run.messages, system_prompt
        )

        if not prompt_text or not response_text:
            return {"status": "skipped", "reason": "no_valid_response"}

        # Perform GRPO update
        stats = self._grpo_update(prompt_text, response_text, run.reward_info.reward)

        # Update counters
        self.total_updates += 1
        self.stage_updates[stage_id] = self.stage_updates.get(stage_id, 0) + 1

        # Record history
        self.training_history.append({
            "stage_id": stage_id,
            "task_id": run.task_id,
            "update_num": self.total_updates,
            "loss": stats.get("loss", 0),
            "reward": run.reward_info.reward,
        })

        # Auto checkpoint
        if self.total_updates % self.config.save_every_n_updates == 0:
            self._save_checkpoint(f"update_{self.total_updates}")

        return stats

    def train_stage(
        self,
        stage_id: str,
        runs: list[SimulationRun],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Train on all runs from a stage.

        For online learning, this calls train_on_experience for each run.

        Args:
            stage_id: Stage identifier
            runs: List of simulation runs
            system_prompt: Optional system prompt

        Returns:
            Aggregated training statistics
        """
        stats = {
            "stage_id": stage_id,
            "num_runs": len(runs),
            "num_updates": 0,
            "total_loss": 0.0,
            "successful_runs": 0,
        }

        for run in runs:
            result = self.train_on_experience(run, stage_id, system_prompt)
            if result.get("status") != "skipped":
                stats["num_updates"] += 1
                stats["total_loss"] += result.get("loss", 0)
                stats["successful_runs"] += 1

        if stats["num_updates"] > 0:
            stats["avg_loss"] = stats["total_loss"] / stats["num_updates"]
        else:
            stats["avg_loss"] = 0.0

        return stats

    def _extract_prompt_response(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        """Extract prompt and response from messages."""
        # Find the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AssistantMessage):
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return "", ""

        # Build prompt from messages before the assistant response
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")

        for msg in messages[:last_assistant_idx]:
            role = msg.role.capitalize()
            content = getattr(msg, 'content', '') or ''
            if content:
                prompt_parts.append(f"{role}: {content}")
            elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_str = json.dumps([
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ])
                prompt_parts.append(f"{role}: [Tool Call] {tool_str}")

        prompt_text = "\n".join(prompt_parts)

        # Format response
        response_msg = messages[last_assistant_idx]
        if response_msg.content:
            response_text = response_msg.content
        elif response_msg.tool_calls:
            response_text = json.dumps({
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response_msg.tool_calls
                ]
            })
        else:
            response_text = ""

        return prompt_text, response_text

    def _grpo_update(
        self,
        prompt_text: str,
        response_text: str,
        reward: float,
    ) -> dict[str, Any]:
        """Perform a single GRPO update."""
        self.model.train()

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        full_text = prompt_text + "\nAssistant: " + response_text
        full_ids = self.tokenizer.encode(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )

        prompt_ids = prompt_ids.to(self.model.device)
        full_ids = full_ids.to(self.model.device)

        # Generate group of responses
        group_responses, group_log_probs = self._generate_group(prompt_ids)

        # Compute rewards for group
        group_rewards = self._compute_group_rewards(
            group_responses, response_text, reward
        )

        # Compute GRPO loss
        loss = self._compute_grpo_loss(
            prompt_ids, full_ids, group_log_probs, group_rewards
        )

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            "status": "updated",
            "loss": loss.item(),
            "group_reward_mean": sum(group_rewards) / len(group_rewards),
            "total_updates": self.total_updates + 1,
        }

    def _generate_group(
        self,
        prompt_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generate a group of responses for GRPO."""
        responses = []
        log_probs_list = []

        for _ in range(self.config.group_size):
            with torch.no_grad():
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences[0, prompt_ids.shape[1]:]
            responses.append(generated_ids)

            # Compute log probabilities
            if outputs.scores:
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    if i < len(generated_ids):
                        probs = F.softmax(score[0], dim=-1)
                        token_id = generated_ids[i]
                        log_prob = torch.log(probs[token_id] + 1e-10)
                        log_probs.append(log_prob)
                log_probs_list.append(
                    torch.stack(log_probs) if log_probs else torch.tensor([0.0], device=self.model.device)
                )
            else:
                log_probs_list.append(torch.tensor([0.0], device=self.model.device))

        return responses, log_probs_list

    def _compute_group_rewards(
        self,
        group_responses: list[torch.Tensor],
        target_response: str,
        actual_reward: float,
    ) -> list[float]:
        """Compute rewards for each response in the group."""
        rewards = []
        target_tokens = set(self.tokenizer.encode(target_response))

        for response_ids in group_responses:
            response_tokens = set(response_ids.tolist())

            # Jaccard similarity
            if target_tokens or response_tokens:
                intersection = len(target_tokens & response_tokens)
                union = len(target_tokens | response_tokens)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0

            # Reward = actual_reward * similarity
            reward = actual_reward * similarity
            rewards.append(reward)

        return rewards

    def _compute_grpo_loss(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
        group_log_probs: list[torch.Tensor],
        group_rewards: list[float],
    ) -> torch.Tensor:
        """Compute GRPO loss with KL penalty."""
        # Compute advantages (normalized rewards)
        mean_reward = sum(group_rewards) / len(group_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_reward = max(std_reward, 1e-8)

        advantages = [(r - mean_reward) / std_reward for r in group_rewards]

        # Policy gradient loss
        pg_loss = torch.tensor(0.0, device=self.model.device)
        for log_probs, advantage in zip(group_log_probs, advantages):
            pg_loss -= advantage * log_probs.sum()
        pg_loss /= len(group_rewards)

        # KL penalty
        kl_loss = self._compute_kl_penalty(prompt_ids, full_ids)

        # Total loss
        total_loss = pg_loss + self.config.beta * kl_loss

        return total_loss

    def _compute_kl_penalty(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        prompt_len = prompt_ids.shape[1]

        if full_ids.shape[1] <= prompt_len:
            return torch.tensor(0.0, device=self.model.device)

        with torch.no_grad():
            ref_outputs = self.ref_model(full_ids)
            ref_logits = ref_outputs.logits

        current_outputs = self.model(full_ids)
        current_logits = current_outputs.logits

        # Only compute KL for response tokens
        ref_logits = ref_logits[:, prompt_len-1:-1, :]
        current_logits = current_logits[:, prompt_len-1:-1, :]

        ref_probs = F.softmax(ref_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        kl = (ref_probs * (torch.log(ref_probs + 1e-10) - current_log_probs)).sum(dim=-1)

        return kl.mean()

    def update_reference_model(self):
        """Update reference model to current model weights."""
        if self.model is None or self.ref_model is None:
            return
        print("[GRPOTrainer] Updating reference model...")
        self.ref_model.load_state_dict(self.model.state_dict())

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        if self.model is None:
            return

        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "total_updates": self.total_updates,
            "stage_updates": self.stage_updates,
            "training_history": self.training_history[-100:],  # Keep last 100
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"[GRPOTrainer] Checkpoint saved: {checkpoint_dir}")

    def save_checkpoint(self, path: str):
        """Save checkpoint to specified path."""
        if self.model is None:
            return

        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        state = {
            "total_updates": self.total_updates,
            "stage_updates": self.stage_updates,
            "training_history": self.training_history,
            "config": {
                "model_name_or_path": self.config.model_name_or_path,
                "learning_rate": self.config.learning_rate,
                "beta": self.config.beta,
                "group_size": self.config.group_size,
            }
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"[GRPOTrainer] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint from specified path."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_dir = Path(path)

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model.to(self._device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self._device)
            )

        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            self.total_updates = state.get("total_updates", 0)
            self.stage_updates = state.get("stage_updates", {})
            self.training_history = state.get("training_history", [])

        print(f"[GRPOTrainer] Checkpoint loaded from {path}")

    def get_training_summary(self) -> dict[str, Any]:
        """Get a summary of training progress."""
        return {
            "total_updates": self.total_updates,
            "stage_updates": self.stage_updates,
            "current_stage": self.current_stage_id,
            "model_loaded": self.model is not None,
            "device": self._device,
        }
