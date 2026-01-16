import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Optional, List
from dataclasses import dataclass, field
from copy import deepcopy

from tau2.data_model.message import Message, AssistantMessage, ToolCall
from tau2.continual.benchmark.agent_interface import (
    ContinualAgent,
    Experience,
    AgentResponse,
)
from tau2.continual.curriculum.stage import LearningStage


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model config
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"

    # GRPO hyperparameters
    learning_rate: float = 1e-6
    beta: float = 0.1  # KL penalty coefficient
    group_size: int = 4  # Number of samples per prompt for GRPO
    max_length: int = 2048
    max_new_tokens: int = 512

    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Generation
    temperature: float = 0.7
    do_sample: bool = True

    # Online learning settings
    update_after_each_experience: bool = True  # Key: update immediately
    min_reward_for_update: float = 0.0  # Only update on successful experiences

    # Checkpointing
    save_every_n_updates: int = 10


class GRPOContinualAgent(ContinualAgent):
    """
    Continual learning agent using GRPO for online policy optimization.

    This agent:
    - Updates parameters after EACH experience (online learning)
    - Uses GRPO for policy optimization
    - Performs full parameter updates
    - Supports sequential domain training
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        """
        Initialize the GRPO continual agent.

        Args:
            config: GRPO configuration
        """
        self.config = config or GRPOConfig()

        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.ref_model = None  # Reference model for KL penalty

        # Training state
        self.current_stage_id = None
        self.total_updates = 0
        self.training_history = []
        self.stage_stats = {}

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the model, tokenizer, and setup optimizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, "auto")

        print(f"[GRPOContinualAgent] Loading model: {self.config.model_name_or_path}")

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
            device_map=self.config.device if self.config.device != "auto" else "auto",
            trust_remote_code=True,
        )

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
        print("[GRPOContinualAgent] Creating reference model for KL penalty...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=self.config.device if self.config.device != "auto" else "auto",
            trust_remote_code=True,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print(f"[GRPOContinualAgent] Model loaded successfully")

    def learn(
        self,
        stage: LearningStage,
        experiences: List[Experience],
    ) -> dict[str, Any]:
        """
        Learn from experiences using GRPO.

        IMPORTANT: This method is called with ALL experiences from the learning phase.
        However, we process each experience individually for online learning.

        For true online learning (update after each task execution),
        use learn_single_experience() instead.
        """
        self.current_stage_id = stage.stage_id

        stats = {
            "stage_id": stage.stage_id,
            "num_experiences": len(experiences),
            "num_updates": 0,
            "total_loss": 0.0,
            "successful_experiences": 0,
        }

        # Process each experience individually (online learning)
        for exp in experiences:
            if exp.success or exp.reward >= self.config.min_reward_for_update:
                update_stats = self._grpo_update_single(exp, stage)
                stats["num_updates"] += 1
                stats["total_loss"] += update_stats.get("loss", 0.0)
                stats["successful_experiences"] += 1

        if stats["num_updates"] > 0:
            stats["avg_loss"] = stats["total_loss"] / stats["num_updates"]
        else:
            stats["avg_loss"] = 0.0

        self.training_history.append(stats)
        self.stage_stats[stage.stage_id] = stats

        return stats

    def learn_single_experience(
        self,
        experience: Experience,
        stage: LearningStage,
    ) -> dict[str, Any]:
        """
        Learn from a single experience immediately after execution.

        This is the key method for online learning - call this right after
        each task execution to update the model immediately.

        Args:
            experience: Single experience from task execution
            stage: Current learning stage

        Returns:
            Update statistics
        """
        if not experience.success and experience.reward < self.config.min_reward_for_update:
            return {"status": "skipped", "reason": "unsuccessful_experience"}

        return self._grpo_update_single(experience, stage)

    def _grpo_update_single(
        self,
        experience: Experience,
        stage: LearningStage,
    ) -> dict[str, Any]:
        """
        Perform a single GRPO update from one experience.

        GRPO (Group Relative Policy Optimization) works by:
        1. Generating multiple responses for the same prompt
        2. Computing rewards for each response
        3. Using relative rewards within the group for policy gradient
        """
        self.model.train()

        # Extract prompt and response from experience
        prompt_messages, response = self._extract_prompt_response(experience)

        if prompt_messages is None or response is None:
            return {"status": "skipped", "reason": "no_valid_response"}

        # Format prompt
        prompt_text = self._format_prompt(prompt_messages, stage)
        response_text = self._format_response(response)

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        full_ids = self.tokenizer.encode(
            prompt_text + response_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )

        prompt_ids = prompt_ids.to(self.model.device)
        full_ids = full_ids.to(self.model.device)

        # Generate group of responses for GRPO
        group_responses, group_log_probs = self._generate_group(
            prompt_ids,
            self.config.group_size
        )

        # Compute rewards for each response in the group
        # For now, use the actual experience reward as the target
        # and estimate rewards for generated responses based on similarity
        group_rewards = self._compute_group_rewards(
            group_responses,
            response_text,
            experience.reward,
        )

        # Compute GRPO loss
        loss = self._compute_grpo_loss(
            prompt_ids,
            full_ids,
            group_responses,
            group_log_probs,
            group_rewards,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # Update
        self.optimizer.step()
        self.total_updates += 1

        # Checkpoint if needed
        if self.total_updates % self.config.save_every_n_updates == 0:
            self._auto_checkpoint()

        return {
            "status": "updated",
            "loss": loss.item(),
            "total_updates": self.total_updates,
            "group_rewards_mean": sum(group_rewards) / len(group_rewards),
        }

    def _extract_prompt_response(
        self,
        experience: Experience
    ) -> tuple[Optional[List[Message]], Optional[AssistantMessage]]:
        """Extract the last prompt-response pair from experience."""
        messages = experience.messages

        # Find the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AssistantMessage):
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return None, None

        # Prompt is everything before the last assistant message
        prompt_messages = messages[:last_assistant_idx]
        response = messages[last_assistant_idx]

        return prompt_messages, response

    def _format_prompt(
        self,
        messages: List[Message],
        stage: LearningStage
    ) -> str:
        """Format messages as prompt text."""
        formatted = []

        # Add stage context
        stage_context = stage.get_learning_prompt_addition()
        if stage_context:
            formatted.append(f"Context:\n{stage_context}\n")

        # Format messages
        for msg in messages:
            role = msg.role.capitalize()
            content = getattr(msg, 'content', '') or ''
            if content:
                formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _format_response(self, msg: AssistantMessage) -> str:
        """Format assistant message as response text."""
        if msg.content:
            return f"\nAssistant: {msg.content}"
        elif msg.tool_calls:
            tool_calls_json = json.dumps({
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            })
            return f"\nAssistant: {tool_calls_json}"
        return ""

    def _generate_group(
        self,
        prompt_ids: torch.Tensor,
        group_size: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate a group of responses for GRPO."""
        responses = []
        log_probs_list = []

        for _ in range(group_size):
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

            # Extract generated tokens (excluding prompt)
            generated_ids = outputs.sequences[0, prompt_ids.shape[1]:]
            responses.append(generated_ids)

            # Compute log probabilities
            if outputs.scores:
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    probs = F.softmax(score[0], dim=-1)
                    token_id = generated_ids[i] if i < len(generated_ids) else self.tokenizer.eos_token_id
                    log_prob = torch.log(probs[token_id] + 1e-10)
                    log_probs.append(log_prob)
                log_probs_list.append(torch.stack(log_probs) if log_probs else torch.tensor([0.0]))
            else:
                log_probs_list.append(torch.tensor([0.0]))

        return responses, log_probs_list

    def _compute_group_rewards(
        self,
        group_responses: List[torch.Tensor],
        target_response: str,
        actual_reward: float,
    ) -> List[float]:
        """
        Compute rewards for each response in the group.

        Uses a combination of:
        1. Similarity to the successful response
        2. The actual task reward
        """
        rewards = []
        target_tokens = set(self.tokenizer.encode(target_response))

        for response_ids in group_responses:
            response_tokens = set(response_ids.tolist())

            # Jaccard similarity
            if target_tokens or response_tokens:
                similarity = len(target_tokens & response_tokens) / len(target_tokens | response_tokens)
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
        group_responses: List[torch.Tensor],
        group_log_probs: List[torch.Tensor],
        group_rewards: List[float],
    ) -> torch.Tensor:
        """
        Compute GRPO loss.

        GRPO uses relative rewards within the group:
        L = -E[A(s,a) * log π(a|s)] + β * KL(π || π_ref)

        where A(s,a) is the advantage computed from relative rewards.
        """
        # Compute advantages (relative rewards)
        mean_reward = sum(group_rewards) / len(group_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_reward = max(std_reward, 1e-8)  # Avoid division by zero

        advantages = [(r - mean_reward) / std_reward for r in group_rewards]

        # Compute policy gradient loss
        pg_loss = torch.tensor(0.0, device=self.model.device)

        for log_probs, advantage in zip(group_log_probs, advantages):
            log_probs = log_probs.to(self.model.device)
            pg_loss -= advantage * log_probs.sum()

        pg_loss /= len(group_responses)

        # Compute KL penalty with reference model
        kl_loss = self._compute_kl_penalty(prompt_ids, full_ids)

        # Total loss
        total_loss = pg_loss + self.config.beta * kl_loss

        return total_loss

    def _compute_kl_penalty(
        self,
        prompt_ids: torch.Tensor,
        full_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty between current and reference model."""
        # Get logits from both models
        with torch.no_grad():
            ref_outputs = self.ref_model(full_ids)
            ref_logits = ref_outputs.logits

        current_outputs = self.model(full_ids)
        current_logits = current_outputs.logits

        # Only compute KL for response tokens (after prompt)
        prompt_len = prompt_ids.shape[1]

        if full_ids.shape[1] <= prompt_len:
            return torch.tensor(0.0, device=self.model.device)

        ref_logits = ref_logits[:, prompt_len-1:-1, :]
        current_logits = current_logits[:, prompt_len-1:-1, :]

        # Compute KL divergence
        ref_probs = F.softmax(ref_logits, dim=-1)
        current_log_probs = F.log_softmax(current_logits, dim=-1)

        kl = (ref_probs * (torch.log(ref_probs + 1e-10) - current_log_probs)).sum(dim=-1)

        return kl.mean()

    def _auto_checkpoint(self):
        """Automatically save checkpoint during training."""
        checkpoint_dir = Path(f"./checkpoints/grpo_update_{self.total_updates}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "total_updates": self.total_updates,
            "training_history": self.training_history,
            "stage_stats": self.stage_stats,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"[GRPOContinualAgent] Auto-checkpoint saved at update {self.total_updates}")

    def act(
        self,
        messages: List[Message],
        available_tools: list[dict],
        stage_context: Optional[str] = None,
    ) -> AgentResponse:
        """Generate response using the trained model."""
        self.model.eval()

        # Format input
        formatted = []
        if stage_context:
            formatted.append(f"Context:\n{stage_context}\n")

        for msg in messages:
            role = msg.role.capitalize()
            content = getattr(msg, 'content', '') or ''
            if content:
                formatted.append(f"{role}: {content}")

        # Add tool information
        if available_tools:
            tools_desc = json.dumps(available_tools, indent=2)
            formatted.append(f"\nAvailable tools:\n{tools_desc}")

        formatted.append("\nAssistant:")
        input_text = "\n".join(formatted)

        # Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse response
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> AgentResponse:
        """Parse model output into AgentResponse."""
        import re

        response_text = response_text.strip()

        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())

                if "tool_calls" in data:
                    tool_calls = []
                    for tc in data["tool_calls"]:
                        tool_calls.append(ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=tc["name"],
                            arguments=tc.get("arguments", {}),
                        ))
                    return AgentResponse(tool_calls=tool_calls)
                elif "name" in data and "arguments" in data:
                    # Single tool call
                    return AgentResponse(tool_calls=[
                        ToolCall(
                            id="call_0",
                            name=data["name"],
                            arguments=data.get("arguments", {}),
                        )
                    ])
            except json.JSONDecodeError:
                pass

        # Default: text response
        return AgentResponse(content=response_text)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )

        # Save training state
        state = {
            "total_updates": self.total_updates,
            "training_history": self.training_history,
            "stage_stats": self.stage_stats,
            "config": {
                "model_name_or_path": self.config.model_name_or_path,
                "learning_rate": self.config.learning_rate,
                "beta": self.config.beta,
                "group_size": self.config.group_size,
            }
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"[GRPOContinualAgent] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_dir = Path(path)

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # Move to device
        self.model.to(self.config.device)

        # Recreate optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Load optimizer state if exists
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            self.total_updates = state.get("total_updates", 0)
            self.training_history = state.get("training_history", [])
            self.stage_stats = state.get("stage_stats", {})

        print(f"[GRPOContinualAgent] Checkpoint loaded from {path}")

    def on_stage_start(self, stage: LearningStage) -> None:
        """Called at stage start."""
        print(f"[GRPOContinualAgent] Starting stage: {stage.stage_name}")
        print(f"  Domain: {stage.stage_id}")
        print(f"  New tools: {stage.new_tools}")
        self.current_stage_id = stage.stage_id

    def on_stage_end(self, stage: LearningStage, metrics: dict) -> None:
        """Called at stage end."""
        print(f"[GRPOContinualAgent] Completed stage: {stage.stage_name}")
        print(f"  Total updates this stage: {self.stage_stats.get(stage.stage_id, {}).get('num_updates', 0)}")
        print(f"  Total updates overall: {self.total_updates}")

        eval_reward = metrics.get("evaluation", {}).get("average_reward", 0)
        print(f"  Eval reward: {eval_reward:.4f}")

    def get_config(self) -> dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "GRPOContinualAgent",
            "model": self.config.model_name_or_path,
            "learning_rate": self.config.learning_rate,
            "beta": self.config.beta,
            "group_size": self.config.group_size,
            "max_length": self.config.max_length,
            "update_after_each_experience": self.config.update_after_each_experience,
        }

    def update_reference_model(self):
        """
        Update the reference model to current model.

        Call this at the end of each domain/stage to prevent
        the KL penalty from becoming too large.
        """
        print("[GRPOContinualAgent] Updating reference model...")
        self.ref_model.load_state_dict(self.model.state_dict())
        print("[GRPOContinualAgent] Reference model updated")
