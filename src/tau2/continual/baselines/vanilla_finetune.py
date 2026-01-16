"""
Vanilla Fine-tuning Baseline

This is an EXAMPLE baseline that performs standard fine-tuning
without any continual learning techniques.

This serves as a lower bound to measure catastrophic forgetting.
"""

import json
from pathlib import Path
from typing import Any, Optional, List

from tau2.data_model.message import Message, AssistantMessage, ToolCall
from tau2.continual.benchmark.agent_interface import (
    ContinualAgent,
    Experience,
    AgentResponse,
)
from tau2.continual.curriculum.stage import LearningStage


class VanillaFinetuneAgent(ContinualAgent):
    """
    Vanilla fine-tuning baseline without continual learning techniques.

    This baseline:
    - Fine-tunes on each stage's data
    - Does NOT use replay buffer
    - Does NOT use regularization (EWC, etc.)
    - Demonstrates maximum forgetting

    This is a TEMPLATE - actual training requires a deep learning framework.
    """

    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 2048,
        device: str = "cuda",
    ):
        """
        Initialize the vanilla fine-tuning agent.

        Args:
            model_name_or_path: HuggingFace model name or local path
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of training epochs per stage
            batch_size: Training batch size
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        self.model_name_or_path = model_name_or_path
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        # Model and tokenizer (to be loaded)
        self.model = None
        self.tokenizer = None

        # Training state
        self.current_stage_id = None
        self.training_history = []

        # Load model
        self._load_model()

    def _load_model(self):
        """
        Load the model and tokenizer.

        Override this method to use your preferred framework.
        """
        # TEMPLATE: Replace with actual model loading
        # Example with HuggingFace:
        #
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        # self.model.to(self.device)

        print(f"[VanillaFinetuneAgent] Model loading placeholder: {self.model_name_or_path}")
        print("  Replace _load_model() with actual implementation")

    def learn(
        self,
        stage: LearningStage,
        experiences: List[Experience],
    ) -> dict[str, Any]:
        """
        Fine-tune on the stage's experiences.

        This is vanilla fine-tuning - no replay, no regularization.
        """
        self.current_stage_id = stage.stage_id

        # Filter successful experiences
        successful = [e for e in experiences if e.success]

        if not successful:
            return {
                "status": "no_successful_experiences",
                "num_experiences": len(experiences),
            }

        # Convert to training format
        train_data = self._prepare_training_data(successful, stage)

        # TEMPLATE: Replace with actual training
        # Example with HuggingFace:
        #
        # from transformers import Trainer, TrainingArguments
        # training_args = TrainingArguments(
        #     output_dir=f"./checkpoints/{stage.stage_id}",
        #     learning_rate=self.learning_rate,
        #     num_train_epochs=self.num_epochs,
        #     per_device_train_batch_size=self.batch_size,
        # )
        # trainer = Trainer(
        #     model=self.model,
        #     args=training_args,
        #     train_dataset=train_data,
        # )
        # trainer.train()

        print(f"[VanillaFinetuneAgent] Training placeholder for stage {stage.stage_id}")
        print(f"  Would train on {len(train_data)} examples")
        print(f"  Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}")

        stats = {
            "stage_id": stage.stage_id,
            "num_experiences": len(experiences),
            "num_successful": len(successful),
            "num_train_examples": len(train_data),
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
        }

        self.training_history.append(stats)
        return stats

    def _prepare_training_data(
        self,
        experiences: List[Experience],
        stage: LearningStage,
    ) -> list[dict]:
        """
        Convert experiences to training format.

        Returns list of {"input": str, "output": str} pairs.
        """
        train_data = []

        # Get learning materials for context
        stage_context = stage.get_learning_prompt_addition()

        for exp in experiences:
            # Extract training examples from conversation
            for i, msg in enumerate(exp.messages):
                if isinstance(msg, AssistantMessage):
                    # Build input from previous messages
                    input_text = self._format_input(
                        exp.messages[:i],
                        stage_context,
                    )

                    # Format output
                    output_text = self._format_output(msg)

                    if input_text and output_text:
                        train_data.append({
                            "input": input_text,
                            "output": output_text,
                        })

        return train_data

    def _format_input(
        self,
        messages: List[Message],
        stage_context: str,
    ) -> str:
        """Format messages as input text."""
        parts = []

        if stage_context:
            parts.append(f"Context:\n{stage_context}\n")

        for msg in messages:
            role = msg.role.capitalize()
            content = getattr(msg, 'content', '') or ''
            if content:
                parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _format_output(self, msg: AssistantMessage) -> str:
        """Format assistant message as output text."""
        if msg.content:
            return msg.content
        elif msg.tool_calls:
            return json.dumps([
                {"name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ])
        return ""

    def act(
        self,
        messages: List[Message],
        available_tools: list[dict],
        stage_context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate response using the fine-tuned model.
        """
        # Format input
        input_text = self._format_input(messages, stage_context or "")

        # TEMPLATE: Replace with actual generation
        # Example:
        #
        # inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs, max_new_tokens=512)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #
        # # Parse response
        # if self._is_tool_call(response):
        #     tool_calls = self._parse_tool_calls(response)
        #     return AgentResponse(tool_calls=tool_calls)
        # else:
        #     return AgentResponse(content=response)

        # Placeholder response
        return AgentResponse(
            content="[VanillaFinetuneAgent] Generation placeholder. Implement act() method."
        )

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).mkdir(parents=True, exist_ok=True)

        # TEMPLATE: Save model
        # self.model.save_pretrained(path)
        # self.tokenizer.save_pretrained(path)

        # Save training history
        history_path = Path(path) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"[VanillaFinetuneAgent] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        # TEMPLATE: Load model
        # self.model = AutoModelForCausalLM.from_pretrained(path)
        # self.tokenizer = AutoTokenizer.from_pretrained(path)

        # Load training history
        history_path = Path(path) / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.training_history = json.load(f)

        print(f"[VanillaFinetuneAgent] Checkpoint loaded from {path}")

    def on_stage_start(self, stage: LearningStage) -> None:
        """Called at stage start."""
        print(f"[VanillaFinetuneAgent] Starting stage: {stage.stage_name}")
        print(f"  New tools: {stage.new_tools}")

    def on_stage_end(self, stage: LearningStage, metrics: dict) -> None:
        """Called at stage end."""
        print(f"[VanillaFinetuneAgent] Completed stage: {stage.stage_name}")
        eval_reward = metrics.get("evaluation", {}).get("average_reward", 0)
        print(f"  Eval reward: {eval_reward:.4f}")

    def get_config(self) -> dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "VanillaFinetuneAgent",
            "model": self.model_name_or_path,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
        }
