"""
Vanilla Continual Trainer

This module provides a basic continual learning trainer without
any special continual learning techniques (no replay, no regularization).
Used as a baseline for comparison.
"""

import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from tau2.data_model.simulation import SimulationRun
from tau2.continual.training.data_converter import DataConverter


@dataclass
class TrainingConfig:
    """Configuration for vanilla continual training."""
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 2048
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    logging_steps: int = 100


@dataclass
class StageCheckpoint:
    """Checkpoint information for a training stage."""
    stage_id: str
    checkpoint_path: str
    training_loss: float
    num_examples: int
    metadata: dict = field(default_factory=dict)


class VanillaContinualTrainer:
    """
    Vanilla continual learning trainer.

    This trainer performs standard SFT training on each stage's data
    without any continual learning techniques. It serves as a baseline
    to measure the impact of catastrophic forgetting.

    Note: This is a framework class. Actual training requires integration
    with a training library (e.g., transformers, trl).
    """

    def __init__(
        self,
        base_model: str,
        output_dir: str,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize the vanilla continual trainer.

        Args:
            base_model: Name or path of the base model
            output_dir: Directory to save checkpoints
            config: Training configuration
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config or TrainingConfig()

        self.data_converter = DataConverter()
        self.stage_checkpoints: dict[str, StageCheckpoint] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_stage(
        self,
        stage_id: str,
        training_runs: list[SimulationRun],
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[list[dict]] = None,
    ) -> StageCheckpoint:
        """
        Train on a single stage's data.

        This performs standard SFT without any replay or regularization.

        Args:
            stage_id: Identifier for this training stage
            training_runs: Simulation runs to train on
            system_prompt: System prompt to include in training
            tool_schemas: Tool schemas for tool-use format

        Returns:
            Checkpoint information for this stage
        """
        # Convert runs to training format
        if tool_schemas:
            training_data = self.data_converter.convert_to_tool_use_format(
                training_runs,
                tool_schemas=tool_schemas,
                system_prompt=system_prompt,
            )
        else:
            training_data = self.data_converter.convert_to_sft_format(
                training_runs,
                system_prompt=system_prompt,
            )

        if not training_data:
            raise ValueError(f"No training data generated for stage {stage_id}")

        # Save training data for reference
        data_path = self.output_dir / f"{stage_id}_training_data.json"
        with open(data_path, "w") as f:
            json.dump(training_data, f, indent=2)

        # Perform training (placeholder - actual implementation depends on training library)
        checkpoint_path = self._run_training(stage_id, training_data)

        # Create checkpoint record
        checkpoint = StageCheckpoint(
            stage_id=stage_id,
            checkpoint_path=checkpoint_path,
            training_loss=0.0,  # Would be set by actual training
            num_examples=len(training_data),
            metadata={
                "base_model": self.base_model,
                "config": self.config.__dict__,
            }
        )

        self.stage_checkpoints[stage_id] = checkpoint

        return checkpoint

    def _run_training(
        self,
        stage_id: str,
        training_data: list[dict],
    ) -> str:
        """
        Run the actual training process.

        This is a placeholder that should be implemented with a specific
        training library (e.g., transformers Trainer, trl SFTTrainer).

        Args:
            stage_id: Stage identifier
            training_data: Prepared training data

        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = str(self.output_dir / f"{stage_id}_checkpoint")

        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Load the model (from base or previous checkpoint)
        # 2. Prepare the dataset
        # 3. Run training
        # 4. Save the checkpoint

        print(f"[VanillaContinualTrainer] Training stage {stage_id}")
        print(f"  - Number of examples: {len(training_data)}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Checkpoint will be saved to: {checkpoint_path}")

        # Save a placeholder checkpoint info
        checkpoint_info = {
            "stage_id": stage_id,
            "num_examples": len(training_data),
            "config": self.config.__dict__,
            "status": "placeholder",
        }
        info_path = Path(checkpoint_path + "_info.json")
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(info_path, "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        return checkpoint_path

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the most recent checkpoint."""
        if not self.stage_checkpoints:
            return None
        latest_stage = list(self.stage_checkpoints.keys())[-1]
        return self.stage_checkpoints[latest_stage].checkpoint_path

    def get_checkpoint_for_stage(self, stage_id: str) -> Optional[str]:
        """Get the checkpoint path for a specific stage."""
        if stage_id in self.stage_checkpoints:
            return self.stage_checkpoints[stage_id].checkpoint_path
        return None

    def save_training_history(self) -> None:
        """Save the complete training history."""
        history = {
            "base_model": self.base_model,
            "config": self.config.__dict__,
            "stages": {
                stage_id: {
                    "checkpoint_path": ckpt.checkpoint_path,
                    "training_loss": ckpt.training_loss,
                    "num_examples": ckpt.num_examples,
                    "metadata": ckpt.metadata,
                }
                for stage_id, ckpt in self.stage_checkpoints.items()
            }
        }

        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    @classmethod
    def load_training_history(cls, output_dir: str) -> dict:
        """Load training history from a previous run."""
        history_path = Path(output_dir) / "training_history.json"
        if not history_path.exists():
            return {}
        with open(history_path, "r") as f:
            return json.load(f)


# Example integration with HuggingFace transformers (commented out as it requires dependencies)
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

class HFVanillaContinualTrainer(VanillaContinualTrainer):
    '''Vanilla trainer using HuggingFace transformers.'''

    def __init__(self, base_model: str, output_dir: str, config: Optional[TrainingConfig] = None):
        super().__init__(base_model, output_dir, config)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)

    def _run_training(self, stage_id: str, training_data: list[dict]) -> str:
        # Convert to HF Dataset
        dataset = Dataset.from_list(training_data)

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["input"],
                truncation=True,
                max_length=self.config.max_length,
            )

        tokenized_dataset = dataset.map(tokenize, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / stage_id),
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
        )

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()

        # Save
        checkpoint_path = str(self.output_dir / f"{stage_id}_checkpoint")
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        return checkpoint_path
"""
