"""
ICL Continual Baseline

This module provides an In-Context Learning baseline for continual
learning evaluation. It uses few-shot examples in the prompt without
any parameter updates.
"""

import json
from typing import Any, Optional

from tau2.data_model.message import Message, AssistantMessage, UserMessage, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.continual.curriculum.stage import LearningStage, LearningMaterial


class ICLContinualBaseline:
    """
    In-Context Learning baseline for continual learning.

    This baseline learns by accumulating examples in the prompt
    without any parameter updates. It tests the ICL capabilities
    of LLMs in a continual learning setting.
    """

    def __init__(
        self,
        max_examples_per_stage: int = 3,
        max_total_examples: int = 10,
        include_tool_results: bool = True,
    ):
        """
        Initialize the ICL baseline.

        Args:
            max_examples_per_stage: Maximum examples to keep per stage
            max_total_examples: Maximum total examples across all stages
            include_tool_results: Whether to include tool results in examples
        """
        self.max_examples_per_stage = max_examples_per_stage
        self.max_total_examples = max_total_examples
        self.include_tool_results = include_tool_results

        # Store examples per stage
        self.stage_examples: dict[str, list[dict]] = {}

        # Store learning materials per stage
        self.stage_materials: dict[str, list[LearningMaterial]] = {}

        # Current stage tracking
        self.current_stage: Optional[str] = None

    def learn_stage(
        self,
        stage: LearningStage,
        successful_runs: Optional[list[SimulationRun]] = None,
    ) -> None:
        """
        Learn from a stage by extracting examples and materials.

        Args:
            stage: The learning stage
            successful_runs: Optional successful runs to extract examples from
        """
        stage_id = stage.stage_id
        self.current_stage = stage_id

        # Store learning materials
        self.stage_materials[stage_id] = stage.learning_materials

        # Extract examples from successful runs
        if successful_runs:
            examples = self._extract_examples(successful_runs)
            # Keep only the best examples (up to max_examples_per_stage)
            self.stage_examples[stage_id] = examples[:self.max_examples_per_stage]

    def _extract_examples(
        self,
        runs: list[SimulationRun]
    ) -> list[dict]:
        """
        Extract few-shot examples from simulation runs.

        Args:
            runs: List of simulation runs (preferably successful ones)

        Returns:
            List of example dictionaries
        """
        examples = []

        for run in runs:
            # Only use successful runs
            if run.reward_info and run.reward_info.reward < 1.0:
                continue

            # Extract conversation turns
            conversation = self._format_conversation(run.messages)
            if conversation:
                examples.append({
                    "task_id": run.task_id,
                    "conversation": conversation,
                    "tools_used": self._extract_tools_used(run.messages),
                })

        return examples

    def _format_conversation(self, messages: list[Message]) -> list[dict]:
        """Format messages into a conversation structure."""
        conversation = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                if msg.content:
                    conversation.append({
                        "role": "user",
                        "content": msg.content,
                    })
                elif msg.tool_calls:
                    # User tool calls (in some domains)
                    conversation.append({
                        "role": "user",
                        "content": f"[User performs action: {msg.tool_calls[0].name}]",
                    })
            elif isinstance(msg, AssistantMessage):
                if msg.content:
                    conversation.append({
                        "role": "assistant",
                        "content": msg.content,
                    })
                elif msg.tool_calls:
                    tool_call_str = json.dumps([
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in msg.tool_calls
                    ], indent=2)
                    conversation.append({
                        "role": "assistant",
                        "content": f"[Tool Call]\n{tool_call_str}",
                    })
            elif isinstance(msg, ToolMessage) and self.include_tool_results:
                if msg.content:
                    # Truncate long tool results
                    content = msg.content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    conversation.append({
                        "role": "tool",
                        "content": content,
                    })

        return conversation

    def _extract_tools_used(self, messages: list[Message]) -> list[str]:
        """Extract list of tools used in the conversation."""
        tools = set()
        for msg in messages:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools.add(tc.name)
        return list(tools)

    def build_prompt_addition(
        self,
        up_to_stage: Optional[str] = None,
    ) -> str:
        """
        Build the prompt addition containing learning materials and examples.

        Args:
            up_to_stage: Include materials up to this stage (inclusive).
                        If None, includes all stages.

        Returns:
            String to add to the system prompt
        """
        sections = []

        # Determine which stages to include
        if up_to_stage is None:
            stages_to_include = list(self.stage_materials.keys())
        else:
            stages_to_include = []
            for stage_id in self.stage_materials.keys():
                stages_to_include.append(stage_id)
                if stage_id == up_to_stage:
                    break

        # Add learning materials
        for stage_id in stages_to_include:
            materials = self.stage_materials.get(stage_id, [])
            for material in materials:
                sections.append(material.to_prompt_section())

        # Add few-shot examples
        examples = self._get_examples_for_prompt(stages_to_include)
        if examples:
            sections.append("\n## Example Interactions\n")
            for i, example in enumerate(examples, 1):
                sections.append(f"### Example {i}")
                sections.append(self._format_example_for_prompt(example))

        return "\n\n".join(sections)

    def _get_examples_for_prompt(
        self,
        stages: list[str]
    ) -> list[dict]:
        """
        Get examples for the prompt, respecting max_total_examples.

        Prioritizes more recent stages while keeping some from earlier stages.
        """
        all_examples = []

        # Collect examples from all stages
        for stage_id in stages:
            examples = self.stage_examples.get(stage_id, [])
            all_examples.extend(examples)

        # If we have too many, prioritize recent ones
        if len(all_examples) > self.max_total_examples:
            # Keep at least one from each stage if possible
            selected = []
            remaining_slots = self.max_total_examples

            # First pass: one from each stage
            for stage_id in reversed(stages):
                examples = self.stage_examples.get(stage_id, [])
                if examples and remaining_slots > 0:
                    selected.append(examples[0])
                    remaining_slots -= 1

            # Second pass: fill remaining slots from recent stages
            for stage_id in reversed(stages):
                examples = self.stage_examples.get(stage_id, [])
                for ex in examples[1:]:
                    if remaining_slots <= 0:
                        break
                    if ex not in selected:
                        selected.append(ex)
                        remaining_slots -= 1

            return selected

        return all_examples

    def _format_example_for_prompt(self, example: dict) -> str:
        """Format a single example for inclusion in the prompt."""
        lines = []

        if "tools_used" in example and example["tools_used"]:
            lines.append(f"Tools used: {', '.join(example['tools_used'])}")

        lines.append("")

        for turn in example.get("conversation", []):
            role = turn["role"].capitalize()
            content = turn["content"]
            lines.append(f"{role}: {content}")
            lines.append("")

        return "\n".join(lines)

    def get_stage_summary(self) -> dict[str, Any]:
        """Get a summary of learned stages."""
        return {
            "num_stages": len(self.stage_materials),
            "stages": {
                stage_id: {
                    "num_materials": len(materials),
                    "num_examples": len(self.stage_examples.get(stage_id, [])),
                }
                for stage_id, materials in self.stage_materials.items()
            },
            "total_examples": sum(
                len(examples) for examples in self.stage_examples.values()
            ),
        }

    def clear_stage(self, stage_id: str) -> None:
        """Clear learned data for a specific stage."""
        if stage_id in self.stage_materials:
            del self.stage_materials[stage_id]
        if stage_id in self.stage_examples:
            del self.stage_examples[stage_id]

    def clear_all(self) -> None:
        """Clear all learned data."""
        self.stage_materials.clear()
        self.stage_examples.clear()
        self.current_stage = None

    def save_state(self, path: str) -> None:
        """Save the current state to a file."""
        state = {
            "stage_materials": {
                stage_id: [m.model_dump() for m in materials]
                for stage_id, materials in self.stage_materials.items()
            },
            "stage_examples": self.stage_examples,
            "current_stage": self.current_stage,
            "config": {
                "max_examples_per_stage": self.max_examples_per_stage,
                "max_total_examples": self.max_total_examples,
                "include_tool_results": self.include_tool_results,
            }
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "ICLContinualBaseline":
        """Load state from a file."""
        with open(path, "r") as f:
            state = json.load(f)

        config = state.get("config", {})
        baseline = cls(
            max_examples_per_stage=config.get("max_examples_per_stage", 3),
            max_total_examples=config.get("max_total_examples", 10),
            include_tool_results=config.get("include_tool_results", True),
        )

        # Restore materials
        for stage_id, materials_data in state.get("stage_materials", {}).items():
            baseline.stage_materials[stage_id] = [
                LearningMaterial(**m) for m in materials_data
            ]

        # Restore examples
        baseline.stage_examples = state.get("stage_examples", {})
        baseline.current_stage = state.get("current_stage")

        return baseline
