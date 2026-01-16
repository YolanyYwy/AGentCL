"""
ICL (In-Context Learning) Baseline Agent

This is an EXAMPLE baseline that uses in-context learning
without any parameter updates.

This serves as an upper bound for what can be achieved
without actual learning/fine-tuning.
"""

import json
from pathlib import Path
from typing import Any, Optional, List

from tau2.data_model.message import Message, AssistantMessage, UserMessage, ToolMessage, ToolCall
from tau2.continual.benchmark.agent_interface import (
    ContinualAgent,
    Experience,
    AgentResponse,
)
from tau2.continual.curriculum.stage import LearningStage


class ICLBaselineAgent(ContinualAgent):
    """
    In-Context Learning baseline without parameter updates.

    This baseline:
    - Accumulates examples in the prompt
    - Uses learning materials as context
    - Does NOT update model parameters
    - Demonstrates ICL capabilities

    This is a TEMPLATE - actual inference requires an LLM API.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_examples_per_stage: int = 3,
        max_total_examples: int = 10,
        include_tool_results: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the ICL baseline agent.

        Args:
            model_name: Name of the LLM to use
            max_examples_per_stage: Maximum examples to keep per stage
            max_total_examples: Maximum total examples across all stages
            include_tool_results: Whether to include tool results in examples
            api_key: API key for the LLM provider
        """
        self.model_name = model_name
        self.max_examples_per_stage = max_examples_per_stage
        self.max_total_examples = max_total_examples
        self.include_tool_results = include_tool_results
        self.api_key = api_key

        # Store examples per stage
        self.stage_examples: dict[str, list[dict]] = {}

        # Store learning materials per stage
        self.stage_materials: dict[str, str] = {}

        # Current stage tracking
        self.current_stage_id: Optional[str] = None

        # Learning history
        self.learning_history: list[dict] = []

    def learn(
        self,
        stage: LearningStage,
        experiences: List[Experience],
    ) -> dict[str, Any]:
        """
        Learn from experiences by storing them for in-context use.

        ICL doesn't update parameters - it just stores examples.
        """
        self.current_stage_id = stage.stage_id

        # Store learning materials
        self.stage_materials[stage.stage_id] = stage.get_learning_prompt_addition()

        # Filter successful experiences
        successful = [e for e in experiences if e.success]

        if not successful:
            return {
                "status": "no_successful_experiences",
                "num_experiences": len(experiences),
            }

        # Extract and store examples
        examples = self._extract_examples(successful)
        self.stage_examples[stage.stage_id] = examples[:self.max_examples_per_stage]

        stats = {
            "stage_id": stage.stage_id,
            "num_experiences": len(experiences),
            "num_successful": len(successful),
            "num_examples_stored": len(self.stage_examples[stage.stage_id]),
            "total_examples": sum(len(e) for e in self.stage_examples.values()),
        }

        self.learning_history.append(stats)
        return stats

    def _extract_examples(self, experiences: List[Experience]) -> list[dict]:
        """Extract few-shot examples from experiences."""
        examples = []

        for exp in experiences:
            conversation = self._format_conversation(exp.messages)
            if conversation:
                examples.append({
                    "task_id": exp.task_id,
                    "conversation": conversation,
                    "tools_used": self._extract_tools_used(exp.messages),
                })

        return examples

    def _format_conversation(self, messages: List[Message]) -> list[dict]:
        """Format messages into a conversation structure."""
        conversation = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                if msg.content:
                    conversation.append({
                        "role": "user",
                        "content": msg.content,
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
                    content = msg.content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    conversation.append({
                        "role": "tool",
                        "content": content,
                    })

        return conversation

    def _extract_tools_used(self, messages: List[Message]) -> list[str]:
        """Extract list of tools used in the conversation."""
        tools = set()
        for msg in messages:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools.add(tc.name)
        return list(tools)

    def _build_context(self, stage_context: Optional[str] = None) -> str:
        """Build the full context including materials and examples."""
        sections = []

        # Add learning materials from all stages
        for stage_id, materials in self.stage_materials.items():
            if materials:
                sections.append(f"## Knowledge from Stage: {stage_id}\n{materials}")

        # Add stage-specific context if provided
        if stage_context:
            sections.append(f"## Current Context\n{stage_context}")

        # Add few-shot examples
        examples = self._get_examples_for_prompt()
        if examples:
            sections.append("\n## Example Interactions\n")
            for i, example in enumerate(examples, 1):
                sections.append(f"### Example {i}")
                sections.append(self._format_example_for_prompt(example))

        return "\n\n".join(sections)

    def _get_examples_for_prompt(self) -> list[dict]:
        """Get examples for the prompt, respecting max_total_examples."""
        all_examples = []

        for stage_id in self.stage_examples.keys():
            examples = self.stage_examples.get(stage_id, [])
            all_examples.extend(examples)

        if len(all_examples) > self.max_total_examples:
            # Prioritize recent examples
            stages = list(self.stage_examples.keys())
            selected = []
            remaining_slots = self.max_total_examples

            # First pass: one from each stage
            for stage_id in reversed(stages):
                examples = self.stage_examples.get(stage_id, [])
                if examples and remaining_slots > 0:
                    selected.append(examples[0])
                    remaining_slots -= 1

            # Second pass: fill remaining slots
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

    def act(
        self,
        messages: List[Message],
        available_tools: list[dict],
        stage_context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate response using in-context learning.
        """
        # Build context with examples
        context = self._build_context(stage_context)

        # Format input
        input_text = self._format_input(messages, context)

        # TEMPLATE: Replace with actual LLM call
        # Example with OpenAI:
        #
        # from openai import OpenAI
        # client = OpenAI(api_key=self.api_key)
        #
        # response = client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[
        #         {"role": "system", "content": context},
        #         *[{"role": m.role, "content": m.content} for m in messages],
        #     ],
        #     tools=[{"type": "function", "function": t} for t in available_tools],
        # )
        #
        # choice = response.choices[0]
        # if choice.message.tool_calls:
        #     tool_calls = [
        #         ToolCall(
        #             id=tc.id,
        #             name=tc.function.name,
        #             arguments=json.loads(tc.function.arguments),
        #         )
        #         for tc in choice.message.tool_calls
        #     ]
        #     return AgentResponse(tool_calls=tool_calls)
        # else:
        #     return AgentResponse(content=choice.message.content)

        # Placeholder response
        return AgentResponse(
            content="[ICLBaselineAgent] Generation placeholder. Implement act() method."
        )

    def _format_input(
        self,
        messages: List[Message],
        context: str,
    ) -> str:
        """Format messages as input text."""
        parts = []

        if context:
            parts.append(f"Context:\n{context}\n")

        for msg in messages:
            role = msg.role.capitalize()
            content = getattr(msg, 'content', '') or ''
            if content:
                parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def save_checkpoint(self, path: str) -> None:
        """Save agent state."""
        Path(path).mkdir(parents=True, exist_ok=True)

        state = {
            "stage_examples": self.stage_examples,
            "stage_materials": self.stage_materials,
            "current_stage_id": self.current_stage_id,
            "learning_history": self.learning_history,
            "config": {
                "model_name": self.model_name,
                "max_examples_per_stage": self.max_examples_per_stage,
                "max_total_examples": self.max_total_examples,
                "include_tool_results": self.include_tool_results,
            }
        }

        state_path = Path(path) / "icl_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        print(f"[ICLBaselineAgent] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent state."""
        state_path = Path(path) / "icl_state.json"

        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)

            self.stage_examples = state.get("stage_examples", {})
            self.stage_materials = state.get("stage_materials", {})
            self.current_stage_id = state.get("current_stage_id")
            self.learning_history = state.get("learning_history", [])

            config = state.get("config", {})
            self.model_name = config.get("model_name", self.model_name)
            self.max_examples_per_stage = config.get("max_examples_per_stage", self.max_examples_per_stage)
            self.max_total_examples = config.get("max_total_examples", self.max_total_examples)
            self.include_tool_results = config.get("include_tool_results", self.include_tool_results)

        print(f"[ICLBaselineAgent] Checkpoint loaded from {path}")

    def on_stage_start(self, stage: LearningStage) -> None:
        """Called at stage start."""
        print(f"[ICLBaselineAgent] Starting stage: {stage.stage_name}")
        print(f"  New tools: {stage.new_tools}")
        print(f"  Current examples: {sum(len(e) for e in self.stage_examples.values())}")

    def on_stage_end(self, stage: LearningStage, metrics: dict) -> None:
        """Called at stage end."""
        print(f"[ICLBaselineAgent] Completed stage: {stage.stage_name}")
        eval_reward = metrics.get("evaluation", {}).get("average_reward", 0)
        print(f"  Eval reward: {eval_reward:.4f}")
        print(f"  Total examples stored: {sum(len(e) for e in self.stage_examples.values())}")

    def get_config(self) -> dict[str, Any]:
        """Return agent configuration."""
        return {
            "type": "ICLBaselineAgent",
            "model": self.model_name,
            "max_examples_per_stage": self.max_examples_per_stage,
            "max_total_examples": self.max_total_examples,
            "include_tool_results": self.include_tool_results,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the agent's state."""
        return {
            "num_stages": len(self.stage_materials),
            "stages": {
                stage_id: {
                    "num_examples": len(self.stage_examples.get(stage_id, [])),
                    "has_materials": bool(self.stage_materials.get(stage_id)),
                }
                for stage_id in self.stage_materials.keys()
            },
            "total_examples": sum(len(e) for e in self.stage_examples.values()),
        }
