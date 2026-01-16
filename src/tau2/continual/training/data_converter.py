"""
Data Converter

This module provides utilities for converting simulation data
to various training formats (SFT, chat, etc.).
"""

import json
from typing import Any, Optional

from tau2.data_model.message import (
    Message,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ToolMessage,
    MultiToolMessage,
    ToolCall,
)
from tau2.data_model.simulation import SimulationRun


class DataConverter:
    """
    Converts simulation runs to various training data formats.

    Supports conversion to:
    - SFT format (input/output pairs)
    - Chat format (conversation turns)
    - Tool-use format (with tool schemas)
    """

    def __init__(
        self,
        include_system_prompt: bool = True,
        include_tool_results: bool = True,
        max_context_length: int = 8192,
    ):
        """
        Initialize the data converter.

        Args:
            include_system_prompt: Whether to include system prompts
            include_tool_results: Whether to include tool results in context
            max_context_length: Maximum context length in tokens (approximate)
        """
        self.include_system_prompt = include_system_prompt
        self.include_tool_results = include_tool_results
        self.max_context_length = max_context_length

    def convert_to_sft_format(
        self,
        runs: list[SimulationRun],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Convert simulation runs to SFT training format.

        Each assistant message becomes a training example with
        the conversation history as input and the assistant's
        response as output.

        Args:
            runs: List of simulation runs
            system_prompt: Optional system prompt to prepend

        Returns:
            List of {"input": str, "output": str} dictionaries
        """
        examples = []

        for run in runs:
            # Only use successful runs for training
            if run.reward_info and run.reward_info.reward < 1.0:
                continue

            messages = run.messages
            for i, msg in enumerate(messages):
                if isinstance(msg, AssistantMessage):
                    # Build context from previous messages
                    context = self._build_context(
                        messages[:i],
                        system_prompt=system_prompt
                    )

                    # Format the assistant's response
                    output = self._format_assistant_message(msg)

                    if context and output:
                        examples.append({
                            "input": context,
                            "output": output,
                        })

        return examples

    def convert_to_chat_format(
        self,
        runs: list[SimulationRun],
        system_prompt: Optional[str] = None,
    ) -> list[list[dict[str, str]]]:
        """
        Convert simulation runs to chat format.

        Each run becomes a conversation with role/content pairs.

        Args:
            runs: List of simulation runs
            system_prompt: Optional system prompt

        Returns:
            List of conversations, each a list of {"role": str, "content": str}
        """
        conversations = []

        for run in runs:
            conversation = []

            # Add system prompt if provided
            if system_prompt and self.include_system_prompt:
                conversation.append({
                    "role": "system",
                    "content": system_prompt,
                })

            for msg in run.messages:
                formatted = self._format_message_for_chat(msg)
                if formatted:
                    conversation.append(formatted)

            if conversation:
                conversations.append(conversation)

        return conversations

    def convert_to_tool_use_format(
        self,
        runs: list[SimulationRun],
        tool_schemas: list[dict],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Convert simulation runs to tool-use training format.

        Includes tool schemas and formats tool calls appropriately.

        Args:
            runs: List of simulation runs
            tool_schemas: List of tool schemas (OpenAI function format)
            system_prompt: Optional system prompt

        Returns:
            List of training examples with tool information
        """
        examples = []

        for run in runs:
            if run.reward_info and run.reward_info.reward < 1.0:
                continue

            messages = run.messages
            for i, msg in enumerate(messages):
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    context = self._build_context(
                        messages[:i],
                        system_prompt=system_prompt
                    )

                    example = {
                        "input": context,
                        "tools": tool_schemas,
                        "tool_calls": [
                            {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                    examples.append(example)

        return examples

    def _build_context(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build context string from messages."""
        parts = []

        if system_prompt and self.include_system_prompt:
            parts.append(f"System: {system_prompt}")

        for msg in messages:
            formatted = self._format_message_for_context(msg)
            if formatted:
                parts.append(formatted)

        return "\n\n".join(parts)

    def _format_message_for_context(self, msg: Message) -> Optional[str]:
        """Format a message for inclusion in context."""
        if isinstance(msg, SystemMessage):
            if self.include_system_prompt and msg.content:
                return f"System: {msg.content}"
        elif isinstance(msg, UserMessage):
            if msg.content:
                return f"User: {msg.content}"
            elif msg.tool_calls:
                # User tool calls (in some domains)
                calls = [f"{tc.name}({json.dumps(tc.arguments)})" for tc in msg.tool_calls]
                return f"User Action: {'; '.join(calls)}"
        elif isinstance(msg, AssistantMessage):
            if msg.content:
                return f"Assistant: {msg.content}"
            elif msg.tool_calls:
                calls = [f"{tc.name}({json.dumps(tc.arguments)})" for tc in msg.tool_calls]
                return f"Assistant Action: {'; '.join(calls)}"
        elif isinstance(msg, ToolMessage):
            if self.include_tool_results and msg.content:
                return f"Tool Result: {msg.content}"
        elif isinstance(msg, MultiToolMessage):
            if self.include_tool_results:
                results = [tm.content for tm in msg.tool_messages if tm.content]
                if results:
                    return f"Tool Results: {'; '.join(results)}"

        return None

    def _format_assistant_message(self, msg: AssistantMessage) -> str:
        """Format an assistant message as training output."""
        if msg.content:
            return msg.content
        elif msg.tool_calls:
            # Format tool calls as structured output
            calls = []
            for tc in msg.tool_calls:
                calls.append({
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            return json.dumps(calls, indent=2)
        return ""

    def _format_message_for_chat(self, msg: Message) -> Optional[dict[str, str]]:
        """Format a message for chat format."""
        if isinstance(msg, SystemMessage):
            if msg.content:
                return {"role": "system", "content": msg.content}
        elif isinstance(msg, UserMessage):
            if msg.content:
                return {"role": "user", "content": msg.content}
        elif isinstance(msg, AssistantMessage):
            if msg.content:
                return {"role": "assistant", "content": msg.content}
            elif msg.tool_calls:
                # Format tool calls
                content = json.dumps([
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ])
                return {"role": "assistant", "content": content}
        elif isinstance(msg, ToolMessage):
            if msg.content:
                return {"role": "tool", "content": msg.content}

        return None

    def extract_tool_calls_from_runs(
        self,
        runs: list[SimulationRun]
    ) -> list[dict[str, Any]]:
        """
        Extract all tool calls from simulation runs.

        Returns:
            List of tool call information with context
        """
        tool_calls = []

        for run in runs:
            for i, msg in enumerate(run.messages):
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "task_id": run.task_id,
                            "message_index": i,
                            "tool_name": tc.name,
                            "arguments": tc.arguments,
                            "success": run.reward_info.reward > 0 if run.reward_info else False,
                        })

        return tool_calls

    def get_tool_usage_statistics(
        self,
        runs: list[SimulationRun]
    ) -> dict[str, dict[str, int]]:
        """
        Get statistics on tool usage across runs.

        Returns:
            Dictionary mapping tool names to usage counts
        """
        stats: dict[str, dict[str, int]] = {}

        for run in runs:
            success = run.reward_info and run.reward_info.reward > 0

            for msg in run.messages:
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.name not in stats:
                            stats[tc.name] = {"total": 0, "successful": 0}
                        stats[tc.name]["total"] += 1
                        if success:
                            stats[tc.name]["successful"] += 1

        return stats
