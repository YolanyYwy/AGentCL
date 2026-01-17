"""
HuggingFace Local Model Agent

This module provides an agent that uses locally loaded HuggingFace models
instead of calling remote APIs.
"""

import json
import re
from copy import deepcopy
from typing import List, Optional, Any, Union

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.agent.hf_model_cache import HFModelCache


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<tools>
{tools_description}
</tools>
""".strip()

TOOL_CALL_FORMAT = """
When you need to call a tool, respond with JSON in this exact format:
```json
{{"tool_calls": [{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}]}}
```

When you want to send a message to the user, respond with:
```json
{{"content": "Your message here"}}
```
""".strip()


class HFAgentState(BaseModel):
    """The state of the HuggingFace agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class HFAgent(LocalAgent[HFAgentState]):
    """
    A HuggingFace agent that uses locally loaded models.
    """

    # Class-level model cache to avoid reloading
    _model_cache: dict = {}
    _tokenizer_cache: dict = {}

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        model_name_or_path: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,  # Increased from 512 to allow longer responses
        temperature: float = 0.7,
        do_sample: bool = True,
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the HuggingFace Agent.

        Args:
            tools: List of available tools
            domain_policy: The domain policy string
            model_name_or_path: HuggingFace model name or local path
            device: Device to use ('auto', 'cuda', 'cpu')
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16', 'float32')
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer (independent instance for Agent)."""
        # Agent gets its own model instance (parameters can be updated)
        self.model, self.tokenizer = HFModelCache.get_or_load_model(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            shared=False,  # Agent gets independent instance (can be trained)
        )

    def _format_tools_description(self) -> str:
        """Format tools for the prompt."""
        tools_desc = []
        for tool in self.tools:
            # Use the openai_schema which has all the info we need
            schema = tool.openai_schema
            tool_info = schema.get("function", {})
            tools_desc.append(json.dumps(tool_info, indent=2))
        return "\n\n".join(tools_desc)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
            tools_description=self._format_tools_description(),
        ) + "\n\n" + TOOL_CALL_FORMAT

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> HFAgentState:
        """Get the initial state of the agent."""
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return HFAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def _format_messages_for_model(
        self, messages: list[APICompatibleMessage]
    ) -> list[dict]:
        """Format messages for the model's chat template."""
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                formatted.append({"role": "user", "content": msg.content or ""})
            elif isinstance(msg, AssistantMessage):
                if msg.content:
                    formatted.append({"role": "assistant", "content": msg.content})
                elif msg.tool_calls:
                    # Format tool calls as JSON
                    tool_calls_json = json.dumps({
                        "tool_calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in msg.tool_calls
                        ]
                    })
                    formatted.append({"role": "assistant", "content": tool_calls_json})
            else:
                # Tool message
                content = getattr(msg, 'content', str(msg))
                formatted.append({"role": "user", "content": f"Tool result: {content}"})
        return formatted

    def _parse_response(self, response_text: str) -> AssistantMessage:
        """Parse the model's response into an AssistantMessage."""
        response_text = response_text.strip()

        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        # Try to parse as JSON
        try:
            data = json.loads(response_text)

            if "tool_calls" in data:
                tool_calls = []
                for tc in data["tool_calls"]:
                    tool_calls.append(ToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    ))
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls,
                )
            elif "content" in data:
                return AssistantMessage(
                    role="assistant",
                    content=data["content"],
                    tool_calls=None,
                )
        except json.JSONDecodeError:
            pass

        # Try to find tool call pattern in text
        tool_call_match = re.search(
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}',
            response_text
        )
        if tool_call_match:
            try:
                name = tool_call_match.group(1)
                arguments = json.loads(tool_call_match.group(2))
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_0", name=name, arguments=arguments)],
                )
            except json.JSONDecodeError:
                pass

        # Default: treat as plain text response
        return AssistantMessage(
            role="assistant",
            content=response_text,
            tool_calls=None,
        )

    def _clean_response(self, text: str) -> str:
        """
        Clean up the model response by removing think tags.

        Args:
            text: Raw model output

        Returns:
            Cleaned text
        """
        # Remove <think>...</think> tags and their content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: HFAgentState
    ) -> tuple[AssistantMessage, HFAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages
        formatted_messages = self._format_messages_for_model(messages)

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for tokenizers without chat template
            text = "\n".join([
                f"{m['role']}: {m['content']}" for m in formatted_messages
            ])
            text += "\nassistant:"

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        logger.debug(f"[HFAgent] Raw response: {response_text[:200]}...")  # Log first 200 chars

        # Clean up the response: remove <think> tags
        response_text = self._clean_response(response_text)

        logger.debug(f"[HFAgent] Cleaned response: {response_text}")

        # Parse response
        assistant_message = self._parse_response(response_text)
        state.messages.append(assistant_message)

        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for generation."""
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
