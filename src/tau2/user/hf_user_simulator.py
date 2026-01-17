"""
HuggingFace User Simulator

Uses local HuggingFace models instead of API calls.
"""

import json
import re
from typing import Optional, Tuple

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.user.user_simulator import UserSimulator
from tau2.user.base import UserState, ValidUserInputMessage
from tau2.agent.hf_model_cache import HFModelCache


class HFUserSimulator(UserSimulator):
    """
    User simulator that uses local HuggingFace models.

    This class has the same interface as UserSimulator but uses
    a local model instead of API calls.
    """

    # Class-level model cache (shared with HFAgent)
    _model_cache: dict = {}
    _tokenizer_cache: dict = {}

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,  # Not used, kept for compatibility
        llm_args: Optional[dict] = None,
        model_name_or_path: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize HF User Simulator.

        Args:
            tools: Optional list of tools
            instructions: User instructions
            llm: Not used, kept for compatibility
            llm_args: Additional args (can contain model config)
            model_name_or_path: HuggingFace model path
            device: Device to use
            torch_dtype: Torch dtype
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            trust_remote_code: Trust remote code
            load_in_8bit: Load in 8-bit
            load_in_4bit: Load in 4-bit
        """
        super().__init__(tools=tools, instructions=instructions, llm=llm, llm_args=llm_args)

        # Override with llm_args if provided
        if llm_args:
            model_name_or_path = llm_args.get("model_name_or_path", model_name_or_path)
            load_in_4bit = llm_args.get("load_in_4bit", load_in_4bit)
            load_in_8bit = llm_args.get("load_in_8bit", load_in_8bit)
            device = llm_args.get("device", device)
            torch_dtype = llm_args.get("torch_dtype", torch_dtype)
            max_new_tokens = llm_args.get("max_new_tokens", max_new_tokens)
            temperature = llm_args.get("temperature", temperature)
            do_sample = llm_args.get("do_sample", do_sample)

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
        """Load the model and tokenizer (shared read-only instance)."""
        # User uses shared cached model (read-only, parameters won't change)
        self.model, self.tokenizer = HFModelCache.get_or_load_model(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            shared=True,  # User uses shared cached model (fixed parameters)
        )

    def _format_messages_for_model(
        self, messages: list[Message]
    ) -> list[dict]:
        """Format messages for the model's chat template."""
        formatted = []
        for msg in messages:
            role = msg.role
            content = getattr(msg, 'content', None) or ""

            # Map roles
            if role == "system":
                formatted.append({"role": "system", "content": content})
            elif role == "user":
                formatted.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted.append({"role": "assistant", "content": content})
            else:
                # Tool messages
                formatted.append({"role": "user", "content": f"Tool result: {content}"})

        return formatted

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """
        Generate next message using local model.

        This overrides the parent's method to use local model instead of API.
        """
        # Update state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()

        # Format messages for model
        formatted_messages = self._format_messages_for_model(messages)

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback
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

        logger.debug(f"[HFUserSimulator] Response: {response_text}")

        # Create user message
        user_message = UserMessage(
            role="user",
            content=response_text.strip(),
            cost=0.0,  # No cost for local model
        )

        # Check for tool calls (if tools are available)
        if self.tools:
            tool_calls = self._parse_tool_calls(response_text)
            if tool_calls:
                user_message.tool_calls = tool_calls

        # Update state
        state.messages.append(user_message)
        return user_message, state

    def _parse_tool_calls(self, text: str) -> Optional[list[ToolCall]]:
        """Parse tool calls from response text."""
        # Try to find JSON tool call pattern
        tool_call_match = re.search(
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}',
            text
        )
        if tool_call_match:
            try:
                name = tool_call_match.group(1)
                arguments = json.loads(tool_call_match.group(2))
                return [ToolCall(
                    id="call_0",
                    name=name,
                    arguments=arguments,
                    requestor="user",
                )]
            except json.JSONDecodeError:
                pass

        return None

    def set_seed(self, seed: int):
        """Set the seed for generation."""
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        HFModelCache.clear_cache()
