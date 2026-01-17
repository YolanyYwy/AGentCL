"""
OpenAI API User Simulator

Uses OpenAI API instead of local HuggingFace models to save resources.
Supports custom base_url for API relay services.
"""

from typing import Optional, Tuple

from loguru import logger

from tau2.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.user.user_simulator import UserSimulator
from tau2.user.base import UserState, ValidUserInputMessage
from tau2.utils.llm_utils import generate


class OpenAIUserSimulator(UserSimulator):
    """
    User simulator that uses OpenAI API.

    This class uses OpenAI API instead of local models to save GPU resources.
    Supports custom base_url for API relay services (e.g., api.lingleap.com).

    The API calls are handled by the existing generate() function which uses litellm.
    """

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        llm: str = "gpt-4o-mini",  # Default to cost-effective model
        llm_args: Optional[dict] = None,
        api_base: Optional[str] = None,  # Custom base URL for API relay
        api_key: Optional[str] = None,   # Custom API key
    ):
        """
        Initialize OpenAI User Simulator.

        Args:
            tools: Optional list of tools
            instructions: User instructions
            llm: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o", "gpt-5")
            llm_args: Additional args for API calls (temperature, max_tokens, etc.)
            api_base: Custom base URL for API relay (e.g., "https://api.lingleap.com/v1")
            api_key: Custom API key for the relay service
        """
        # Set default llm_args if not provided
        if llm_args is None:
            llm_args = {
                "temperature": 0.7,
                "max_tokens": 1024,
            }

        # Add custom API configuration to llm_args if provided
        # litellm supports api_base and api_key parameters
        if api_base is not None:
            llm_args["api_base"] = api_base
            logger.info(f"Using custom API base: {api_base}")

        if api_key is not None:
            llm_args["api_key"] = api_key
            logger.info("Using custom API key")

        super().__init__(
            tools=tools,
            instructions=instructions,
            llm=llm,
            llm_args=llm_args
        )

        logger.info(f"Initialized OpenAI User Simulator with model: {llm}")

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """
        Generate next message using OpenAI API.

        This method uses the parent class implementation which calls
        the generate() function with OpenAI API.
        """
        # Update state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()

        # Generate response using OpenAI API
        # The api_base and api_key are passed through llm_args
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            **self.llm_args,
        )

        user_response = assistant_message.content
        logger.debug(f"[OpenAIUserSimulator] Response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        # Flip the requestor of the tool calls
        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )

        # Update state with response
        state.messages.append(user_message)
        return user_message, state

    def set_seed(self, seed: int):
        """
        Set seed for reproducibility.

        Note: OpenAI API doesn't support deterministic seeding,
        but we can set the seed parameter if the API supports it in the future.
        """
        logger.debug(f"Seed setting requested: {seed} (OpenAI API may not support deterministic seeding)")
        # Store seed in llm_args in case API supports it
        self.llm_args["seed"] = seed

