"""
Agent Interface for Continual Learning Benchmark

This module defines the abstract interface that all continual learning
agents must implement to be evaluated by the benchmark.

The benchmark framework is SEPARATE from the algorithms:
- Benchmark: Defines WHAT to evaluate (tasks, metrics, protocol)
- Agent: Defines HOW to learn and act (your algorithm)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from tau2.data_model.message import Message, AssistantMessage, ToolCall
from tau2.continual.curriculum.stage import LearningStage


@dataclass
class Experience:
    """
    A single learning experience from a task execution.

    This is what the benchmark provides to your agent for learning.
    """
    task_id: str
    messages: list[Message]  # Full conversation history
    tool_calls: list[ToolCall]  # Tool calls made by the agent
    tool_results: list[str]  # Results from tool calls
    reward: float  # Final reward (0.0 or 1.0)
    success: bool  # Whether the task was successful

    # Additional context
    expected_actions: list[dict] = field(default_factory=list)
    stage_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    """
    Response from the agent.

    Either contains text content OR tool calls, not both.
    """
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def to_message(self) -> AssistantMessage:
        """Convert to AssistantMessage."""
        return AssistantMessage(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls,
        )

    @property
    def is_tool_call(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


class ContinualAgent(ABC):
    """
    Abstract interface for continual learning agents.

    To evaluate your algorithm on this benchmark, implement this interface.

    The benchmark will:
    1. Call `on_stage_start()` at the beginning of each stage
    2. Call `learn()` with experiences from learning tasks
    3. Call `act()` during evaluation to get agent responses
    4. Call `on_stage_end()` at the end of each stage
    5. Repeat for all stages

    Your implementation should:
    - Maintain model state across stages
    - Implement your continual learning algorithm in `learn()`
    - Generate responses in `act()`
    """

    @abstractmethod
    def learn(
        self,
        stage: LearningStage,
        experiences: list[Experience],
    ) -> dict[str, Any]:
        """
        Learn from experiences in a stage.

        This is where you implement your continual learning algorithm.
        The benchmark calls this after collecting experiences from learning tasks.

        Args:
            stage: Current learning stage with:
                - stage.learning_materials: Documentation and examples
                - stage.new_tools: Newly introduced tools
                - stage.available_tools: All available tools
            experiences: List of Experience objects from learning tasks

        Returns:
            Dictionary with learning statistics, e.g.:
            {
                "loss": 0.5,
                "num_updates": 100,
                "examples_seen": 50,
            }

        Example implementation:
            def learn(self, stage, experiences):
                # Extract successful experiences
                successful = [e for e in experiences if e.success]

                # Convert to training format
                train_data = self.prepare_training_data(successful)

                # Apply your continual learning algorithm
                stats = self.train_with_ewc(train_data)

                return stats
        """
        pass

    @abstractmethod
    def act(
        self,
        messages: list[Message],
        available_tools: list[dict],
        stage_context: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a response given conversation history and available tools.

        This is called during evaluation. Your agent should:
        - Analyze the conversation history
        - Decide whether to respond with text or make a tool call
        - Return an appropriate AgentResponse

        Args:
            messages: Conversation history (system, user, assistant, tool messages)
            available_tools: List of tool schemas in OpenAI function format:
                [
                    {
                        "name": "get_user_details",
                        "description": "...",
                        "parameters": {...}
                    },
                    ...
                ]
            stage_context: Optional additional context from learning materials

        Returns:
            AgentResponse with either:
            - content: Text response to the user
            - tool_calls: List of ToolCall objects

        Example implementation:
            def act(self, messages, available_tools, stage_context=None):
                # Format input for your model
                prompt = self.format_prompt(messages, available_tools)

                # Generate response
                output = self.model.generate(prompt)

                # Parse output
                if self.is_tool_call(output):
                    tool_calls = self.parse_tool_calls(output)
                    return AgentResponse(tool_calls=tool_calls)
                else:
                    return AgentResponse(content=output)
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.

        Called by the benchmark to save agent state between stages
        or for later analysis.

        Args:
            path: Directory path to save checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Directory path to load checkpoint from
        """
        pass

    def on_stage_start(self, stage: LearningStage) -> None:
        """
        Called at the beginning of each stage.

        Override to perform stage-specific initialization, e.g.:
        - Update tool schemas
        - Prepare for new tools
        - Reset stage-specific state

        Args:
            stage: The stage that is about to start
        """
        pass

    def on_stage_end(self, stage: LearningStage, metrics: dict) -> None:
        """
        Called at the end of each stage.

        Override to perform stage-specific cleanup or logging, e.g.:
        - Log metrics
        - Update EWC Fisher information
        - Consolidate replay buffer

        Args:
            stage: The stage that just ended
            metrics: Evaluation metrics for this stage
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Return agent configuration for logging.

        Override to include your algorithm's hyperparameters.
        """
        return {}


class DummyAgent(ContinualAgent):
    """
    A dummy agent for testing the benchmark framework.

    Always returns a fixed response. Useful for debugging.
    """

    def learn(self, stage, experiences):
        return {"status": "dummy_agent_no_learning"}

    def act(self, messages, available_tools, stage_context=None):
        return AgentResponse(content="I am a dummy agent.")

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass
