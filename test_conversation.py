#!/usr/bin/env python3
"""
Test script to verify user-agent conversation works correctly.
"""

import sys
from loguru import logger

# Configure logger to show debug messages
logger.remove()
logger.add(sys.stderr, level="DEBUG")

from tau2.user.hf_user_simulator import HFUserSimulator
from tau2.agent.hf_agent import HFAgent
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.base import UserState
from tau2.agent.hf_agent import HFAgentState
from tau2.data_model.message import SystemMessage

def test_conversation():
    """Test a simple conversation between user and agent."""

    # Create user simulator
    user_instructions = """
You are a customer who wants to exchange a product.
You bought a mechanical keyboard but want to exchange it for one with RGB lighting.
"""

    user = HFUserSimulator(
        instructions=user_instructions,
        model_name_or_path="Qwen/Qwen3-4B",
        device="cuda",
        max_new_tokens=256,
        temperature=0.7,
    )

    # Create agent
    domain_policy = """
You are a customer service agent. Help customers with their requests.
Be polite and professional.
"""

    agent = HFAgent(
        tools=[],
        domain_policy=domain_policy,
        model_name_or_path="Qwen/Qwen3-4B",
        device="cuda",
        max_new_tokens=256,
        temperature=0.7,
    )

    # Initialize states
    user_state = user.get_init_state()
    agent_state = agent.get_init_state()

    # Agent starts the conversation
    print("\n" + "="*80)
    print("AGENT: Hi! How can I help you today?")
    print("="*80)

    agent_msg = AssistantMessage(
        role="assistant",
        content="Hi! How can I help you today?",
        cost=0.0
    )

    # User responds
    print("\nGenerating user response...")
    user_msg, user_state = user.generate_next_message(agent_msg, user_state)
    print("\n" + "="*80)
    print(f"USER: {user_msg.content}")
    print("="*80)

    # Agent responds
    print("\nGenerating agent response...")
    agent_msg, agent_state = agent.generate_next_message(user_msg, agent_state)
    print("\n" + "="*80)
    print(f"AGENT: {agent_msg.content}")
    print("="*80)

    print("\n✓ Conversation test completed successfully!")

if __name__ == "__main__":
    test_conversation()
