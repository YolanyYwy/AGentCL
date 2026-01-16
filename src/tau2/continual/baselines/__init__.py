"""
Baseline implementations for continual learning.

These are EXAMPLE implementations to demonstrate how to use the benchmark.
They are NOT part of the benchmark framework itself.

Users should implement their own agents based on ContinualAgent interface.
"""

from tau2.continual.baselines.vanilla_finetune import VanillaFinetuneAgent
from tau2.continual.baselines.icl_baseline import ICLBaselineAgent
from tau2.continual.baselines.grpo_agent import GRPOContinualAgent, GRPOConfig

__all__ = [
    "VanillaFinetuneAgent",
    "ICLBaselineAgent",
    "GRPOContinualAgent",
    "GRPOConfig",
]
