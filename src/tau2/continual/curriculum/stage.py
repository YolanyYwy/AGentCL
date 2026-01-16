"""
Learning Stage Definition

This module defines the structure of a learning stage in the
continual learning curriculum.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class MaterialType(str, Enum):
    """Type of learning material."""
    DOCUMENTATION = "documentation"
    EXAMPLE = "example"
    DEMONSTRATION = "demonstration"
    FEEDBACK = "feedback"


class LearningMaterial(BaseModel):
    """Learning material for a tool or concept."""

    material_type: MaterialType = Field(
        description="Type of learning material"
    )
    content: str = Field(
        description="Content of the learning material"
    )
    tool_names: Annotated[
        list[str],
        Field(description="Tools this material relates to", default_factory=list)
    ]
    difficulty: int = Field(
        description="Difficulty level (1-5)", default=1, ge=1, le=5
    )

    # Optional metadata
    title: Optional[str] = Field(
        description="Title of the material", default=None
    )
    tags: Annotated[
        list[str],
        Field(description="Tags for categorization", default_factory=list)
    ]

    def to_prompt_section(self) -> str:
        """Convert material to a prompt section."""
        if self.material_type == MaterialType.DOCUMENTATION:
            header = f"## Tool Documentation: {', '.join(self.tool_names)}"
        elif self.material_type == MaterialType.EXAMPLE:
            header = f"## Usage Example"
        elif self.material_type == MaterialType.DEMONSTRATION:
            header = f"## Demonstration"
        elif self.material_type == MaterialType.FEEDBACK:
            header = f"## Feedback and Corrections"
        else:
            header = "## Learning Material"

        return f"{header}\n\n{self.content}"


class LearningStage(BaseModel):
    """
    Definition of a single learning stage in the curriculum.

    A stage represents a phase where the agent learns new tools
    and is evaluated on both new and previously learned capabilities.
    """

    # Identification
    stage_id: str = Field(description="Unique identifier for the stage")
    stage_name: str = Field(description="Human-readable name of the stage")

    # Tool configuration
    available_tools: Annotated[
        list[str],
        Field(description="All tools available in this stage")
    ]
    new_tools: Annotated[
        list[str],
        Field(description="Tools newly introduced in this stage", default_factory=list)
    ]
    changed_tools: Annotated[
        list[str],
        Field(description="Tools that changed from previous stage", default_factory=list)
    ]
    removed_tools: Annotated[
        list[str],
        Field(description="Tools removed in this stage", default_factory=list)
    ]

    # Task configuration
    learning_tasks: Annotated[
        list[str],
        Field(description="Task IDs for learning phase")
    ]
    eval_tasks: Annotated[
        list[str],
        Field(description="Task IDs for evaluation phase")
    ]
    retention_tasks: Annotated[
        list[str],
        Field(description="Task IDs for retention testing", default_factory=list)
    ]

    # Learning materials
    learning_materials: Annotated[
        list[LearningMaterial],
        Field(description="Learning materials for this stage", default_factory=list)
    ]

    # Evaluation configuration
    num_learning_trials: int = Field(
        description="Number of trials per learning task", default=3
    )
    num_eval_trials: int = Field(
        description="Number of trials per evaluation task", default=4
    )

    # Stage gating
    min_pass_rate: float = Field(
        description="Minimum pass rate to proceed to next stage",
        default=0.5, ge=0.0, le=1.0
    )
    prerequisites: Annotated[
        list[str],
        Field(description="Stage IDs that must be completed first", default_factory=list)
    ]

    def get_all_tasks(self) -> list[str]:
        """Get all task IDs in this stage."""
        return self.learning_tasks + self.eval_tasks + self.retention_tasks

    def get_learning_prompt_addition(self) -> str:
        """
        Generate the prompt addition containing learning materials.

        This is added to the agent's system prompt during this stage.
        """
        if not self.learning_materials:
            return ""

        sections = ["\n\n# New Tools and Updates for This Stage\n"]

        for material in self.learning_materials:
            sections.append(material.to_prompt_section())
            sections.append("")  # Empty line between sections

        return "\n".join(sections)

    def has_new_tools(self) -> bool:
        """Check if this stage introduces new tools."""
        return len(self.new_tools) > 0

    def has_changed_tools(self) -> bool:
        """Check if this stage has changed tools."""
        return len(self.changed_tools) > 0

    def get_tool_diff_summary(self) -> str:
        """Get a summary of tool changes in this stage."""
        parts = []
        if self.new_tools:
            parts.append(f"New: {', '.join(self.new_tools)}")
        if self.changed_tools:
            parts.append(f"Changed: {', '.join(self.changed_tools)}")
        if self.removed_tools:
            parts.append(f"Removed: {', '.join(self.removed_tools)}")
        return " | ".join(parts) if parts else "No changes"

    def __str__(self) -> str:
        return (
            f"LearningStage({self.stage_id}: {self.stage_name}, "
            f"tools={len(self.available_tools)}, "
            f"new={len(self.new_tools)}, "
            f"tasks={len(self.get_all_tasks())})"
        )
