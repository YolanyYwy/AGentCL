"""
Curriculum Definition

This module defines the complete curriculum structure for
continual learning evaluation.
"""

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

from tau2.continual.curriculum.stage import LearningStage, LearningMaterial
from tau2.data_model.tasks import Task


CurriculumType = Literal["progressive", "tool_expansion", "tool_evolution", "cross_domain", "mixed"]


class Curriculum(BaseModel):
    """
    Complete curriculum definition for continual learning evaluation.

    A curriculum defines a sequence of learning stages, each introducing
    new tools or modifying existing ones, along with associated tasks
    for learning and evaluation.
    """

    # Identification
    curriculum_id: str = Field(description="Unique identifier for the curriculum")
    curriculum_name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of the curriculum", default="")

    # Domain configuration
    domain: str = Field(description="Domain this curriculum is for (e.g., airline, retail)")
    curriculum_type: CurriculumType = Field(
        description="Type of continual learning scenario",
        default="progressive"
    )

    # Stages
    stages: Annotated[
        list[LearningStage],
        Field(description="Ordered list of learning stages")
    ]

    # Evaluation configuration
    evaluation_config: Annotated[
        dict[str, Any],
        Field(description="Configuration for evaluation", default_factory=dict)
    ]

    # Computed fields (set in __post_init__)
    total_tools: int = Field(description="Total unique tools across all stages", default=0)
    total_tasks: int = Field(description="Total unique tasks across all stages", default=0)

    # Task cache (populated when tasks are loaded)
    _task_cache: dict[str, Task] = {}

    @model_validator(mode='after')
    def compute_totals(self) -> 'Curriculum':
        """Compute total tools and tasks after initialization."""
        all_tools = set()
        all_tasks = set()

        for stage in self.stages:
            all_tools.update(stage.available_tools)
            all_tasks.update(stage.get_all_tasks())

        self.total_tools = len(all_tools)
        self.total_tasks = len(all_tasks)

        return self

    def get_stage(self, stage_id: str) -> Optional[LearningStage]:
        """Get a stage by its ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_stage_index(self, stage_id: str) -> int:
        """Get the index of a stage by its ID."""
        for i, stage in enumerate(self.stages):
            if stage.stage_id == stage_id:
                return i
        return -1

    def get_all_task_ids(self) -> list[str]:
        """Get all unique task IDs across all stages."""
        task_ids = set()
        for stage in self.stages:
            task_ids.update(stage.get_all_tasks())
        return list(task_ids)

    def get_all_tools(self) -> list[str]:
        """Get all unique tools across all stages."""
        tools = set()
        for stage in self.stages:
            tools.update(stage.available_tools)
        return list(tools)

    def set_task_cache(self, tasks: list[Task]) -> None:
        """Set the task cache from a list of tasks."""
        self._task_cache = {task.id: task for task in tasks}

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID from the cache."""
        return self._task_cache.get(task_id)

    def validate_tasks(self, available_task_ids: set[str]) -> list[str]:
        """
        Validate that all referenced tasks exist.

        Returns list of missing task IDs.
        """
        required_tasks = set(self.get_all_task_ids())
        missing = required_tasks - available_task_ids
        return list(missing)

    def get_tools_at_stage(self, stage_index: int) -> list[str]:
        """Get all tools available up to and including a stage."""
        if stage_index < 0 or stage_index >= len(self.stages):
            return []
        return self.stages[stage_index].available_tools

    def get_new_tools_at_stage(self, stage_index: int) -> list[str]:
        """Get tools newly introduced at a specific stage."""
        if stage_index < 0 or stage_index >= len(self.stages):
            return []
        return self.stages[stage_index].new_tools

    def get_cumulative_tools(self, up_to_stage: int) -> set[str]:
        """Get all tools introduced up to a given stage."""
        tools = set()
        for i in range(min(up_to_stage + 1, len(self.stages))):
            tools.update(self.stages[i].available_tools)
        return tools

    def get_retention_tasks_for_stage(self, stage_index: int) -> list[str]:
        """
        Get appropriate retention tasks for a stage.

        These are tasks from previous stages that test old knowledge.
        """
        if stage_index < 0 or stage_index >= len(self.stages):
            return []
        return self.stages[stage_index].retention_tasks

    def summary(self) -> str:
        """Generate a summary of the curriculum."""
        lines = [
            f"Curriculum: {self.curriculum_name} ({self.curriculum_id})",
            f"Domain: {self.domain}",
            f"Type: {self.curriculum_type}",
            f"Stages: {len(self.stages)}",
            f"Total Tools: {self.total_tools}",
            f"Total Tasks: {self.total_tasks}",
            "",
            "Stages:",
        ]

        for i, stage in enumerate(self.stages):
            lines.append(
                f"  {i}. {stage.stage_name}: "
                f"{len(stage.available_tools)} tools, "
                f"{len(stage.new_tools)} new, "
                f"{len(stage.get_all_tasks())} tasks"
            )

        return "\n".join(lines)

    @classmethod
    def from_json(cls, path: Path | str) -> "Curriculum":
        """Load a curriculum from a JSON file."""
        if isinstance(path, str):
            path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert stage dictionaries to LearningStage objects
        if "stages" in data:
            stages = []
            for stage_data in data["stages"]:
                # Convert learning materials
                if "learning_materials" in stage_data:
                    materials = []
                    for mat_data in stage_data["learning_materials"]:
                        materials.append(LearningMaterial(**mat_data))
                    stage_data["learning_materials"] = materials
                stages.append(LearningStage(**stage_data))
            data["stages"] = stages

        return cls(**data)

    def to_json(self, path: Path | str) -> None:
        """Save the curriculum to a JSON file."""
        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def __str__(self) -> str:
        return f"Curriculum({self.curriculum_id}: {len(self.stages)} stages, {self.total_tools} tools)"

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)

    def __getitem__(self, index: int) -> LearningStage:
        return self.stages[index]
