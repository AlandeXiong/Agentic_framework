"""Runner for executing workflows with tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from agentic.core.tool import Tool
from agentic.core.workflow import Workflow, WorkflowContext


class WorkflowRunnerConfig(BaseModel):
    """Configuration for workflow runner."""

    enable_logging: bool = Field(True, description="Enable execution logging")


class WorkflowRunner:
    """
    WorkflowRunner executes a Workflow with a set of tools.

    This is intentionally lightweight and deployment-agnostic.
    More advanced runners (async, distributed, agent-integrated) can build on top of this.
    """

    def __init__(self, enable_logging: bool = True):
        self.config = WorkflowRunnerConfig(enable_logging=enable_logging)
        self.execution_log: list[dict[str, Any]] = []

    def run(
        self,
        workflow: Workflow,
        tools: Dict[str, Tool],
        context: Optional[WorkflowContext] = None,
    ) -> WorkflowContext:
        """
        Execute a workflow with given tools.

        Args:
            workflow: Workflow definition
            tools: Mapping from tool name to Tool instance
            context: Optional initial context

        Returns:
            Final workflow context
        """
        if self.config.enable_logging:
            self._log("workflow_start", workflow_id=workflow.id, workflow_name=workflow.name)

        ctx = workflow.run(tools=tools, context=context)

        if self.config.enable_logging:
            self._log(
                "workflow_complete",
                workflow_id=workflow.id,
                last_step_id=ctx.last_step_id,
            )

        return ctx

    def _log(self, event: str, **data: Any) -> None:
        """Log an execution event."""
        entry = {"event": event, **data}
        self.execution_log.append(entry)

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Get execution log."""
        return self.execution_log.copy()

    def clear_log(self) -> None:
        """Clear execution log."""
        self.execution_log = []

    def __repr__(self) -> str:
        return "<WorkflowRunner>"

