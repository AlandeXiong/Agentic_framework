"""Workflow definition for orchestrating tool-based workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic.core.tool import Tool, ToolExecutionError
from agentic.core.workflow_step import StepType, WorkflowStep


class WorkflowContext(BaseModel):
    """Execution context for a workflow."""

    # Arbitrary key-value context shared across steps
    data: Dict[str, Any] = Field(default_factory=dict, description="Shared workflow context")
    # Per-step results: {step_id: {\"result\": Any, \"error\": Optional[str], ...}}
    step_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Results for each step"
    )
    # Last executed step id
    last_step_id: Optional[str] = Field(None, description="Last executed step id")
    # Last step result (for convenience)
    last_result: Optional[Any] = Field(None, description="Last step result")


class Workflow(BaseModel):
    """
    A workflow is an orchestrated sequence of steps.

    Design goals:
    - Code-first: defined in Python, not config files
    - Tool-decoupled: only depends on Tool interface
    - Deployment-agnostic: no IO or infra assumptions

    This class focuses on core orchestration logic. Execution strategies
    (e.g., integration with agents, async, distributed) are handled by
    higher-level runners.
    """

    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Human-readable workflow name")
    description: Optional[str] = Field(None, description="Workflow description")

    # Step registry
    steps: Dict[str, WorkflowStep] = Field(
        default_factory=dict, description="Mapping from step id to step definition"
    )

    # Entry step
    start_step_id: str = Field(..., description="ID of the first step to execute")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def add_step(self, step: WorkflowStep) -> None:
        """Register a step in the workflow."""
        if step.id in self.steps:
            raise ValueError(f"Step '{step.id}' is already registered in workflow '{self.id}'")
        self.steps[step.id] = step

    def get_step(self, step_id: str) -> WorkflowStep:
        """Get a step by id."""
        try:
            return self.steps[step_id]
        except KeyError:
            raise KeyError(f"Step '{step_id}' not found in workflow '{self.id}'")

    def run(
        self,
        tools: Dict[str, Tool],
        context: Optional[WorkflowContext] = None,
    ) -> WorkflowContext:
        """
        Run the workflow synchronously.

        Args:
            tools: Mapping from tool name to Tool instance
            context: Optional initial context

        Returns:
            Final workflow context with all step results
        """
        if context is None:
            context = WorkflowContext()

        current_step_id: Optional[str] = self.start_step_id
        visited: List[str] = []

        while current_step_id:
            if current_step_id in visited:
                # Prevent infinite loops caused by misconfigured workflows
                raise RuntimeError(
                    f"Detected loop at step '{current_step_id}' in workflow '{self.id}'. "
                    "Configure LOOP-type steps explicitly instead of cyclic references."
                )
            visited.append(current_step_id)

            step = self.get_step(current_step_id)
            context.last_step_id = step.id

            if step.step_type == StepType.TOOL:
                next_step_id = self._run_tool_step(step, tools, context)
            elif step.step_type == StepType.CONDITION:
                next_step_id = self._run_condition_step(step, context)
            elif step.step_type == StepType.PARALLEL:
                next_step_id = self._run_parallel_step(step, tools, context)
            elif step.step_type == StepType.LOOP:
                next_step_id = self._run_loop_step(step, tools, context)
            else:
                raise ValueError(f"Unsupported step type: {step.step_type}")

            current_step_id = next_step_id

        return context

    # ---- Step handlers -------------------------------------------------

    def _run_tool_step(
        self,
        step: WorkflowStep,
        tools: Dict[str, Tool],
        context: WorkflowContext,
    ) -> Optional[str]:
        """Execute a TOOL-type step."""
        if not step.tool_name:
            raise ValueError(f"Step '{step.id}' is TOOL type but has no tool_name")

        tool = tools.get(step.tool_name)
        if not tool:
            raise ToolExecutionError(
                tool_name=step.tool_name,
                message=f"Tool '{step.tool_name}' not found for step '{step.id}'",
            )

        # Resolve parameters from context
        params = step.resolve_params(
            {
                "context": context.data,
                "step_results": context.step_results,
            }
        )

        step_result: Dict[str, Any] = {}
        try:
            if not tool.validate(**params):
                raise ToolExecutionError(
                    tool_name=tool.name,
                    message=f"Validation failed for tool '{tool.name}' in step '{step.id}'",
                )

            result = tool.execute(**params)
            step_result["result"] = result
            step_result["error"] = None
            context.last_result = result

            # Optionally expose result at workflow context root
            if step.output_key:
                context.data[step.output_key] = result
        except Exception as exc:
            error_msg = str(exc)
            step_result["result"] = None
            step_result["error"] = error_msg
            context.last_result = None

            if step.on_error:
                # Store result and jump to error handler step
                if "step_results" not in context.step_results:
                    # step_results is already a dict-of-dicts; no extra nesting
                    pass
                context.step_results[step.id] = step_result
                return step.on_error

            if not step.continue_on_error:
                raise

        # Store step result in context
        context.step_results[step.id] = step_result

        # Default: stop unless caller wires explicit next step using condition
        return None

    def _run_condition_step(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
    ) -> Optional[str]:
        """Execute a CONDITION-type step."""
        condition_met = step.evaluate_condition(
            {
                "context": context.data,
                "step_results": context.step_results,
            }
        )

        branch = "on_true" if condition_met else "on_false"
        next_steps = step.on_true if condition_met else step.on_false

        # For now, we support single-next-step branching.
        # Multiple next steps can be modeled via an explicit PARALLEL step.
        if not next_steps:
            # No explicit next step; terminate.
            return None
        if len(next_steps) > 1:
            # Keep simple and explicit in v1
            raise ValueError(
                f"Step '{step.id}' {branch} has multiple next steps defined. "
                "Use a dedicated PARALLEL step for fan-out."
            )
        return next_steps[0]

    def _run_parallel_step(
        self,
        step: WorkflowStep,
        tools: Dict[str, Tool],
        context: WorkflowContext,
    ) -> Optional[str]:
        """
        Execute a PARALLEL-type step.

        NOTE: This initial implementation executes steps sequentially for simplicity.
        A future enhancement can introduce true concurrency (threads, async, etc.)
        without changing the public API.
        """
        if not step.parallel_steps:
            return None

        for child_step_id in step.parallel_steps:
            child_step = self.get_step(child_step_id)
            if child_step.step_type == StepType.TOOL:
                self._run_tool_step(child_step, tools, context)
            elif child_step.step_type == StepType.CONDITION:
                # Nested condition inside parallel – allowed but advanced.
                next_id = self._run_condition_step(child_step, context)
                if next_id:
                    # Execute nested chain sequentially
                    nested_current = next_id
                    while nested_current:
                        nested_step = self.get_step(nested_current)
                        if nested_step.step_type == StepType.TOOL:
                            nested_current = self._run_tool_step(
                                nested_step, tools, context
                            )
                        else:
                            break
            # LOOP inside PARALLEL is intentionally not supported in v1

        # PARALLEL step does not define its own next step; caller must wire via condition
        return None

    def _run_loop_step(
        self,
        step: WorkflowStep,
        tools: Dict[str, Tool],
        context: WorkflowContext,
    ) -> Optional[str]:
        """Execute a LOOP-type step."""
        if not step.loop_steps:
            return None

        iterations = 0
        while iterations < step.max_iterations and step.evaluate_loop_condition(
            {
                "context": context.data,
                "step_results": context.step_results,
            }
        ):
            iterations += 1
            for child_step_id in step.loop_steps:
                child_step = self.get_step(child_step_id)
                if child_step.step_type == StepType.TOOL:
                    self._run_tool_step(child_step, tools, context)
                elif child_step.step_type == StepType.CONDITION:
                    next_id = self._run_condition_step(child_step, context)
                    if next_id:
                        nested_current = next_id
                        while nested_current:
                            nested_step = self.get_step(nested_current)
                            if nested_step.step_type == StepType.TOOL:
                                nested_current = self._run_tool_step(
                                    nested_step, tools, context
                                )
                            else:
                                break
                # PARALLEL inside LOOP is intentionally not supported in v1

        return None

