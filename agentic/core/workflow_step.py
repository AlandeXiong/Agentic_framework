"""Workflow step definition for tool orchestration."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

from agentic.core.tool import Tool


class StepType(str, Enum):
    """Types of workflow steps."""

    TOOL = "tool"  # Execute a tool
    CONDITION = "condition"  # Conditional branching
    PARALLEL = "parallel"  # Parallel execution
    LOOP = "loop"  # Loop execution


class WorkflowStep(BaseModel):
    """
    A step in a workflow.
    
    Steps can be:
    - Tool execution: Execute a single tool
    - Condition: Branch based on condition
    - Parallel: Execute multiple steps in parallel
    - Loop: Repeat steps based on condition
    """

    id: str = Field(..., description="Unique step identifier")
    step_type: StepType = Field(..., description="Type of step")
    name: str = Field(..., description="Human-readable step name")
    description: Optional[str] = Field(None, description="Step description")

    # Tool execution fields
    tool_name: Optional[str] = Field(None, description="Tool name to execute (for TOOL type)")
    tool_params: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters (can use template variables)"
    )

    # Condition fields
    condition: Optional[Callable[[Dict[str, Any]], bool]] = Field(
        None, description="Condition function for CONDITION type"
    )
    condition_expression: Optional[str] = Field(
        None, description="Condition expression (e.g., 'result > 10')"
    )
    on_true: Optional[List[str]] = Field(
        None, description="Step IDs to execute if condition is true"
    )
    on_false: Optional[List[str]] = Field(
        None, description="Step IDs to execute if condition is false"
    )

    # Parallel execution fields
    parallel_steps: Optional[List[str]] = Field(
        None, description="Step IDs to execute in parallel (for PARALLEL type)"
    )

    # Loop fields
    loop_steps: Optional[List[str]] = Field(
        None, description="Step IDs to execute in loop (for LOOP type)"
    )
    loop_condition: Optional[Callable[[Dict[str, Any]], bool]] = Field(
        None, description="Loop condition function"
    )
    loop_expression: Optional[str] = Field(
        None, description="Loop condition expression"
    )
    max_iterations: int = Field(10, description="Maximum loop iterations")

    # Output mapping
    output_key: Optional[str] = Field(
        None, description="Key to store step result in workflow context"
    )

    # Error handling
    on_error: Optional[str] = Field(
        None, description="Step ID to execute on error"
    )
    continue_on_error: bool = Field(
        False, description="Continue workflow execution on error"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def resolve_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve template variables in tool parameters.
        
        Supports:
        - ${step_id.result} - Get result from a previous step
        - ${step_id.output_key} - Get specific output key from step
        - ${context.key} - Get value from context
        
        Args:
            context: Workflow execution context
            
        Returns:
            Resolved parameters
        """
        resolved = {}
        for key, value in self.tool_params.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a single value (recursive for nested structures)."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # Template variable: ${step_id.result} or ${context.key}
            path = value[2:-1]
            return self._resolve_path(path, context)
        elif isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]
        else:
            return value

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """Resolve a path like 'step1.result' or 'context.key'."""
        parts = path.split(".")
        if parts[0] == "context":
            # Access context directly
            return context.get(".".join(parts[1:]))
        else:
            # Access step result: step_id.result or step_id.output_key
            step_id = parts[0]
            step_results = context.get("step_results", {})
            step_result = step_results.get(step_id, {})
            
            if len(parts) == 2:
                if parts[1] == "result":
                    return step_result.get("result")
                else:
                    return step_result.get(parts[1])
            else:
                # Nested access
                return self._get_nested_value(step_result, parts[1:])

    def _get_nested_value(self, obj: Any, path: List[str]) -> Any:
        """Get nested value from object."""
        for key in path:
            if isinstance(obj, dict):
                obj = obj.get(key)
            elif isinstance(obj, (list, tuple)) and key.isdigit():
                obj = obj[int(key)]
            else:
                return None
            if obj is None:
                return None
        return obj

    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition for this step.
        
        Args:
            context: Workflow execution context
            
        Returns:
            True if condition is met
        """
        if self.condition:
            return self.condition(context)
        elif self.condition_expression:
            # Simple expression evaluation (can be extended)
            try:
                # For now, support simple comparisons
                # In production, use a proper expression evaluator
                return eval(self.condition_expression, {"context": context})
            except Exception:
                return False
        return True

    def evaluate_loop_condition(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate loop condition.
        
        Args:
            context: Workflow execution context
            
        Returns:
            True if loop should continue
        """
        if self.loop_condition:
            return self.loop_condition(context)
        elif self.loop_expression:
            try:
                return eval(self.loop_expression, {"context": context})
            except Exception:
                return False
        return False

    def __repr__(self) -> str:
        """String representation of the step."""
        return f"<WorkflowStep: {self.id} ({self.step_type})>"
