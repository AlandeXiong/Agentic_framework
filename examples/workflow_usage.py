"""
Workflow usage example for the Agentic framework.

This example shows how to:
- Define workflow steps
- Chain tools via a workflow
- Execute the workflow with WorkflowRunner
"""

from agentic import (
    WorkflowStep,
    StepType,
    Workflow,
    WorkflowContext,
    WorkflowRunner,
)
from agentic.tools import CalculatorTool, WeatherTool


def build_sample_workflow() -> Workflow:
    """
    Build a simple workflow:

    1. Use calculator to compute 15 + 27, store as `sum_result`
    2. If sum_result > 30, query weather in San Francisco
    """
    # Step 1: calculator
    step_calculate = WorkflowStep(
        id="calculate_sum",
        step_type=StepType.TOOL,
        name="Calculate Sum",
        tool_name="calculator",
        tool_params={
            "operation": "add",
            "a": 15,
            "b": 27,
        },
        output_key="sum_result",
    )

    # Step 2: condition on result
    step_condition = WorkflowStep(
        id="check_sum",
        step_type=StepType.CONDITION,
        name="Check Sum > 30",
        condition_expression="context.get('sum_result', 0) > 30",
        on_true=["weather_sf"],
        on_false=[],
    )

    # Step 3: weather query (only runs if condition true)
    step_weather = WorkflowStep(
        id="weather_sf",
        step_type=StepType.TOOL,
        name="Get SF Weather",
        tool_name="weather",
        tool_params={
            "location": "San Francisco, CA",
        },
        output_key="sf_weather",
    )

    wf = Workflow(
        id="calculator_weather_workflow",
        name="Calculator + Weather Workflow",
        start_step_id="calculate_sum",
    )
    wf.add_step(step_calculate)
    wf.add_step(step_condition)
    wf.add_step(step_weather)

    return wf


def main() -> None:
    """Run the workflow example."""
    # Prepare tools
    calculator = CalculatorTool()
    weather = WeatherTool()
    tools = {
        calculator.name: calculator,
        weather.name: weather,
    }

    # Build workflow
    workflow = build_sample_workflow()

    # Initial context (can pre-fill shared data)
    ctx = WorkflowContext()

    # Run
    runner = WorkflowRunner(enable_logging=True)
    final_ctx = runner.run(workflow, tools=tools, context=ctx)

    print("=== Workflow Execution ===")
    print(f"Last step: {final_ctx.last_step_id}")
    print(f"Sum result: {final_ctx.data.get('sum_result')}")
    print(f"SF Weather: {final_ctx.data.get('sf_weather')}")
    print("\nStep results:")
    for step_id, result in final_ctx.step_results.items():
        print(f"  {step_id}: {result}")


if __name__ == "__main__":
    main()

