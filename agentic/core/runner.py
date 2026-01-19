"""Runner for orchestrating agent execution and multi-agent coordination."""

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field

from agentic.core.agent import Agent
from agentic.core.message import Message, MessageRole


class RunnerConfig(BaseModel):
    """Configuration for the runner."""

    max_rounds: int = Field(10, description="Maximum rounds of multi-agent interaction")
    enable_logging: bool = Field(True, description="Enable execution logging")
    error_handler: Optional[Callable[[Exception, Agent, Message], Message]] = Field(
        None, description="Custom error handler function"
    )


class Runner:
    """
    Runner orchestrates agent execution and multi-agent coordination.
    
    The runner is deployment-agnostic and handles:
    - Single agent execution
    - Multi-agent coordination
    - Error handling
    - Execution flow control
    
    Design principles:
    - Deployment agnostic: No assumptions about where/how agents run
    - Multi-agent support: Coordinate multiple agents working together
    - Error resilience: Graceful error handling and recovery
    - Extensible: Easy to add custom execution strategies
    """

    def __init__(
        self,
        max_rounds: int = 10,
        enable_logging: bool = True,
        error_handler: Optional[Callable[[Exception, Agent, Message], Message]] = None,
    ):
        """
        Initialize the runner.
        
        Args:
            max_rounds: Maximum rounds for multi-agent interactions
            enable_logging: Enable execution logging
            error_handler: Custom error handler function
        """
        self.config = RunnerConfig(
            max_rounds=max_rounds,
            enable_logging=enable_logging,
            error_handler=error_handler,
        )
        self.execution_log: List[Dict[str, Any]] = []

    def run(
        self,
        agent: Agent,
        message: Message,
        **kwargs: Any
    ) -> Message:
        """
        Run a single agent with a message.
        
        Args:
            agent: Agent to run
            message: Input message
            **kwargs: Additional parameters
            
        Returns:
            Agent's response message
        """
        if self.config.enable_logging:
            self._log("run_start", agent=agent, message=message)
        
        try:
            response = agent.process_message(message)
            
            if self.config.enable_logging:
                self._log("run_complete", agent=agent, response=response)
            
            return response
        except Exception as e:
            if self.config.enable_logging:
                self._log("run_error", agent=agent, error=str(e))
            
            if self.config.error_handler:
                return self.config.error_handler(e, agent, message)
            
            # Default error response
            return Message(
                role=MessageRole.ASSISTANT,
                content=f"Error processing request: {str(e)}",
            )

    def run_multi_agent(
        self,
        agents: List[Agent],
        initial_message: Message,
        routing_strategy: Optional[Callable[[Message, List[Agent]], Agent]] = None,
        **kwargs: Any
    ) -> List[Message]:
        """
        Run multiple agents in a coordinated manner.
        
        Args:
            agents: List of agents to coordinate
            initial_message: Initial message to start the conversation
            routing_strategy: Function to determine which agent handles a message.
                            If None, uses round-robin.
            **kwargs: Additional parameters
            
        Returns:
            List of messages from the multi-agent interaction
        """
        if self.config.enable_logging:
            self._log("multi_agent_start", agents=agents, message=initial_message)
        
        messages: List[Message] = []
        current_message = initial_message
        current_agent_idx = 0
        rounds = 0
        
        # Default routing: round-robin
        if routing_strategy is None:
            def routing_strategy(msg: Message, agent_list: List[Agent]) -> Agent:
                nonlocal current_agent_idx
                agent = agent_list[current_agent_idx % len(agent_list)]
                current_agent_idx += 1
                return agent
        
        while rounds < self.config.max_rounds:
            rounds += 1
            
            # Select agent
            agent = routing_strategy(current_message, agents)
            
            if self.config.enable_logging:
                self._log("agent_selected", agent=agent, round=rounds)
            
            try:
                # Process message with selected agent
                response = agent.process_message(current_message)
                messages.append(response)
                
                # Check if conversation should continue
                if self._should_continue(response):
                    current_message = response
                else:
                    break
            except Exception as e:
                if self.config.enable_logging:
                    self._log("agent_error", agent=agent, error=str(e))
                
                if self.config.error_handler:
                    error_response = self.config.error_handler(e, agent, current_message)
                    messages.append(error_response)
                    current_message = error_response
                else:
                    # Stop on error if no handler
                    break
        
        if self.config.enable_logging:
            self._log("multi_agent_complete", messages=messages, rounds=rounds)
        
        return messages

    def _should_continue(self, message: Message) -> bool:
        """
        Determine if multi-agent conversation should continue.
        
        Override this method to customize continuation logic.
        
        Args:
            message: Last message in conversation
            
        Returns:
            True if conversation should continue
        """
        # Default: continue if there are tool calls or if message suggests continuation
        if message.tool_calls:
            return True
        
        # Check for explicit continuation signals in content
        stop_signals = ["done", "complete", "finished", "end"]
        content_lower = message.content.lower()
        if any(signal in content_lower for signal in stop_signals):
            return False
        
        return True

    def _log(self, event: str, **data: Any) -> None:
        """Log an execution event."""
        # Convert Agent objects to their string representation for logging
        log_data = {}
        for key, value in data.items():
            if hasattr(value, "config"):  # Agent object
                log_data[key] = {
                    "name": value.config.name,
                    "repr": repr(value),
                }
            else:
                log_data[key] = value
        
        log_entry = {
            "event": event,
            **log_data,
        }
        self.execution_log.append(log_entry)

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log."""
        return self.execution_log.copy()

    def clear_log(self) -> None:
        """Clear the execution log."""
        self.execution_log = []

    def __repr__(self) -> str:
        """String representation of the runner."""
        return f"<Runner: max_rounds={self.config.max_rounds}>"
