import json
import uuid
import time
from typing import List, Dict, Any, Optional

# Simple token cost estimation (provider-agnostic heuristic)
# This is a rough estimate; real costs depend on the specific model and provider.
# A common heuristic is ~4 chars per token, and then a cost per 1k tokens.
# Here, we assume a generic cost for input and output, scaled.
COST_PER_1K_INPUT_TOKENS = 0.0005  # Example: $0.50 per 1M input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.0015 # Example: $1.50 per 1M output tokens
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    """Estimates tokens based on character count."""
    return len(text) // CHARS_PER_TOKEN

class AgentTracer:
    """
    A simple, provider-agnostic tracer for agent execution paths.
    Captures steps, LLM calls, tool calls, and allows for custom events.
    """
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.trace: List[Dict[str, Any]] = []
        self._current_run_id: Optional[str] = None

    def start_run(self, run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Starts a new agent run."""
        if self._current_run_id:
            raise RuntimeError(f"Run {self._current_run_id} is already active. End it before starting a new one.")
        self._current_run_id = run_id or str(uuid.uuid4())
        self.log_event("agent_run_start", {"run_id": self._current_run_id, "metadata": metadata or {}})

    def end_run(self, metadata: Optional[Dict[str, Any]] = None):
        """Ends the current agent run."""
        if not self._current_run_id:
            raise RuntimeError("No active run to end. Call start_run() first.")
        self.log_event("agent_run_end", {"run_id": self._current_run_id, "metadata": metadata or {}})
        self._current_run_id = None

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Logs a generic event."""
        if not self._current_run_id:
            raise RuntimeError("No active run. Call start_run() first.")
        self.trace.append({
            "timestamp": time.time(),
            "agent_id": self.agent_id,
            "run_id": self._current_run_id,
            "event_type": event_type,
            "payload": payload
        })

    def log_llm_call(self, model_name: str, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Logs an LLM call with prompt and response."""
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_tokens(response)
        cost = (prompt_tokens / 1000 * COST_PER_1K_INPUT_TOKENS) + \
               (response_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)

        self.log_event("llm_call", {
            "model_name": model_name,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "estimated_cost": cost,
            "metadata": metadata or {}
        })

    def log_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None):
        """Logs a tool call with input and output."""
        self.log_event("tool_call", {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "metadata": metadata or {}
        })

    def log_agent_step(self, step_description: str, observation: Any, metadata: Optional[Dict[str, Any]] = None):
        """Logs a general agent processing step."""
        self.log_event("agent_step", {
            "description": step_description,
            "observation": observation,
            "metadata": metadata or {}
        })

    def log_user_feedback(self, feedback_type: str, feedback_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Logs user feedback related to an agent interaction."""
        self.log_event("user_feedback", {
            "feedback_type": feedback_type,
            "feedback_data": feedback_data,
            "metadata": metadata or {}
        })

    def get_trace(self) -> List[Dict[str, Any]]:
        """Returns the collected trace events."""
        return self.trace

    def save_trace(self, filepath: str):
        """Saves the trace to a file, one JSON object per line."""
        with open(filepath, 'a') as f:
            for event in self.trace:
                f.write(json.dumps(event) + '\n')
        # Clear trace after saving to prevent duplicates if append is called repeatedly
        self.trace.clear()
