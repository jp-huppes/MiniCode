from dataclasses import dataclass, field
from typing import Dict, Any, List
from src.agent.state import State

@dataclass
class AgentStateData:
    state: str = State.RECEIVED.value
    iteration: int = 0
    max_iterations: int = 20
    tests_passed: bool = False
    modified_after_last_test: bool = True
    parse_failures: int = 0
    last_usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    failure_reason: str = ""

@dataclass
class ActionRequest:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)