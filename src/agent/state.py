from enum import Enum
from typing import Dict, Optional, Set

from src.config import MAX_REPAIR_ATTEMPTS
from src.security.errors import IllegalTransition
from src.tools.catalog import WRITE_TOOLS


class State(str, Enum):
    RECEIVED = "RECEIVED"
    SPECIFIED = "SPECIFIED"
    PLANNED = "PLANNED"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    REPAIRING = "REPAIRING"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS: Dict[State, Set[State]] = {
    State.RECEIVED: {State.SPECIFIED, State.FAILED},
    State.SPECIFIED: {State.PLANNED, State.FAILED},
    State.PLANNED: {State.IMPLEMENTING, State.FAILED},
    State.IMPLEMENTING: {State.VERIFYING, State.FAILED},
    State.VERIFYING: {State.DONE, State.REPAIRING, State.FAILED},
    State.REPAIRING: {State.VERIFYING, State.FAILED},
    State.DONE: set(),
    State.FAILED: set(),
}


class AgentStateMachine:
    def __init__(self, max_repair_attempts: int = MAX_REPAIR_ATTEMPTS):
        self.state = State.RECEIVED
        self.sequence_counter = 0
        self.last_write_seq = 0
        self.last_test_seq = 0
        self.last_test_ok = False
        self.repair_attempts = 0
        self.max_repair_attempts = max_repair_attempts

    def transition(self, new_state: State):
        if new_state not in TRANSITIONS.get(self.state, set()):
            raise IllegalTransition(self.state, new_state)

        self.state = new_state
        if new_state == State.REPAIRING:
            self.repair_attempts += 1
            if self.repair_attempts >= self.max_repair_attempts:
                self.state = State.FAILED

    def record_write(self):
        self.sequence_counter += 1
        self.last_write_seq = self.sequence_counter
        self.last_test_ok = False

    def record_test(self, passed: bool):
        self.sequence_counter += 1
        self.last_test_seq = self.sequence_counter
        self.last_test_ok = passed

    def validate_done_gate(self) -> bool:
        return (self.last_test_seq > self.last_write_seq) and self.last_test_ok

    def check_tool_permission(self, tool_name: str) -> Optional[str]:
        if tool_name in WRITE_TOOLS:
            if self.state not in {State.IMPLEMENTING, State.REPAIRING}:
                return (
                    f"Escrita proibida no estado atual '{self.state}'. "
                    "A escrita só é permitida em IMPLEMENTING ou REPAIRING.")
        return None