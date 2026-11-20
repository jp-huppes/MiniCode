from typing import Any, Dict, Optional

from src.agent.schemas import AgentStateData
from src.agent.state import AgentStateMachine, State
from src.tools.catalog import WRITE_TOOLS
class ExecutionGateManager:
    def __init__(self, state_machine: AgentStateMachine, state_data: AgentStateData):
        self.state_machine = state_machine
        self.state_data = state_data

    def check_gate(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        if tool_name in WRITE_TOOLS:
            permission_error = self.state_machine.check_tool_permission(tool_name)
            if permission_error:
                return f"Gate de Estado Violado (WRITE_TOOLS Proibida): {permission_error}"

        if tool_name == "done":
            if self.state_machine.state in {State.IMPLEMENTING, State.REPAIRING}: return None
            if not self.state_data.tests_passed:
                return "Gate 'DONE' Rejeitado: Nenhum teste bem-sucedido foi executado."
            if self.state_data.modified_after_last_test:
                return "Gate 'DONE' Rejeitado: O código foi modificado após a última execução bem-sucedida de testes."
            if not self.state_machine.validate_done_gate():
                return "Gate 'DONE' Rejeitado pela Máquina de Estados."

        return None

    def update_flags_after_execution(self, tool_name: str, result: Dict[str, Any]):
        if tool_name in WRITE_TOOLS:
            if result.get("ok", False):
                self.state_data.tests_passed = False
                self.state_data.modified_after_last_test = True
                self.state_machine.record_write()

        elif tool_name == "run_tests":
            passed = result.get("ok", False)
            self.state_data.tests_passed = passed
            if passed:
                self.state_data.modified_after_last_test = False
            self.state_machine.record_test(passed=passed)