from typing import Dict, Any, Optional
from src.agent.state import AgentStateMachine, State
from src.agent.schemas import AgentStateData

class ExecutionGateManager:
    # Gerencia os Hard Gates de execucao e as regras de transição baseadas em estado
    def __init__(self, state_machine: AgentStateMachine, state_data: AgentStateData):
        self.state_machine = state_machine; self.state_data = state_data

    def check_gate(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """
        Valida se a acao solicitada a permitida no estado atual do agente.
            Retorna uma string com o motivo da rejeicao se violado, ou None se aprovado.
        """
        # 1. gate de permissao de escrita sem SPEC/PLAN = sem escrita)
        if tool_name in {"write_file", "replace_text"}:
            permission_error = self.state_machine.check_tool_permission(tool_name)
            if permission_error:
                return f"Gate de Estado Violado (Escrita Proibida): {permission_error}"

        # 2. gate da ferramenta 'done' (sem teste bem-sucedido apos ultiomo alteracao, sem DONE)
        if tool_name == "done":
            if self.state_machine.state in {State.IMPLEMENTING, State.REPAIRING}:
                return None
            if not self.state_data.tests_passed:
                return "Gate 'DONE' Rejeitado: Nenhum teste bem-sucedido foi executado. Execute 'run_tests' com sucesso antes de finalizar."
            if self.state_data.modified_after_last_test:
                return "Gate 'DONE' Rejeitado: O código foi modificado após a última execução bem-sucedida de testes. Rode 'run_tests' novamente."
            
            if not self.state_machine.validate_done_gate():
                return "Gate 'DONE' Rejeitado pela Máquina de Estados."

        return None

    def update_flags_after_execution(self, tool_name: str, result: Dict[str, Any]):
        # Atualiza as flags de controle estruturado (tests_passed e modified_after_last_test) com base no resultado da ferramenta executada
        if tool_name in {"write_file", "replace_text"}:
            if result.get("ok", False):
                self.state_data.tests_passed = False
                self.state_data.modified_after_last_test = True
                if hasattr(self.state_machine, "record_write"): self.state_machine.record_write()

        elif tool_name == "run_tests":
            passed = result.get("ok", False); self.state_data.tests_passed = passed
            if passed: self.state_data.modified_after_last_test = False
            if hasattr(self.state_machine, "record_test"): self.state_machine.record_test(passed=passed)