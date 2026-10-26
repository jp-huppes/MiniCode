from enum import Enum
from typing import Dict, Set, Optional
from src.security.errors import IllegalTransition, GateViolation
from src.config import MAX_REPAIR_ATTEMPTS

class State(str, Enum):
    RECEIVED = "RECEIVED"
    SPECIFIED = "SPECIFIED"
    PLANNED = "PLANNED"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    REPAIRING = "REPAIRING"
    DONE = "DONE"
    FAILED = "FAILED"

# tabela estrita de transicoes validas entre estados
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
        """
        Inicializa a maquina de estados do agente no estado RECEIVED, configurando contadores sequencias monotonicos para controle dos hard gates de escrita e testes
        """
        self.state = State.RECEIVED
        self.sequence_counter = 0
        self.last_write_seq = 0
        self.last_test_seq = 0
        self.last_test_ok = False
        self.repair_attempts = 0
        self.max_repair_attempts = max_repair_attempts

    def transition(self, new_state: State):
        # Valida e executa a transição de estado estritamente de acordo com a tabela permitida
        if new_state not in TRANSITIONS.get(self.state, set()):
            raise IllegalTransition(self.state, new_state)
        
        self.state = new_state
        if new_state == State.REPAIRING:
            self.repair_attempts += 1
            if self.repair_attempts >= self.max_repair_attempts:
                self.state = State.FAILED

    def record_write(self):
        """
        Atualiza o contador monotonico de escrita (last_write_seq) para invalidar o status de testes anteriores (sem teste apos escrita = sem DONE)
        """
        self.sequence_counter += 1
        self.last_write_seq = self.sequence_counter
        self.last_test_ok = False  # toda escrita ivalida o atestado de testes anterior

    def record_test(self, passed: bool):
        """
        Atualiza o contador monotonico de testes (last_test_seq) e o resultado (last_test_ok) com base no exit code real retornado pelo executor de testes (pytest)
        """
        self.sequence_counter += 1
        self.last_test_seq = self.sequence_counter
        self.last_test_ok = passed

    def validate_done_gate(self) -> bool:
        """
        Avalia o Hard Gate (sem teste = sem DONE)
        Garantindo que DONE só é permitido se o ultimo teste foi executado APOS a ultima escrita e se obteve sucesso absoluto (last_test_ok = True)
        """
        return (self.last_test_seq > self.last_write_seq) and self.last_test_ok

    def check_tool_permission(self, tool_name: str) -> Optional[str]:
        """
        Avalia o Hard Gate (sem SPEC/PLAN = sem write):
        Garantindo que ferramentas de modificacao de arquivos só podem ser chamadas quando o agente estiver nos estados:
            IMPLEMENTING
            REPAIRING
        """
        write_tools = {"write_file", "replace_text", "create_file", "append_file"}
        if tool_name in write_tools:
            if self.state not in {State.IMPLEMENTING, State.REPAIRING}:
                return f"Escrita proibida no estado atual '{self.state}'. A escrita só é permitida em IMPLEMENTING ou REPAIRING."
        return None