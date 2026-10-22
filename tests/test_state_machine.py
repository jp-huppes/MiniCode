import pytest
from src.agent.state import State, TRANSITIONS, AgentStateMachine
from src.security.errors import IllegalTransition

def test_cartesian_product_transitions():
    # Valida o produto cartesiano 8x8 (64 pares) de transições da maquina de estados.
    states = list(State)
    for s1 in states:
        for s2 in states:
            machine = AgentStateMachine()
            machine.state = s1  # forca o estado inicial para o teste exaustivo
            allowed = TRANSITIONS.get(s1, set())
            if s2 in allowed:
                machine.transition(s2)
                assert machine.state == s2
            else:
                with pytest.raises(IllegalTransition): machine.transition(s2)

def test_terminal_states_no_transition():
    # Garante que estados terminais (DONE e FAILED) ñ permitem nenhuma transicao de output
    for terminal in [State.DONE, State.FAILED]:
        machine = AgentStateMachine()
        machine.state = terminal
        for target in State:
            if target != terminal:
                with pytest.raises(IllegalTransition): machine.transition(target)

def test_gate_sem_teste_sem_done():
    # Valida o Hard Gate (sem teste = sem DONE)
    machine = AgentStateMachine()
    
    # estado inicial: sem testes executados
    assert machine.validate_done_gate() is False

    # executa teste com sucesso, sem escrita posterior -> DONE permitido
    machine.record_test(passed=True)
    assert machine.validate_done_gate() is True

    # realiza uma escrita após o teste -> ivalida o DONE (exige novo teste)
    machine.record_write()
    assert machine.validate_done_gate() is False

    # executa novo teste com sucesso apos a escrita -> DONE permitido novamente
    machine.record_test(passed=True)
    assert machine.validate_done_gate() is True

    # caso o teste falhe (passed=False) -> DONE bloqueado
    machine.record_write()
    machine.record_test(passed=False)
    assert machine.validate_done_gate() is False

def test_gate_sem_spec_plan_sem_escrita():
    # valida o Hard Gate (sem SPEC/PLAN = sem escrita)
    machine = AgentStateMachine()
    
    # em RECEIVED, SPECIFIED ou PLANNED, escrita deve ser rejeitada
    for s in [State.RECEIVED, State.SPECIFIED, State.PLANNED]:
        machine.state = s
        err = machine.check_tool_permission("write_file")
        assert err is not None

    # nos estados permitidos (IMPLEMENTING e REPAIRING), escrita é autorizada
    for s in [State.IMPLEMENTING, State.REPAIRING]:
        machine.state = s
        err = machine.check_tool_permission("write_file")
        assert err is None

def test_max_repair_attempts_leads_to_failed():
    # Valida que o estouro de MAX_REPAIR_ATTEMPTS transita o agente para FAILED
    machine = AgentStateMachine(max_repair_attempts=2)
    machine.state = State.VERIFYING
    
    machine.transition(State.REPAIRING)
    assert machine.repair_attempts == 1
    
    machine.state = State.VERIFYING
    machine.transition(State.REPAIRING)
    assert machine.repair_attempts == 2
    assert machine.state == State.FAILED  # estourou o limite configurado (2)