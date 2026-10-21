class SecurityError(Exception):
    # Exceção base para violacoes de segurança e sandbox
    pass

class IllegalTransition(SecurityError):
    # Exceção levantada quando uma transicao de estado da maquina de estados é inválida
    def __init__(self, current_state, target_state):
        super().__init__(f"Transição ilegal de '{current_state}' para '{target_state}'.")
        self.current_state = current_state
        self.target_state = target_state

class GateViolation(SecurityError):
    #Exceção levantada quando um Hard Gate de seguranca ou validacao é violado
    pass