import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from src.config import MAX_ITERATIONS, MAX_MODEL_LEN, T_RESP_RESERVE, MAX_PARSE_FAILURES, MAX_REPAIR_ATTEMPTS
from src.llm.client import LLMClient
from src.agent.state import AgentStateMachine, State
from src.agent.schemas import AgentStateData, ActionRequest
from src.agent.validator import ActionValidator
from src.agent.gates import ExecutionGateManager
from src.agent.executor import ToolExecutor
from src.agent.context_manager import estimar_tokens, truncar_pytest_output, formatar_historico_reparos
from src.agent.verifier import HarnessVerifier
from src.agent.repair_context import RepairContextBuilder

ORCHESTRATOR_SYSTEM_PROMPT = """Você é o Agente Autônomo de Programação do MiniCode.
    Sua missão é executar o plano estabelecido em 'plan.md' para cumprir os critérios da especificação 'specification.md'.

    Em cada turno, você deve responder estritamente com um objeto JSON contendo a ferramenta a ser executada e seus argumentos, encapsulado em um bloco de código markdown:
    ```json
    {
    "thought": "Explicação breve do seu raciocínio estratégico para este passo",
    "tool": "nome_da_ferramenta",
    "args": {
        "param1": "valor1"
        }
    }
    Ferramentas disponíveis: list_files, read_file, write_file, replace_text, run_tests, done."""

class AgentOrchestrator:
    # orquestrador central responsavel por gerenciar o loop de execucao
    def __init__(self, workspace_dir: str | Path, minicode_dir: str | Path, llm_client: LLMClient):
        self.workspace = Path(workspace_dir).resolve()
        self.minicode_dir = Path(minicode_dir).resolve()
        self.llm_client = llm_client

        self.state_machine = AgentStateMachine()
        self.state_data = AgentStateData(max_iterations=MAX_ITERATIONS)
            
        self.validator = ActionValidator(self.workspace)
        self.gates = ExecutionGateManager(self.state_machine, self.state_data)
        self.executor = ToolExecutor(self.workspace, self.minicode_dir)
            
        self.verifier = HarnessVerifier(self.workspace)
        self.repair_builder = RepairContextBuilder(self.workspace)
            
        self.action_history: List[Dict[str, Any]] = []
        self.touched_files: List[str] = []

    def _parse_action_response(self, response_text: str) -> Tuple[Optional[ActionRequest], Optional[str]]:
        # Extrai e valida o JSON de acao retornado pelo LLM de forma tolerante a cercas
        text = response_text.strip()
        
            # extracao flexivel de blocos de codigo ou chaves JSON diretas
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    text = cleaned
                    break
        
            # fallback para buscar a primeira chave '{' e a última '}'
        if not (text.startswith("{") and text.endswith("}")):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start: text = text[start:end + 1]

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None, "O JSON retornado não é um objeto/dicionário válido."
            
            tool = data.get("tool")
            args = data.get("args", {})
            if not tool: return None, "O campo obrigatório 'tool' está ausente na ação."
            if not isinstance(args, dict): return None, "O campo 'args' deve ser um dicionário."

            return ActionRequest(tool=tool, args=args), None
        except json.JSONDecodeError as e:
            return None, f"Erro ao decodificar JSON de ação: {str(e)}. Certifique-se de retornar um JSON válido."

    def _montar_prompt_execucao(self, spec_content: str, plan_content: str, ultima_observacao: str) -> str:
        # Monta o prompt dinamico de execucao ou reparo dependendo do estado atual

        if self.state_machine.state == State.REPAIRING:
            last_output = ultima_observacao if "Saída:" not in ultima_observacao else ultima_observacao.split("Saída:")[-1]
            return self.repair_builder.montar_prompt_reparo(spec_content, self.touched_files, last_output)

        historico_resumido = formatar_historico_reparos(self.action_history)
        
        prompt = f"""SPECIFICATION:
        {spec_content}
        PLAN:
        {plan_content}

        ESTADO ATUAL DO AGENTE:

        Iteração: {self.state_data.iteration} / {self.state_data.max_iterations}

        Testes Passaram: {self.state_data.tests_passed}

        Modificado após último teste: {self.state_data.modified_after_last_test}

        HISTÓRICO RECENTE:
        {historico_resumido}

        ÚLTIMA OBSERVAÇÃO / RESULTADO DA FERRAMENTA:
        {ultima_observacao}

        Com base nas informações acima, decida qual a próxima ação utilitária a ser tomada. Responda estritamente no formato JSON exigido.
        """
        return prompt



    def run(self, spec_content: str, plan_content: str) -> State:
        # executa o loop principal de desenvolvimento autonomo

        # 1. transicionar para EXECUTING e salvar estado inicial atraves dos estados validos
        while self.state_machine.state in {State.RECEIVED, State.SPECIFIED, State.PLANNED}:
            if self.state_machine.state == State.RECEIVED: self.state_machine.transition(State.SPECIFIED)
            elif self.state_machine.state == State.SPECIFIED: self.state_machine.transition(State.PLANNED)
            elif self.state_machine.state == State.PLANNED: self.state_machine.transition(State.IMPLEMENTING)
        
        self.state_data.state = self.state_machine.state.value
        self.executor.save_state_atomic(self.state_data, task="Execução Autônoma MiniCode")

        ultima_observacao = "Início do loop de execução. Nenhuma ferramenta executada ainda."

        while self.state_data.iteration < self.state_data.max_iterations:
            self.state_data.iteration += 1

            # verificacao auto conduzida pelo harness ao entrar em VERIFYING
            if self.state_machine.state == State.VERIFYING:
                test_result = self.verifier.run_verification(self.state_machine.sequence_counter)
                self.state_data.tests_passed = test_result.tests_passed
                self.state_machine.record_test(passed=test_result.tests_passed)
                print(f"DEBUG VERIFYING -> passed: {test_result.tests_passed}, gate: {self.state_machine.validate_done_gate()}, last_test: {self.state_machine.last_test_seq}, last_write: {self.state_machine.last_write_seq}")
                if test_result.tests_passed and self.state_machine.validate_done_gate():
                    self.state_machine.transition(State.DONE)
                    self.state_data.state = State.DONE.value
                    self.executor.save_state_atomic(self.state_data)
                    break
                else:
                    if self.state_machine.repair_attempts >= self.state_machine.max_repair_attempts:
                        self.state_machine.transition(State.FAILED)
                        self.state_data.failure_reason = "max_repair_attempts"
                        self.state_data.state = State.FAILED.value
                        self.executor.save_state_atomic(self.state_data)
                        break
                    
                    self.repair_builder.registrar_tentativa(
                        self.state_machine.repair_attempts + 1,
                        self.touched_files,
                        test_result.output,
                        False
                    )
                    self.state_machine.transition(State.REPAIRING)
                    self.state_data.state = State.REPAIRING.value
                    ultima_observacao = f"Testes falharam no harness. Entrando em REPAIRING.\nSaída:\n{test_result.output}"
                    continue

            # 2. montar prompt e checar orçamento de tokens
            prompt_texto = self._montar_prompt_execucao(spec_content, plan_content, ultima_observacao)
            tokens_estimados = estimar_tokens(prompt_texto)
            limite_tokens = 0.8 * (MAX_MODEL_LEN - T_RESP_RESERVE)

            if tokens_estimados > limite_tokens:
                self.state_data.failure_reason = "context_overflow"
                self.state_machine.state = State.FAILED
                self.state_data.state = State.FAILED.value
                self.executor.save_state_atomic(self.state_data)
                break

            # 3. chamada da LLM
            messages = [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_texto}
            ]

            try: resposta_llm = self.llm_client.chat(messages)
            except Exception as e:
                self.state_data.failure_reason = "llm_error"
                self.state_machine.state = State.FAILED
                self.state_data.state = State.FAILED.value
                self.executor.save_state_atomic(self.state_data)
                break

            # 4. parsear acao
            action_req, parse_error = self._parse_action_response(resposta_llm)
            if parse_error:
                self.state_data.parse_failures += 1
                if self.state_data.parse_failures >= MAX_PARSE_FAILURES:
                    self.state_data.failure_reason = "parse_failure"
                    self.state_machine.state = State.FAILED
                    self.state_data.state = State.FAILED.value
                    self.executor.save_state_atomic(self.state_data)
                    break
                ultima_observacao = f"Erro de Parse de Ação: {parse_error}. Por favor, retorne um JSON válido."
                continue

            # reset parse failures em caso de sucesso no parse
            self.state_data.parse_failures = 0

            # 5. validação de acao e hard gates
            validator_error = self.validator.validate(action_req.tool, action_req.args)
            if validator_error:
                self.executor.log_action(self.state_data.iteration, action_req.tool, action_req.args, validator_error, None, None, {})
                ultima_observacao = f"Ação rejeitada pelo validador: {validator_error}"
                continue

            # Proteção de testes em REPAIRING
            if self.state_machine.state == State.REPAIRING:
                if not self.verifier.validate_repair_path_protection(action_req.tool, action_req.args):
                    ultima_observacao = "Ação rejeitada: proibido alterar arquivos na pasta 'tests/' durante o reparo."
                    continue

            gate_error = self.gates.check_gate(action_req.tool, action_req.args)
            if gate_error:
                self.executor.log_action(self.state_data.iteration, action_req.tool, action_req.args, None, gate_error, None, {})
                if action_req.tool == "done":
                        # gate done rejeitado ñ encerra o loop, apenas avisa o LLM
                    ultima_observacao = f"Aviso do Gate: {gate_error}"
                    continue
                else:
                    ultima_observacao = f"Ação bloqueada pelo Hard Gate: {gate_error}"
                    continue

            # 6. execucao da ferramenta
                # bloqueio de chamada direta de 'run_tests' pela LLM
            if action_req.tool == "run_tests":
                ultima_observacao = "Ação rejeitada: 'run_tests' é executado exclusivamente pelo harness em VERIFYING."
                continue
            
            result = self.executor.execute(action_req.tool, action_req.args)
            if action_req.tool == "run_tests": result["output"] = truncar_pytest_output(result["output"])

            # rastreia arquivos tocados
            if action_req.tool in {"write_file", "replace_text"}:
                p = action_req.args.get("path") or action_req.args.get("file_path")
                if p and p not in self.touched_files: self.touched_files.append(p)

            # 7. atualizar flags e logar
            self.gates.update_flags_after_execution(action_req.tool, result)
            self.executor.log_action(self.state_data.iteration, action_req.tool, action_req.args, None, None, result, {})

                # armazenar no historico recente
            self.action_history.append({"tool": action_req.tool, "args": action_req.args, "ok": result["ok"], "output": result["output"]})

            # 8. condicao de saida / transicao para verificacao
            if action_req.tool == "done":
                self.state_machine.transition(State.VERIFYING)
                self.state_data.state = State.VERIFYING.value
                self.executor.save_state_atomic(self.state_data)
                continue

            ultima_observacao = f"Ferramenta '{action_req.tool}' executada. Status Ok: {result['ok']}.\nSaída:\n{result['output']}"

        # se estourar iteracoes sem finalizar
        if self.state_data.iteration >= self.state_data.max_iterations and self.state_machine.state not in {State.VERIFYING, State.DONE, State.FAILED}:
            self.state_data.failure_reason = "max_iterations"
            self.state_machine.state = State.FAILED
            self.state_data.state = State.FAILED.value
            self.executor.save_state_atomic(self.state_data)

        return self.state_machine.state