import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable

from src.config import MAX_ITERATIONS, MAX_MODEL_LEN, T_RESP_RESERVE, MAX_PARSE_FAILURES, MAX_REPAIR_ATTEMPTS
from src.llm.client import LLMClient
from src.agent.state import AgentStateMachine, State
from src.agent.schemas import AgentStateData, ActionRequest
from src.agent.validator import ActionValidator
from src.agent.gates import ExecutionGateManager, ESCRITA
from src.agent.executor import ToolExecutor
from src.agent.context_manager import estimar_tokens, truncar_pytest_output, formatar_historico_reparos
from src.agent.verifier import HarnessVerifier, DEFAULT_TEST_CMD
from src.agent.repair_context import RepairContextBuilder
from src.agent.cobertura import obter_testes_exigidos
from src.tools.registry import ToolRegistry

_SYSTEM_PROMPT_TEMPLATE = """Você é o Agente Autônomo de Programação do MiniCode.
Sua missão é executar o plano em 'plan.md' para cumprir os critérios de 'specification.md'.

A cada turno, responda com UM único objeto JSON, dentro de um bloco ```json:
{
  "thought": "raciocínio breve",
  "tool": "nome_da_ferramenta",
  "args": { ... }
}

Ferramentas disponíveis (use exatamente estes nomes e argumentos):
__CATALOGO__
- done: encerra a implementação. Sem argumentos. Use quando todos os passos do plano estiverem concluídos. Os testes são executados automaticamente pelo harness depois de 'done'.

Regras:
- Use sempre 'path' para caminhos de arquivo. Não existe 'file_path'.
- 'write_file' SUBSTITUI o arquivo inteiro. Para acrescentar funções ou testes a um arquivo que já existe, use 'append_file'.
- Os testes exigidos pela SPEC (test_ac_n) devem existir em tests/ com o nome exato. O harness verifica isso ao final.
- Não chame 'run_tests': a verificação é feita pelo harness.
- Chame 'done' somente quando todos os passos do plano estiverem concluídos.
- Não invente argumentos fora da lista acima.
"""


def montar_system_prompt(workspace: Path) -> str:
    catalogo = ToolRegistry(workspace).export_prompt_text()
    return _SYSTEM_PROMPT_TEMPLATE.replace("__CATALOGO__", catalogo)


class AgentOrchestrator:
    def __init__(
        self,
        workspace_dir: str | Path,
        minicode_dir: str | Path,
        llm_client: LLMClient,
        max_iterations: int = MAX_ITERATIONS,
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
        test_cmd: Optional[str] = None,
        state_machine: Optional[AgentStateMachine] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.workspace = Path(workspace_dir).resolve()
        self.minicode_dir = Path(minicode_dir).resolve()
        self.llm_client = llm_client
        self.on_event = on_event or (lambda tipo, payload: None)

        self.system_prompt = montar_system_prompt(self.workspace)

        self.state_machine = state_machine or AgentStateMachine(max_repair_attempts=max_repair_attempts)
        self.state_data = AgentStateData(max_iterations=max_iterations)

        self.validator = ActionValidator(self.workspace)
        self.gates = ExecutionGateManager(self.state_machine, self.state_data)
        self.executor = ToolExecutor(self.workspace, self.minicode_dir)

        self.verifier = HarnessVerifier(self.workspace, test_cmd=test_cmd or DEFAULT_TEST_CMD)
        self.repair_builder = RepairContextBuilder(self.workspace)

        self.action_history: List[Dict[str, Any]] = []
        self.touched_files: List[str] = []

    # ---------- transições com persistência e evento ----------

    def _mudar_estado(self, novo: State):
        antigo = self.state_machine.state
        self.state_machine.transition(novo)
        self.state_data.state = novo.value
        self.executor.save_state_atomic(self.state_data)
        self.on_event("state_changed", {"old": antigo.value, "new": novo.value})

    def _falhar(self, motivo: str):
        antigo = self.state_machine.state
        self.state_machine.state = State.FAILED
        self.state_data.failure_reason = motivo
        self.state_data.state = State.FAILED.value
        self.executor.save_state_atomic(self.state_data)
        self.on_event("state_changed", {"old": antigo.value, "new": State.FAILED.value})

    # ---------- parse e prompt ----------

    def _parse_action_response(self, response_text: str) -> Tuple[Optional[ActionRequest], Optional[str]]:
        text = response_text.strip()

        if "```" in text:
            for part in text.split("```"):
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    text = cleaned
                    break

        if not (text.startswith("{") and text.endswith("}")):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                text = text[start:end + 1]

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None, "O JSON retornado não é um objeto/dicionário válido."
            tool = data.get("tool")
            args = data.get("args", {})
            if not tool:
                return None, "O campo obrigatório 'tool' está ausente na ação."
            if not isinstance(args, dict):
                return None, "O campo 'args' deve ser um dicionário."
            return ActionRequest(tool=tool, args=args), None
        except json.JSONDecodeError as e:
            return None, f"Erro ao decodificar JSON de ação: {str(e)}. Certifique-se de retornar um JSON válido."

    def _montar_prompt_execucao(self, spec_content: str, plan_content: str, ultima_observacao: str) -> str:
        if self.state_machine.state == State.REPAIRING:
            last_output = ultima_observacao if "Saída:" not in ultima_observacao else ultima_observacao.split("Saída:")[-1]
            return self.repair_builder.montar_prompt_reparo(spec_content, self.touched_files, last_output)

        historico_resumido = formatar_historico_reparos(self.action_history)
        exigidos = obter_testes_exigidos(self.workspace / "specification.md")
        bloco_testes = ", ".join(exigidos) if exigidos else "(nenhum listado na SPEC)"

        return f"""SPECIFICATION:
{spec_content}

PLAN:
{plan_content}

TESTES EXIGIDOS PELA SPEC (devem existir em tests/ com estes nomes exatos): {bloco_testes}

ESTADO ATUAL DO AGENTE:
Iteração: {self.state_data.iteration} / {self.state_data.max_iterations}
Testes Passaram: {self.state_data.tests_passed}
Modificado após último teste: {self.state_data.modified_after_last_test}

HISTÓRICO RECENTE:
{historico_resumido}

ÚLTIMA OBSERVAÇÃO / RESULTADO DA FERRAMENTA:
{ultima_observacao}

Com base nas informações acima, decida qual a próxima ação a ser tomada. Responda estritamente no formato JSON exigido.
"""

    # ---------- loop principal ----------

    def run(self, spec_content: str, plan_content: str) -> State:
        while self.state_machine.state in {State.RECEIVED, State.SPECIFIED, State.PLANNED}:
            if self.state_machine.state == State.RECEIVED:
                self.state_machine.transition(State.SPECIFIED)
            elif self.state_machine.state == State.SPECIFIED:
                self.state_machine.transition(State.PLANNED)
            elif self.state_machine.state == State.PLANNED:
                self.state_machine.transition(State.IMPLEMENTING)

        self.state_data.state = self.state_machine.state.value
        self.executor.save_state_atomic(self.state_data, task="Execução Autônoma MiniCode")

        ultima_observacao = "Início do loop de execução. Nenhuma ferramenta executada ainda."

        while self.state_data.iteration < self.state_data.max_iterations:
            self.state_data.iteration += 1

            # 1. verificação conduzida pelo harness
            if self.state_machine.state == State.VERIFYING:
                test_result = self.verifier.run_verification(self.state_machine.sequence_counter)
                self.state_data.tests_passed = test_result.tests_passed
                self.state_machine.record_test(passed=test_result.tests_passed)
                self.on_event("tests_run", {
                    "command": self.verifier.test_cmd,
                    "passed": test_result.tests_passed,
                    "output": test_result.output,
                })

                if test_result.tests_passed and self.state_machine.validate_done_gate():
                    self._mudar_estado(State.DONE)
                    break

                tentativa = self.state_machine.repair_attempts + 1
                if tentativa >= self.state_machine.max_repair_attempts:
                    self._falhar("max_repair_attempts")
                    break

                self.repair_builder.registrar_tentativa(tentativa, self.touched_files, test_result.output, False)
                self.on_event("repair_attempt", {"k": tentativa, "K": self.state_machine.max_repair_attempts})
                self._mudar_estado(State.REPAIRING)
                ultima_observacao = f"Testes falharam no harness. Entrando em REPAIRING.\nSaída:\n{test_result.output}"
                continue

            # 2. orçamento de tokens
            prompt_texto = self._montar_prompt_execucao(spec_content, plan_content, ultima_observacao)
            limite_tokens = 0.8 * (MAX_MODEL_LEN - T_RESP_RESERVE)
            if estimar_tokens(prompt_texto) > limite_tokens:
                self._falhar("context_overflow")
                break

            # 3. chamada da LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_texto},
            ]
            try:
                resposta_llm = self.llm_client.chat(messages)
            except Exception:
                self._falhar("llm_error")
                break

            texto_resposta = resposta_llm.content if hasattr(resposta_llm, "content") else str(resposta_llm)

            # 4. parse
            action_req, parse_error = self._parse_action_response(texto_resposta or "")
            if parse_error:
                self.state_data.parse_failures += 1
                self.on_event("action", {"tool": "(resposta inválida)", "args": {}, "error": parse_error})
                if self.state_data.parse_failures >= MAX_PARSE_FAILURES:
                    self._falhar("parse_failure")
                    break
                ultima_observacao = f"Erro de Parse de Ação: {parse_error}. Por favor, retorne um JSON válido."
                continue

            self.state_data.parse_failures = 0
            tool, args = action_req.tool, action_req.args

            # 5. validador
            erro = self.validator.validate(tool, args)
            if erro:
                self.executor.log_action(self.state_data.iteration, tool, args, erro, None, None, {})
                self.on_event("action", {"tool": tool, "args": args, "error": erro})
                ultima_observacao = f"Ação rejeitada pelo validador: {erro}"
                continue

            # 6. hard gates
            erro = self.gates.check_gate(tool, args)
            if erro:
                self.executor.log_action(self.state_data.iteration, tool, args, None, erro, None, {})
                self.on_event("action", {"tool": tool, "args": args, "error": erro})
                ultima_observacao = (f"Aviso do Gate: {erro}" if tool == "done"
                                     else f"Ação bloqueada pelo Hard Gate: {erro}")
                continue

            # 7. 'run_tests' é exclusivo do harness
            if tool == "run_tests":
                erro = "'run_tests' é executado automaticamente pelo harness. Quando a implementação estiver completa, use 'done'."
                self.on_event("action", {"tool": tool, "args": args, "error": erro})
                ultima_observacao = f"Ação rejeitada: {erro}"
                continue

            # 8. 'done' dispara a verificação
            if tool == "done":
                self.executor.log_action(self.state_data.iteration, "done", args, None, None,
                                         {"ok": True, "output": "", "duration": 0.0}, {})
                self.on_event("action", {"tool": "done", "args": {}, "error": None})
                self._mudar_estado(State.VERIFYING)
                continue

            # 9. execução de ferramenta
            result = self.executor.execute(tool, args)

            if tool in ESCRITA and result["ok"]:
                p = args.get("path")
                if p and p not in self.touched_files:
                    self.touched_files.append(p)

            self.gates.update_flags_after_execution(tool, result)
            self.executor.log_action(self.state_data.iteration, tool, args, None, None, result, {})
            self.on_event("action", {
                "tool": tool,
                "args": args,
                "error": None if result["ok"] else (result.get("error") or result["output"]),
            })

            self.action_history.append({"tool": tool, "args": args, "ok": result["ok"], "output": result["output"]})
            ultima_observacao = f"Ferramenta '{tool}' executada. Status Ok: {result['ok']}.\nSaída:\n{result['output']}"

        # estouro de iterações
        if (self.state_data.iteration >= self.state_data.max_iterations
                and self.state_machine.state not in {State.VERIFYING, State.DONE, State.FAILED}):
            self._falhar("max_iterations")

        return self.state_machine.state