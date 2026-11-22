import json
import posixpath
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable

from src.tools.catalog import ALLOWED_TOOLS, WRITE_TOOLS, export_normalized_envelope_schema, export_prompt_text
from src.tools.registry import ToolRegistry

from src.config import MAX_ITERATIONS, MAX_MODEL_LEN, T_RESP_RESERVE, MAX_PARSE_FAILURES, MAX_REPAIR_ATTEMPTS

from src.agent.state import AgentStateMachine, State
from src.agent.schemas import AgentStateData, ActionRequest
from src.agent.validator import ActionValidator
from src.agent.gates import ExecutionGateManager
from src.agent.executor import ToolExecutor
from src.agent.context_manager import estimar_tokens, truncar_pytest_output, formatar_historico_reparos
from src.agent.verifier import HarnessVerifier, DEFAULT_TEST_CMD
from src.agent.repair_context import RepairContextBuilder
from src.agent.cobertura import obter_testes_exigidos
from src.agent.lint_testes import verificar_nomes_indefinidos_em_teste

from src.llm.action_contract import interpretar_acao
from src.llm.action_parser import parse_acao_json
from src.llm.errors import ParseError
from src.llm.routing import ROLE_REPAIR, ROLE_TOOL_SELECTION
from src.llm.client import LLMClient

from src.security.sandbox import resolve_in_root

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
- Em arquivos de teste, inclua os imports no topo (ex.: `from src.pure_ops import clamp`). Nunca use um módulo ou função sem importá-lo.
- Arquivos que já existem em tests/ no início são testes da tarefa: NÃO os altere. Coloque os testes 'test_ac_n' exigidos pela SPEC em um arquivo novo (ex.: tests/test_spec_ac.py), como funções `def test_ac_...` no nível do módulo.
"""


def montar_system_prompt(workspace: Path, catalogo_enxuto: bool = False) -> str:
    if catalogo_enxuto:
        linhas = [
            f"- {nome}: {info.description}"
            for nome, info in sorted(ALLOWED_TOOLS.items())
            if not info.harness_only
        ]
        catalogo = "\n".join(linhas)
    else:
        catalogo = export_prompt_text(include_harness_only=False)
    return _SYSTEM_PROMPT_TEMPLATE.replace("__CATALOGO__", catalogo)


class AgentOrchestrator:
    def __init__(self, workspace_dir: str | Path, minicode_dir: str | Path, llm_client: LLMClient, max_iterations: int = MAX_ITERATIONS, max_repair_attempts: int = MAX_REPAIR_ATTEMPTS, test_cmd: Optional[str] = None, state_machine: Optional[AgentStateMachine] = None, on_event: Optional[Callable[[str, dict], None]] = None, run_id: Optional[str] = None, temperature: Optional[float] = None, proteger_testes_existentes: bool = False,):

        self.workspace = Path(workspace_dir).resolve()
        self.minicode_dir = Path(minicode_dir).resolve()
        self.llm_client = llm_client
        self.on_event = on_event or (lambda tipo, payload: None)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.temperature = temperature
        self._ultimo_pytest_output = ""
        self._blocos_antiloop_seguidos = 0

        catalogo_enxuto = getattr(self.llm_client, "constrained_mode", "none") != "none"
        self.system_prompt = montar_system_prompt(self.workspace, catalogo_enxuto=catalogo_enxuto)


        self._state_machine = state_machine or AgentStateMachine(max_repair_attempts=max_repair_attempts)
        self.state_data = AgentStateData(max_iterations=max_iterations)

        self.validator = ActionValidator(self.workspace)
        self.gates = ExecutionGateManager(self.state_machine, self.state_data)
        self.executor = ToolExecutor(self.workspace, self.minicode_dir)

        self.verifier = HarnessVerifier(self.workspace, test_cmd=test_cmd or DEFAULT_TEST_CMD)
        self.repair_builder = RepairContextBuilder(self.workspace)

        self.action_history: List[Dict[str, Any]] = []
        self.touched_files: List[str] = []
        self._testes_protegidos = set()

        if proteger_testes_existentes:
            d = self.workspace / "tests"
            if d.is_dir(): self._testes_protegidos = {p.relative_to(self.workspace).as_posix() for p in d.rglob("*") if p.is_file()}

        self.eval_action_attempts: List[Dict[str, Any]] = []


    @property
    def state_machine(self) -> AgentStateMachine:
        return self._state_machine

    @state_machine.setter
    def state_machine(self, value: AgentStateMachine) -> None:
        self._state_machine = value
        if hasattr(self, "gates") and self.gates is not None: self.gates.state_machine = value

    # ---------- transições com persistência e evento ----------

    def _mudar_estado(self, novo: State):
        antigo = self.state_machine.state
        self.state_machine.transition(novo)
        real = self.state_machine.state
        self.state_data.state = real.value
        if real == State.FAILED and not self.state_data.failure_reason:
            self.state_data.failure_reason = "max_repair_attempts"
        self.executor.save_state_atomic(self.state_data)
        self.on_event("state_changed", {"old": antigo.value, "new": real.value})

    def _falhar(self, motivo: str):
        antigo = self.state_machine.state
        self.state_machine.state = State.FAILED
        self.state_data.failure_reason = motivo
        self.state_data.state = State.FAILED.value
        self.executor.save_state_atomic(self.state_data)
        self.on_event("state_changed", {"old": antigo.value, "new": State.FAILED.value})
    
    def _lint_teste(self, tool: str, args: Dict[str, Any]) -> Optional[str]:
        if tool not in {"write_file", "append_file"}: return None
        path = str(args.get("path", ""))
        existente = ""
        if tool == "append_file":
            try:
                existente = resolve_in_root(self.workspace, path).read_text(encoding="utf-8") + "\n"
            except Exception:
                existente = ""
        return verificar_nomes_indefinidos_em_teste(self.workspace, path, str(args.get("content", "")), existente)


    def _checar_teste_protegido(self, tool: str, args: Dict[str, Any]) -> Optional[str]:
        if tool not in WRITE_TOOLS or not self._testes_protegidos: return None
        p = posixpath.normpath(str(args.get("path", "")).replace("\\", "/"))
        if p in self._testes_protegidos:
            return (f"'{p}' é um teste fornecido pela tarefa e é somente leitura. "
                    "Coloque os testes exigidos pela SPEC em um arquivo NOVO, por exemplo 'tests/test_spec_ac.py', "
                    "como funções 'def test_ac_...' no nível do módulo.")
        return None
    # ---------- parse e prompt ----------
    def _parse_action_response(self, response_text: str, finish_reason: Optional[str] = None) -> Tuple[Optional[ActionRequest], Optional[str]]:
        """
        Extrai e valida o JSON de acao retornado pelo LLM (camada 2 ativa sempre),com fallback defensivo para o parser legado caso necessário.
        """

        avaliacao = interpretar_acao(response_text or "", finish_reason=finish_reason)
        if avaliacao.ok and avaliacao.action is not None:
            return ActionRequest(tool=avaliacao.action.tool, args=avaliacao.action.args), None

        try:
            data = parse_acao_json(response_text, list(ALLOWED_TOOLS))
            tool = data.get("tool") or data.get("action")
            args = data.get("args", {})

            if not tool: return None, "O campo obrigatório 'tool' está ausente na ação."
            if not isinstance(args, dict): return None, "O campo 'args' deve ser um dicionário."

            return ActionRequest(tool=tool, args=args), None

        except ParseError as pe:
            if avaliacao.diagnostic is not None: return None, avaliacao.diagnostic.format_retry_message()
            return None, str(pe)
        except Exception as e:
            if avaliacao.diagnostic is not None: return None, avaliacao.diagnostic.format_retry_message()
            return None, f"Erro inesperado ao processar resposta: {str(e)}"

    def _montar_prompt_execucao(self, spec_content: str, plan_content: str, ultima_observacao: str) -> str:
        
        if self.state_machine.state == State.REPAIRING:
            testes = []
            d = self.workspace / "tests"
            if d.is_dir():
                for p in sorted(d.rglob("test_*.py"))[:3]:
                    if p.stat().st_size <= 3000:
                        testes.append(p.relative_to(self.workspace).as_posix())
            
            arquivos_ctx = self.touched_files + [t for t in testes if t not in self.touched_files]
            base = self.repair_builder.montar_prompt_reparo(spec_content, arquivos_ctx, self._ultimo_pytest_output or ultima_observacao)
            
            if not ultima_observacao.startswith("Testes falharam"):
                base += f"\n\nRESULTADO DA SUA ÚLTIMA AÇÃO (não é a saída do pytest):\n{ultima_observacao[:1500]}"
            
            base += ("\n\nNOTA: o conteúdo ATUAL dos arquivos acima já está completo neste prompt. "
            "NÃO use read_file/list_files neles: grave a correção com write_file ou replace_text e chame 'done'.")
            
            return base

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
            if self.state_machine.state in {State.FAILED, State.DONE}: break
           
            self.state_data.iteration += 1

            # 1. verificação conduzida pelo harness
            if self.state_machine.state == State.VERIFYING:
                test_result = self.verifier.run_verification(self.state_machine.sequence_counter)
                self.state_data.tests_passed = test_result.tests_passed
                self._ultimo_pytest_output = test_result.output
                
                self.executor.log_action(self.state_data.iteration, "(harness_verify)", {}, None, None, {"ok": test_result.tests_passed, "output": test_result.output, "duration": test_result.duration},{}, run_id=self.run_id, prompt="", raw_response="", parsed_action=None, fsm_state="VERIFYING",)

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

            # 3. chamada da LLM (caminho unificado com response_schema quando em modo json)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_texto},
            ]
            response_schema = (
                export_normalized_envelope_schema()
                if getattr(self.llm_client, "tool_mode", "json") == "json"
                and "export_normalized_envelope_schema" in globals()
                and callable(globals()["export_normalized_envelope_schema"])
                else None
            )
            role_atual = (
                ROLE_REPAIR
                if self.state_machine.state == State.REPAIRING
                else ROLE_TOOL_SELECTION
            )
            try:
                try:
                    if self.temperature is not None:
                        resposta_llm = self.llm_client.chat(
                            messages,
                            response_schema=response_schema,
                            role=role_atual,
                            temperature=self.temperature,
                        )
                    else:
                        resposta_llm = self.llm_client.chat(
                            messages,
                            response_schema=response_schema,
                            role=role_atual,
                        )
                except TypeError:
                    try:
                        if self.temperature is not None:
                            resposta_llm = self.llm_client.chat(
                                messages,
                                response_schema=response_schema,
                                temperature=self.temperature,
                            )
                        else:
                            resposta_llm = self.llm_client.chat(
                                messages,
                                response_schema=response_schema,
                            )
                    except TypeError:
                        if self.temperature is not None:
                            resposta_llm = self.llm_client.chat(
                                messages,
                                temperature=self.temperature,
                            )
                        else: resposta_llm = self.llm_client.chat(messages)
            except Exception as exc:
                import traceback
                self.executor.log_action(self.state_data.iteration, "(llm_error)", {}, f"{type(exc).__name__}: {exc}", None, None, {},run_id=self.run_id, prompt=prompt_texto, raw_response=traceback.format_exc(),parsed_action=None, fsm_state=self.state_machine.state.value)
                self._falhar("llm_error")
                break

            texto_resposta = resposta_llm.content if hasattr(resposta_llm, "content") else str(resposta_llm)
            finish_reason = getattr(resposta_llm, "finish_reason", None)

            p_tok = getattr(resposta_llm, "prompt_tokens", 0) or 0
            c_tok = getattr(resposta_llm, "completion_tokens", 0) or 0
            tokens_turn = {"prompt_tokens": int(p_tok), "completion_tokens": int(c_tok)}
            if p_tok or c_tok:
                usage_atual = self.state_data.last_usage or {"prompt_tokens": 0, "completion_tokens": 0}
                self.state_data.last_usage = {"prompt_tokens": int(usage_atual.get("prompt_tokens", 0)) + int(p_tok),"completion_tokens": int(usage_atual.get("completion_tokens", 0)) + int(c_tok),}

            estado_atual = self.state_machine.state.value

            # 4. parse + validação pós-geração (camada 2 ativa sempre)
            avaliacao_contrato = interpretar_acao(texto_resposta or "", finish_reason=finish_reason)
            action_req, parse_error = self._parse_action_response(texto_resposta or "", finish_reason=finish_reason)
            if parse_error:
                categoria = (
                    avaliacao_contrato.diagnostic.category.value
                    if avaliacao_contrato.diagnostic is not None
                    and getattr(avaliacao_contrato.diagnostic, "category", None) is not None
                    else "parse_error"
                )

                self.state_data.parse_failures += 1
                esgotou_falhas = self.state_data.parse_failures >= MAX_PARSE_FAILURES
                self.eval_action_attempts.append({"iteration": self.state_data.iteration,"valid": False,"error_category": categoria,"retries": 0 if esgotou_falhas else 1,})
                
                self.executor.log_action(
                    self.state_data.iteration,
                    "(resposta inválida)",
                    {},
                    None,
                    None,
                    None,
                    tokens_turn,
                    run_id=self.run_id,
                    prompt=prompt_texto,
                    raw_response=texto_resposta,
                    parsed_action=None,
                    fsm_state=estado_atual,
                )
                
                self.on_event("action", {"tool": "(resposta inválida)", "args": {}, "error": parse_error})
                if esgotou_falhas:
                    self._falhar("parse_failure")
                    break
                ultima_observacao = f"Erro de Parse de Ação: {parse_error}. Por favor, retorne um JSON válido."
                continue

            self.eval_action_attempts.append({
                "iteration": self.state_data.iteration,
                "valid": bool(avaliacao_contrato.ok),
                "error_category": (
                    None
                    if avaliacao_contrato.ok
                    else (
                        avaliacao_contrato.diagnostic.category.value
                        if avaliacao_contrato.diagnostic is not None
                        and getattr(avaliacao_contrato.diagnostic, "category", None) is not None
                        else "fallback_parse"
                    )
                ),
                "retries": 0,
            })

            self.state_data.parse_failures = 0
            tool, args = action_req.tool, action_req.args
            acao_parseada = {"tool": tool, "args": args}

            # 5. validador
            erro = self.validator.validate(tool, args) or self._lint_teste(tool, args) or self._checar_teste_protegido(tool, args)
            if erro:

                self.executor.log_action(
                    self.state_data.iteration,
                    tool,
                    args,
                    erro,
                    None,
                    None,
                    tokens_turn,
                    run_id=self.run_id,
                    prompt=prompt_texto,
                    raw_response=texto_resposta,
                    parsed_action=acao_parseada,
                    fsm_state=estado_atual,
                )

                self.on_event("action", {"tool": tool, "args": args, "error": erro})
                ultima_observacao = f"Ação rejeitada pelo validador: {erro}"
                continue

            # 6. hard gates
            erro = self.gates.check_gate(tool, args)
            if erro:
                
                self.executor.log_action(
                    self.state_data.iteration,
                    tool,
                    args,
                    None,
                    erro,
                    None,
                    tokens_turn,
                    run_id=self.run_id,
                    prompt=prompt_texto,
                    raw_response=texto_resposta,
                    parsed_action=acao_parseada,
                    fsm_state=estado_atual,
                )

                self.on_event("action", {"tool": tool, "args": args, "error": erro})
                ultima_observacao = (f"Aviso do Gate: {erro}" if tool == "done"else f"Ação bloqueada pelo Hard Gate: {erro}")
                
                continue

            # 6.5. anti-loop de renomeação cosmética de testes em REPAIRING
            if self.state_machine.state == State.REPAIRING and tool in {"replace_text", "write_file"}:
                alvo_path = str(args.get("path", "")).replace("\\", "/")
                if alvo_path.startswith("tests/") or alvo_path == "test_task.py":
                    old_s = str(args.get("old_str", ""))
                    new_s = str(args.get("new_str", ""))
                    parece_renomeacao_teste = (("def test_" in old_s and "def test_" in new_s)or ("test_ac_" in old_s and "test_ac_" in new_s))
                    
                    if parece_renomeacao_teste:
                        assinatura_rename = f"{alvo_path}:{old_s.strip()}->{new_s.strip()}"
                        historico_renames = getattr(self, "_historico_renames_teste", set())
                        par_inverso = f"{alvo_path}:{new_s.strip()}->{old_s.strip()}"
                        if assinatura_rename in historico_renames or par_inverso in historico_renames:
                            erro_loop = (
                                "Bloqueado pelo anti-loop: renomear repetidamente funções de teste em "
                                "REPAIRING não corrige a falha. Altere a implementação em 'src/' ou corrija "
                                "a lógica/asserção do teste sem ficar alternando nomes de função."
                            )

                            self.executor.log_action(
                                self.state_data.iteration,
                                tool,
                                args,
                                None,
                                erro_loop,
                                None,
                                tokens_turn,
                                run_id=self.run_id,
                                prompt=prompt_texto,
                                raw_response=texto_resposta,
                                parsed_action=acao_parseada,
                                fsm_state=estado_atual,
                            )

                            self.on_event("action", {"tool": tool, "args": args, "error": erro_loop})
                            ultima_observacao = f"Ação bloqueada: {erro_loop}"
                            continue

                        historico_renames.add(assinatura_rename)
                        self._historico_renames_teste = historico_renames
                        
            # 6.6. anti-loop de leitura/escrita repetitiva sem avanço
            erro_loop_geral = None
            if self.state_machine.state == State.REPAIRING and tool in {"read_file", "list_files"}:
                alvo_obs = str(args.get("path") or args.get("directory") or args.get("dir") or ".").replace("\\", "/")
                chave_obs = f"{self.state_machine.repair_attempts}:{tool}:{alvo_obs}"
                cont_obs = getattr(self, "_cont_obs_repairing", {})
                cont_obs[chave_obs] = cont_obs.get(chave_obs, 0) + 1
                self._cont_obs_repairing = cont_obs
                if cont_obs[chave_obs] > 2:
                    erro_loop_geral = (
                        f"Bloqueado pelo anti-loop: você já executou '{tool}' em '{alvo_obs}' "
                        f"{cont_obs[chave_obs]} vezes neste ciclo de REPAIRING. Pare de reler/listar "
                        "e aplique a correção em 'src/' (ou 'tests/') e depois chame 'done'."
                    )

            if self.state_machine.state == State.IMPLEMENTING and tool in {"write_file", "append_file", "replace_text"}:
                writes_impl = getattr(self, "_writes_implementing_sem_done", 0) + 1
                self._writes_implementing_sem_done = writes_impl
                if writes_impl > 4:
                    erro_loop_geral = (
                        f"Bloqueado pelo anti-loop: você já realizou {writes_impl} operações de escrita "
                        "em IMPLEMENTING sem verificar. Chame 'done' agora para executar os testes e validar o código."
                    )
            elif tool == "done": self._writes_implementing_sem_done = 0

            if erro_loop_geral:

                self.executor.log_action(
                    self.state_data.iteration,
                    tool,
                    args,
                    None,
                    erro_loop_geral,
                    None,
                    tokens_turn,
                    run_id=self.run_id,
                    prompt=prompt_texto,
                    raw_response=texto_resposta,
                    parsed_action=acao_parseada,
                    fsm_state=estado_atual,
                )

                self.on_event("action", {"tool": tool, "args": args, "error": erro_loop_geral})
                ultima_observacao = f"Ação bloqueada: {erro_loop_geral}"
                self._blocos_antiloop_seguidos += 1
                if self._blocos_antiloop_seguidos >= 3:
                    self._blocos_antiloop_seguidos = 0
                    # força re-teste e consome tentativa de reparo
                    self._mudar_estado(State.VERIFYING)
                
                continue

            # 7. 'run_tests' é exclusivo do harness
            if tool == "run_tests":
                erro = "'run_tests' é executado automaticamente pelo harness. Quando a implementação estiver completa, use 'done'."
                self.executor.log_action(
                    self.state_data.iteration,
                    tool,
                    args,
                    None,
                    erro,
                    None,
                    tokens_turn,
                    run_id=self.run_id,
                    prompt=prompt_texto,
                    raw_response=texto_resposta,
                    parsed_action=acao_parseada,
                    fsm_state=estado_atual,
                )
                self.on_event("action", {"tool": tool, "args": args, "error": erro})
                ultima_observacao = f"Ação rejeitada: {erro}"
                continue

            # 8. 'done' dispara a verificação
            if tool == "done":
                self.executor.log_action(
                    self.state_data.iteration,
                    "done",
                    args,
                    None,
                    None,
                    {"ok": True, "output": "", "duration": 0.0},
                    tokens_turn,
                    run_id=self.run_id,
                    prompt=prompt_texto,
                    raw_response=texto_resposta,
                    parsed_action=acao_parseada,
                    fsm_state=estado_atual,
                )
                self.on_event("action", {"tool": "done", "args": {}, "error": None})
                self._mudar_estado(State.VERIFYING)
                continue

            # 9. execução de ferramenta
            result = self.executor.execute(tool, args)
            self._blocos_antiloop_seguidos = 0

            if tool in WRITE_TOOLS and result["ok"]:
                p = args.get("path")
                if p and p not in self.touched_files: self.touched_files.append(p)

            self.gates.update_flags_after_execution(tool, result)
            self.executor.log_action(
                self.state_data.iteration,
                tool,
                args,
                None,
                None,
                result,
                tokens_turn,
                run_id=self.run_id,
                prompt=prompt_texto,
                raw_response=texto_resposta,
                parsed_action=acao_parseada,
                fsm_state=estado_atual,
            )
            self.on_event("action", {
                "tool": tool,
                "args": args,
                "error": None if result["ok"] else (result.get("error") or result["output"]),
            })

            self.action_history.append({"tool": tool, "args": args, "ok": result["ok"], "output": result["output"]})
            ultima_observacao = f"Ferramenta '{tool}' executada. Status Ok: {result['ok']}.\nSaída:\n{result['output']}"


        if self.state_machine.state == State.VERIFYING:
            test_result = self.verifier.run_verification(self.state_machine.sequence_counter)
            self._ultimo_pytest_output = test_result.output
            self.state_data.tests_passed = test_result.tests_passed
            self.state_machine.record_test(passed=test_result.tests_passed)
            self.executor.log_action(
                self.state_data.iteration, "(harness_verify)", {}, None, None,
                {"ok": test_result.tests_passed, "output": test_result.output, "duration": test_result.duration},
                {}, run_id=self.run_id, prompt="", raw_response="", parsed_action=None, fsm_state="VERIFYING",
            )
            if test_result.tests_passed and self.state_machine.validate_done_gate():
                self._mudar_estado(State.DONE)
            else: self._falhar("max_iterations")

        # estouro de iterações
        if (self.state_data.iteration >= self.state_data.max_iterations
                and self.state_machine.state not in {State.VERIFYING, State.DONE, State.FAILED}):
            self._falhar("max_iterations")

        return self.state_machine.state