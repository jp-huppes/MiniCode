from pathlib import Path
from typing import Tuple
from src.llm.client import LLMClient
from src.agent.state import AgentStateMachine, State
from src.spec.context import coletar_contexto
from src.spec.generator import gerar_spec_com_retry, gerar_plano_com_retry
from src.spec.persistence import salvar_artefato_spec
from src.spec.models import Spec

def executar_spec_pipeline(tarefa: str, workspace_dir: str | Path, llm_client: LLMClient,  state_machine: AgentStateMachine) -> Tuple[Spec, str]:
    """
    Orquestra o pipeline completo de Spec-Driven Development:
        1. Coleta o contexto do projeto.
        2. Gera e valida a especificacao (SPEC) com retry.
        3. Persiste 'specification.md' no workspace e transita o agente para SPECIFIED.
        4. Gera e valida o Plano de execucao (PLAN) com retry.
        5. Persiste 'plan.md' no workspace e transita o agente para PLANNED.
    """
    root = Path(workspace_dir).resolve()
    
        # 1. coletar visao/contexto do projeto 
    contexto = coletar_contexto(root)
    
        # 2. gerar e validar a SPEC com loop de retry
    spec, spec_raw = gerar_spec_com_retry(tarefa, contexto, llm_client)
    
        # 3. persistirspecification.md e avançar estado para SPECIFIED
    salvar_artefato_spec(root, "specification.md", spec_raw, tarefa)
    
    if state_machine.state == State.RECEIVED: state_machine.transition(State.SPECIFIED)

    # 4. gerar e validar o PLAN com base na SPEC e contexto
    plano_raw = gerar_plano_com_retry(spec, contexto, llm_client)
    
    # 5. persistir plan.md e avançar estado para PLANNED
    salvar_artefato_spec(root, "plan.md", plano_raw, tarefa)
    
    if state_machine.state == State.SPECIFIED: state_machine.transition(State.PLANNED)

    return spec, plano_raw