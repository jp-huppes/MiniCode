import re
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.agent.context_manager import estimar_tokens, truncar_pytest_output

def calcular_assinatura_erro(pytest_output: str) -> str:
    # Calcula uma assinatura hash unica para o erro baseado no traceback e ultima linha
    linhas = [l.strip() for l in pytest_output.splitlines() if l.strip()]
    
    # filtra linhas de erro ou falha relevantes
    relevantes = [l for l in linhas if "FAILED" in l or "ERROR" in l or "AssertionError" in l or l.startswith("E ")]
    conteudo_assinatura = " ".join(relevantes[-3:]) if relevantes else "erro_desconhecido"
    
    return hashlib.md5(conteudo_assinatura.encode("utf-8")).hexdigest()[:10]

class RepairContextBuilder:
    """Constrói o contexto de reparo estruturado para o LLM."""

    def __init__(self, workspace_dir: Path):
        self.workspace = workspace_dir
        self.tentativas_historico: List[Dict[str, Any]] = []

    @staticmethod
    def _resumo_erro(output: str) -> str:
        linhas = [l.strip() for l in output.splitlines() if l.strip()]
        for prefixos in (("[HARNESS]",), ("FAILED ", "ERROR "), ("E ",)):
            for l in reversed(linhas):
                if l.startswith(prefixos): return l[:200]
        
        uteis = [l for l in linhas if l not in {"STDOUT:", "STDERR:"}]
        return (uteis[-1] if uteis else "Falha desconhecida nos testes.")[:200]

    def registrar_tentativa(self, attempt_id: int, arquivos_tocados: List[str], pytest_output: str, ok: bool):
        assinatura = calcular_assinatura_erro(pytest_output)
        erro_resumo = self._resumo_erro(pytest_output)

        self.tentativas_historico.append({
            "attempt_id": attempt_id,
            "files": arquivos_tocados,
            "error_summary": erro_resumo,
            "signature": assinatura,
            "passed": ok,
        })

    def montar_prompt_reparo(self, spec_content: str, arquivos_tocados: List[str], pytest_output: str, max_tokens: int = 6000, ) -> str:
        # 1. leitura atualizada dos arquivos tocados do disco
        bloco_codigo = "ARQUIVOS TOCADOS (Versão Atual):\n"
        for arq in dict.fromkeys(arquivos_tocados):
            caminho_arq = self.workspace / arq
            if caminho_arq.is_file():
                conteudo = caminho_arq.read_text(encoding="utf-8")
                bloco_codigo += f"\n--- {arq} ---\n{conteudo}\n"

        # 2. saida do pytest truncada
        pytest_truncado = truncar_pytest_output(pytest_output, max_tokens=1500)

        # 3. historico de tentativas e detecção de repetição
        bloco_hist = "HISTÓRICO DE TENTATIVAS ANTERIORES:\n"
        ultima_assinatura = self.tentativas_historico[-1]["signature"] if self.tentativas_historico else ""

        repeticoes = sum(1 for t in self.tentativas_historico[:-1] if t["signature"] == ultima_assinatura)
        aviso_repeticao = ""
        
        if repeticoes > 0:
            aviso_repeticao = (
                "\n⚠️ ATENÇÃO: Este mesmo erro já ocorreu em tentativas anteriores. "
                "Mude radicalmente a sua abordagem de correção!\n"
            )

        for t in self.tentativas_historico[-3:]:
            bloco_hist += (
                f"- Tentativa {t['attempt_id']}: Erro: {t['error_summary']} "
                f"(Arquivos: {t['files']})\n"
            )

        prompt = f"""[MODO DE REPARO / CORREÇÃO DE FALHA]
{aviso_repeticao}
SPECIFICATION (specification.md):
{spec_content}

{bloco_codigo}
SAÍDA DO PYTEST (FALHA):
{pytest_truncado}

{bloco_hist}
INSTRUÇÃO: Corrija a causa raiz da falha no código de implementação em 'src/' (ou na lógica do teste quando o teste estiver errado). NÃO fique renomeando funções de teste existentes (como alternar entre nomes curtos e descritivos) a menos que o pytest reporte explicitamente que um nome específico não foi encontrado ou que há cobertura incompleta exigida pela SPEC. Responda estritamente no formato de ação JSON."""

        return prompt