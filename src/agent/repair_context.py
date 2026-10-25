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
    # Constroi o contexto de reparo estruturado para o LLM
    def __init__(self, workspace_dir: Path):
        self.workspace = workspace_dir
        self.tentativas_historico: List[Dict[str, Any]] = []

    def registrar_tentativa(self, attempt_id: int, arquivos_tocados: List[str], pytest_output: str, ok: bool):
        assinatura = calcular_assinatura_erro(pytest_output)
        
        # extrai resumo de 1-2 linhas do erro
        linhas = [l for l in pytest_output.splitlines() if l.strip()]
        erro_resumo = linhas[-1] if linhas else "Falha desconhecida nos testes."

        self.tentativas_historico.append({
            "attempt_id": attempt_id,
            "files": arquivos_tocados,
            "error_summary": erro_resumo,
            "signature": assinatura,
            "passed": ok
        })

    def montar_prompt_reparo(self, spec_content: str, arquivos_tocados: List[str], pytest_output: str, max_tokens: int = 6000) -> str:
        # Monta o prompt de reparo estruturado com os blocos fixos exigidos
            # 1. leitura atualizada dos arquivos tocados do disco
        bloco_codigo = "ARQUIVOS TOCADOS (Versão Atual):\n"
        for arq in set(arquivos_tocados):
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
            aviso_repeticao = "\n⚠️ ATENÇÃO: Este mesmo erro já ocorreu em tentativas anteriores. Mude radicalmente a sua abordagem de correção!\n"

        for t in self.tentativas_historico[-3:]:  # ultimas 3
            bloco_hist += f"- Tentativa {t['attempt_id']}: Erro: {t['error_summary']} (Arquivos: {t['files']})\n"

        prompt = f"""[MODO DE REPARO / CORREÇÃO DE FALHA]
        {aviso_repeticao}

        SPECIFICATION (specification.md):
        {spec_content}

        {bloco_codigo}

        SAÍDA DO PYTEST (FALHA):
        {pytest_truncado}

        {bloco_hist}

        INSTRUÇÃO: Corrija a causa raiz do erro acima. Não altere arquivos na pasta 'tests/'. Responda estritamente no formato de ação JSON.
        """
        return prompt