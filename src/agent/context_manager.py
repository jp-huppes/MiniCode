import re
from typing import List, Dict, Any

def estimar_tokens(texto: str, chars_por_token: float = 3.5) -> int:
    # Estima o num de tokens de um txt com base em uma taxa conservadora de caracteres
    if not texto: return 0
    return int(len(texto) / chars_por_token)

def truncar_pytest_output(output: str, max_tokens: int = 1500) -> str:
    # Trunca a saida do pytest preservando o resumo final e o blocos:hort test summary info
    if estimar_tokens(output) <= max_tokens: return output

    linhas = output.splitlines(); short_summary_idx = -1
    
    for i, linha in enumerate(linhas):
        if "short test summary info" in linha.lower() or "=== short test summary info ===" in linha:
            short_summary_idx = i
            break

    resumo_final = "\n".join(linhas[-10:]) if len(linhas) >= 10 else "\n".join(linhas)
    short_summary_text = ""
    if short_summary_idx != -1:
        short_summary_text = "\n".join(linhas[short_summary_idx:min(short_summary_idx + 20, len(linhas))])

    resultado_truncado = f"[Saída longa de testes truncada por orçamento de contexto]\n\n{short_summary_text}\n\n... [detalhes intermediários omitidos] ...\n\n{resumo_final}"
    
    return resultado_truncado

def formatar_historico_reparos(tentativas: List[Dict[str, Any]], max_retidas: int = 3) -> str:
    # Comprime o historico de tentativas do repair loop mantendo apenas as mais recentes
    if not tentativas: return "Nenhuma tentativa anterior registrada nesta fase."

    retidas = tentativas[-max_retidas:]
    linhas = [f"Histórico recente de tentativas (Total acumulado: {len(tentativas)}):"]
    
    for idx, t in enumerate(retidas, start=max(1, len(tentativas) - max_retidas + 1)):
        linhas.append(f"- Tentativa {idx}: Ação={t.get('tool')}, Sucesso={t.get('ok')}, Saída/Erro={str(t.get('output', ''))[:150]}")
        
    return "\n".join(linhas)