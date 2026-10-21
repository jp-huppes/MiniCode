def truncate(s: str, max_chars: int = 4000, head_ratio: float = 0.2) -> str:
    """
    Trunca uma string longa mantendo uma fracao inicial (head) e a parte final (tail), inserindo um marcador informando quantos caracteres foram omitidos.
    """
    if not s or len(s) <= max_chars: return s or ""

    head_len = int(max_chars * head_ratio)
    # reserva espaco para o marcador de corte
    marker = f"\n\n... [Trecho omitido: {len(s) - max_chars} caracteres] ...\n\n"
    tail_len = max_chars - head_len - len(marker)

    if tail_len < 0:
        tail_len = max_chars // 2
        head_len = max_chars - tail_len - len(marker)

    head = s[:head_len]
    tail = s[-tail_len:] if tail_len > 0 else ""

    return head + marker + tail