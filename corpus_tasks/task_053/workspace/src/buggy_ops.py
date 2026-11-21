def limitar_tamanho(texto, n):
    if len(texto) < n:
        return texto[:n]
    return texto
