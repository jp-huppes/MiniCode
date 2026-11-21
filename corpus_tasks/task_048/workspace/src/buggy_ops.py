def contar_ocorrencias(items, alvo):
    qtd = 0
    for it in items:
        if it != alvo:
            qtd += 1
    return qtd
