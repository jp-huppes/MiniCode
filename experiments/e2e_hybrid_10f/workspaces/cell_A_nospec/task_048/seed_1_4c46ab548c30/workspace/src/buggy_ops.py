def contar_ocorrencias(items, alvo):
    return alvo in items and sum(x == alvo for x in items)