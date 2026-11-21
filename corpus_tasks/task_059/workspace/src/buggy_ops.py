def indice_alvo(items, alvo):
    for i, it in enumerate(items):
        if it == alvo:
            return i + 1
    return -1
