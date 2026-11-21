def remover_duplicatas_ordem(items):
    vistos = set()
    out = []
    for it in items:
        if it in vistos:
            out.append(it)
        vistos.add(it)
    return out
