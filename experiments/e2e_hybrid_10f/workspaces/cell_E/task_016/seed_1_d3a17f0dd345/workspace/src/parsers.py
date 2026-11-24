def parse_kv_line(line):
    kv_pairs = {}
    tokens = line.split()
    for token in tokens:
        if '=' in token:
            key, value = token.split('=')
            kv_pairs[key] = value
    return kv_pairs