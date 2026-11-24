def parse_kv_line(line):
    key_value_pairs = {}
    pairs = line.split()
    for pair in pairs:
        key, value = pair.split('=')
        key_value_pairs[key] = value
    return key_value_pairs