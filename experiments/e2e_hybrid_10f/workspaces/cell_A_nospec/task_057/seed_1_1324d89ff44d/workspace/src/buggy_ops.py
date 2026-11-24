def join_non_empty(parts, sep=','):
    return sep.join([part for part in parts if part])


def join_non_empty_strings(strings):
    result = ''
    for string in strings:
        if string.strip():
            result += ', '.join([string.strip()])
    return result