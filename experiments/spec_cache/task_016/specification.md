## Objective
Implementar a função parse_kv_line que converte uma string em um dicionário

## Parameters
- line

## Acceptance Criteria
- **AC-1**: A função deve converter corretamente uma string com pares chave-valor separados por espaços em branco em um dicionário Python
- **AC-2**: A função deve ignorar tokens que não contêm '='
- **AC-3**: A função deve retornar um dicionário vazio se a entrada for uma string vazia

## Tests
- **AC-1**: test_ac_1_converts_string_to_dict
- **AC-2**: test_ac_2_ignores_tokens_without_equal_sign
- **AC-3**: test_ac_3_returns_empty_dict_for_empty_string