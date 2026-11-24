## Objective
Implementar a classe KeyValueStore com métodos set(k, v), get(k, default=None), delete(k) e keys() (ordenado)

## Parameters
- set(k, v)
- get(k, default=None)
- delete(k)
- keys()

## Acceptance Criteria
- **AC-1**: A chave 'b' deve ser adicionada com valor 2 usando o método set
- **AC-2**: A chave 'a' deve ser adicionada com valor 1 usando o método set
- **AC-3**: O método keys() deve retornar uma lista ordenada ['a', 'b']
- **AC-4**: Após deletar a chave 'a', o método get('a') deve retornar None
- **AC-5**: O arquivo tests/test_task.py deve passar nos testes

## Tests
- **AC-1**: test_ac_1_add_b_with_value_2
- **AC-2**: test_ac_2_add_a_with_value_1
- **AC-3**: test_ac_3_keys_return_sorted_list_ab
- **AC-4**: test_ac_4_get_a_returns_none_after_deletion
- **AC-5**: test_ac_5_test_file_passes_tests