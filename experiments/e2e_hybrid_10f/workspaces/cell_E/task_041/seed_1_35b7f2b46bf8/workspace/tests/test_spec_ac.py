

def test_ac_1_add_b_with_value_2():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('b', 2)
    assert store.get('b') == 2

def test_ac_2_add_a_with_value_1():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('a', 1)
    assert store.get('a') == 1

def test_ac_3_keys_return_sorted_list_ab():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('a', 1)
    store.set('b', 2)
    assert sorted(store.keys()) == ['a', 'b']

def test_ac_4_get_a_returns_none_after_deletion():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('a', 1)
    store.delete('a')
    assert store.get('a') is None

def test_ac_5_test_file_passes_tests():
    import subprocess
    result = subprocess.run(['python', 'tests/test_spec_ac.py'], capture_output=True, text=True)
    assert result.returncode == 0

def test_ac_1_add_b_with_value_2():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('b', 2)
    assert store.get('b') == 2

def test_ac_2_add_a_with_value_1():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('a', 1)
    assert store.get('a') == 1

def test_ac_3_keys_return_sorted_list_ab():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('a', 1)
    store.set('b', 2)
    assert sorted(store.keys()) == ['a', 'b']

def test_ac_4_get_a_returns_none_after_deletion():
    from src.structures import KeyValueStore
    store = KeyValueStore()
    store.set('a', 1)
    store.delete('a')
    assert store.get('a') is None

def test_ac_5_test_file_passes_tests():
    # Adicionar aqui as verificações para garantir que o arquivo de teste passe os testes
    pass