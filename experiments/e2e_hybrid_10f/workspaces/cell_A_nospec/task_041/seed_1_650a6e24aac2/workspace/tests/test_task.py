from src.structures import KeyValueStore

def test_keyvaluestore():
    cls = KeyValueStore
    kv = cls(); kv.set('b', 2); kv.set('a', 1); assert kv.keys() == ['a', 'b']; kv.delete('a'); assert kv.get('a') is None
