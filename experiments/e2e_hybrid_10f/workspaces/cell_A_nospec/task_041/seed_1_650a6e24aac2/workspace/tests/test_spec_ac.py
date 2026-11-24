import unittest
from src.structures import KeyValueStore
class TestKeyValueStore(unittest.TestCase):
    def setUp(self):
        self.store = KeyValueStore()

    def test_set_get(self):
        self.store.set('key1', 'value1')
        self.assertEqual(self.store.get('key1'), 'value1')

    def test_delete(self):
        self.store.set('key2', 'value2')
        self.store.delete('key2')
        self.assertIsNone(self.store.get('key2'))

    def test_keys(self):
        self.store.set('key3', 'value3')
        self.assertIn('key3', self.store.keys())
if __name__ == '__main__':
    unittest.main()
    def test_keys_order_independent(self):
        self.store.set('b', 2)
        self.store.set('a', 1)
        keys = self.store.keys()
        self.assertIn('a', keys)
        self.assertIn('b', keys)
    def test_keys_order_independent(self):
        self.store.set('b', 2)
        self.store.set('a', 1)
        keys = self.store.keys()
        self.assertIn('a', keys)
        self.assertIn('b', keys)