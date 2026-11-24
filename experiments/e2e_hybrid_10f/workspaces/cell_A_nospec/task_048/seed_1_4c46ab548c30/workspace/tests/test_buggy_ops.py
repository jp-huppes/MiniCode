import unittest
from src.buggy_ops import contar_ocorrencias

class TestContarOcorrencias(unittest.TestCase):
    def test_contar_ocorrencias(self):
        self.assertEqual(contar_ocorrencias([1, 2, 3, 2, 1], 1), 2)
        self.assertEqual(contar_ocorrencias([1, 2, 3, 2, 1], 2), 2)
        self.assertEqual(contar_ocorrencias([1, 2, 3, 2, 1], 3), 1)
        self.assertEqual(contar_ocorrencias([], 1), 0)
