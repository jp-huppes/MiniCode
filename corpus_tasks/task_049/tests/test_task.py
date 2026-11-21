from src.buggy_ops import remover_duplicatas_ordem

def test_remover_duplicatas_ordem():
    fn = remover_duplicatas_ordem
    assert fn([3, 1, 3, 2, 1]) == [3, 1, 2]
