from src.structures import PairMap

def test_pairmap():
    cls = PairMap
    pm = cls(); pm.put_pair('u1', 't1'); assert pm.get_forward('u1') == 't1' and pm.get_backward('t1') == 'u1'
