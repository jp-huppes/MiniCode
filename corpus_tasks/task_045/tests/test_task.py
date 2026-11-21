from src.structures import StatsAccumulator

def test_statsaccumulator():
    cls = StatsAccumulator
    sa = cls(); sa.add(5); sa.add(15); assert sa.count() == 2 and sa.total() == 20
