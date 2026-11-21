from src.pure_ops import mean_list

def test_mean_list():
    fn = mean_list
    assert abs(fn([2, 4, 6]) - 4.0) < 1e-9 and abs(fn([10]) - 10.0) < 1e-9
