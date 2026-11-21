from src.buggy_ops import media_segura

def test_media_segura():
    fn = media_segura
    assert abs(fn([1, 2]) - 1.5) < 1e-9 and fn([]) == 0.0
