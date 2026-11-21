from src.pure_ops import slugify

def test_slugify():
    fn = slugify
    assert fn('Hello, World 2026!') == 'hello-world-2026'
