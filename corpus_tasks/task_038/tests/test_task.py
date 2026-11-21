from src.structures import RingBuffer

def test_ringbuffer():
    cls = RingBuffer
    r = cls(3); r.write(1); r.write(2); r.write(3); r.write(4); assert r.read_all() == [2, 3, 4]
