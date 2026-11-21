from src.structures import Queue

def test_queue():
    cls = Queue
    q = cls(); q.enqueue('a'); q.enqueue('b'); assert q.size() == 2; assert q.dequeue() == 'a'; assert q.dequeue() == 'b'; assert q.dequeue() is None
