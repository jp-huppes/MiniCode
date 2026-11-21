from src.structures import UndoBuffer

def test_undobuffer():
    cls = UndoBuffer
    u = cls('init'); u.set_value('v1'); u.set_value('v2'); assert u.current() == 'v2'; u.undo(); assert u.current() == 'v1'
