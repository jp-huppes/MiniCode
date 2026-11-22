import os, tempfile, pathlib
import shutil
from src.tools.registry import ToolRegistry
os.environ["PYTHONPATH"] = str(pathlib.Path.cwd())
ws = pathlib.Path("corpus_probe_smoke") / "ws"
shutil.rmtree(ws.parent, ignore_errors=True); ws.mkdir(parents=True)
(ws/"tests").mkdir(); (ws/"src").mkdir()
(ws/"tests/test_task.py").write_text("from src.pure_ops import clamp\ndef test_c(): assert clamp(5,0,3)==3\n")
r = ToolRegistry(ws)
print("SEM arquivo:\n", r.dispatch("run_tests", {"command": "pytest -x -q"}).output[:300])
(ws/"src/pure_ops.py").write_text("def clamp(x,a,b): return max(a,min(b,x))\n")
print("COM arquivo:", r.dispatch("run_tests", {"command": "pytest -x -q"}).meta)