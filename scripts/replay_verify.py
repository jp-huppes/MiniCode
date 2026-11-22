import sys, pathlib
from src.tools.registry import ToolRegistry
ws = pathlib.Path(sys.argv[1])
r = ToolRegistry(ws).dispatch("run_tests", {"command": "pytest -x --tb=short -q"})
print(r.meta)
print(r.output[:1800])