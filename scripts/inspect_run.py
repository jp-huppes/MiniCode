import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8")
b = pathlib.Path(sys.argv[1])
st = json.loads((b / ".minicode" / "state.json").read_text(encoding="utf-8"))
print("failure_reason:", st["failure_reason"], "| iteration:", st["iteration"], "| state:", st["state"])
for linha in (b / ".minicode" / "actions.jsonl").read_text(encoding="utf-8").splitlines():
    j = json.loads(linha)
    err = j.get("validator_error") or j.get("gate_error") or ""
    print(f'{j["iteration"]:>2} {str(j.get("fsm_state")):<12} {j["tool"]:<18} {err[:110]}')
    if j["tool"] == "(harness_verify)":
        out = (j.get("result") or {}).get("output", "")
        print("    >> " + "\n    >> ".join(out.strip().splitlines()[:18]))