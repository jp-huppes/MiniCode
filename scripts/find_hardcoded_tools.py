import ast
from pathlib import Path

TOOL_NAMES = {
    "list_files",
    "read_file",
    "write_file",
    "append_file",
    "replace_text",
    "run_tests",
    "done",
}

CONTAINER_NODES = (ast.List, ast.Set, ast.Tuple)


def scan_file(path: Path) -> list[tuple[int, str, list[str]]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[tuple[int, str, list[str]]] = []

    for node in ast.walk(tree):
        if isinstance(node, CONTAINER_NODES):
            matched = sorted({
                el.value
                for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str) and el.value in TOOL_NAMES
            })
            if len(matched) >= 2:
                line = ast.get_source_segment(source, node) or ""
                single_line = " ".join(line.split())
                findings.append((node.lineno, single_line[:120], matched))

    return findings


def main(root: Path = Path("src")) -> None:
    if not root.exists():
        print(f"Diretório '{root}' não encontrado.")
        return

    total = 0
    for py_file in sorted(root.rglob("*.py")):
        matches = scan_file(py_file)
        for lineno, snippet, matched in matches:
            total += 1
            print(f"{py_file}:{lineno} -> {matched} | {snippet}")

    print(f"\nTotal de ocorrências encontradas: {total}")


if __name__ == "__main__":
    main()