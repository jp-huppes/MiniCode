import sys
import tempfile
from pathlib import Path

# Garante que a raiz do repositório esteja no sys.path para importar src.*
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.validator import ActionValidator
from src.llm.action_parser import parse_acao_json

try:
    from src.llm.errors import LLMFormatError, ParseError
except ImportError:
    from src.llm.errors import LLMFormatError
    from src.llm.action_parser import ParseError

from src.tools.catalog import ALLOWED_TOOLS

CORPUS = [
    {
        "id": "C1",
        "categoria": "file_path no lugar de path",
        "input": '{"tool": "write_file", "args": {"file_path": "main.py", "content": "print()"}}',
    },
    {
        "id": "C2",
        "categoria": "Campo extra em args",
        "input": '{"tool": "read_file", "args": {"path": "main.py", "extra_field": 123}}',
    },
    {
        "id": "C3",
        "categoria": "Tipo errado (content numérico)",
        "input": '{"tool": "write_file", "args": {"path": "main.py", "content": 12345}}',
    },
    {
        "id": "C4",
        "categoria": "Ferramenta inexistente",
        "input": '{"tool": "hack_system", "args": {"path": "main.py"}}',
    },
    {
        "id": "C5",
        "categoria": "JSON truncado",
        "input": '{"tool": "write_file", "args": {"path": "main.py"',
    },
    {
        "id": "C6",
        "categoria": "Chave action no lugar de tool",
        "input": '{"action": "write_file", "args": {"path": "main.py", "content": "ok"}}',
    },
    {
        "id": "C7",
        "categoria": "Args de outra ferramenta (faltando path)",
        "input": '{"tool": "read_file", "args": {"content": "olá"}}',
    },
    {
        "id": "C8",
        "categoria": "Texto livre",
        "input": "Aqui está o seu arquivo modificado com sucesso.",
    },
    {
        "id": "C9",
        "categoria": "Dois objetos JSON na mesma resposta",
        "input": '{"tool": "read_file", "args": {"path": "a.py"}} {"tool": "read_file", "args": {"path": "b.py"}}',
    },
    {
        "id": "C10",
        "categoria": "Cerca markdown sem crases com prefixo json",
        "input": 'json\n{"tool": "read_file", "args": {"path": "main.py"}}',
    },
]


def avaliar_corpus():
    print(f"{'ID':<4} | {'Categoria':<38} | {'Desfecho no Pipeline Atual'}")
    print("-" * 82)

    with tempfile.TemporaryDirectory() as tmp_dir:
        validator = ActionValidator(Path(tmp_dir))
        ferramentas = list(ALLOWED_TOOLS)

        for item in CORPUS:
            cid = item["id"]
            cat = item["categoria"]
            texto = item["input"]

            try:
                parsed = parse_acao_json(texto, ferramentas)
                tool = parsed.get("tool")
                args = parsed.get("args", {})

                val_result = validator.validate(tool, args)
                if getattr(val_result, "valid", False) or (
                    isinstance(val_result, dict) and val_result.get("valid")
                ):
                    if isinstance(args, dict) and "file_path" in args and "path" not in args:
                        desfecho = "FALSO ACEITE (Passou com file_path)"
                    else:
                        desfecho = "ACEITO INDEVIDAMENTE / VÁLIDO"
                else:
                    err = getattr(val_result, "error", None) or (
                        val_result.get("error") if isinstance(val_result, dict) else "inválido"
                    )
                    desfecho = f"Rejeitado pelo validador: {err}"
                    
            except (LLMFormatError, ParseError) as e:
                desfecho = f"Rejeitado no parse ({type(e).__name__}): {e}"
            except Exception as e:
                desfecho = f"EXCEÇÃO VAZADA: {type(e).__name__}: {e}"

            print(f"{cid:<4} | {cat:<38} | {desfecho}")


if __name__ == "__main__":
    avaliar_corpus()