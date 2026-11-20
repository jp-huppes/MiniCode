import sys
import json
from pathlib import Path

# Adiciona src ao path para importar os módulos atuais
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from llm.action_parser import parse_acao_json
    from llm.errors import LLMFormatError
    from agent.validator import ActionValidator
except ImportError as e:
    print(f"Erro ao importar módulos do MiniCode: {e}")
    sys.exit(1)

CORPUS = [
    {
        "id": "C1",
        "categoria": "file_path no lugar de path",
        "input": '{"tool": "write_file", "args": {"file_path": "main.py", "content": "print()"}}'
    },
    {
        "id": "C2",
        "categoria": "Campo extra em args",
        "input": '{"tool": "read_file", "args": {"path": "main.py", "extra_field": 123}}'
    },
    {
        "id": "C3",
        "categoria": "Tipo errado (content numérico)",
        "input": '{"tool": "write_file", "args": {"path": "main.py", "content": 12345}}'
    },
    {
        "id": "C4",
        "categoria": "Ferramenta inexistente",
        "input": '{"tool": "hack_system", "args": {"path": "main.py"}}'
    },
    {
        "id": "C5",
        "categoria": "JSON truncado",
        "input": '{"tool": "write_file", "args": {"path": "main.py"'
    },
    {
        "id": "C6",
        "categoria": "Chave action no lugar de tool",
        "input": '{"action": "write_file", "args": {"path": "main.py", "content": "ok"}}'
    },
    {
        "id": "C7",
        "categoria": "Args de outra ferramenta (faltando path)",
        "input": '{"tool": "read_file", "args": {"content": "olá"}}'
    },
    {
        "id": "C8",
        "categoria": "Texto livre",
        "input": 'Aqui está o seu arquivo modificado com sucesso.'
    },
    {
        "id": "C9",
        "categoria": "Dois objetos JSON na mesma resposta",
        "input": '{"tool": "read_file", "args": {"path": "a.py"}} {"tool": "read_file", "args": {"path": "b.py"}}'
    },
    {
        "id": "C10",
        "categoria": "Cerca markdown sem crases com prefixo json",
        "input": 'json\n{"tool": "read_file", "args": {"path": "main.py"}}'
    }
]

def avaliar_corpus():
    print(f"{'ID':<4} | {'Categoria':<35} | {'Desfecho no Pipeline Atual'}")
    print("-" * 75)
    
    validator = ActionValidator()
    
    for item in CORPUS:
        cid = item["id"]
        cat = item["categoria"]
        texto = item["input"]
        
        desfecho = ""
        try:
            # Tenta o parser atual
            parsed = parse_action_response(texto)
            tool = parsed.get("tool")
            args = parsed.get("args", {})
            
            # Tenta o validador atual
            val_result = validator.validate(tool, args)
            if val_result.get("valid"):
                # Verifica se há divergência D6 (ex: file_path passou)
                if "file_path" in args and "path" not in args:
                    desfecho = "FALSO ACEITE (Passou no parser/validador com file_path)"
                else:
                    desfecho = "ACEITO INDEVIDAMENTE / VÁLIDO"
            else:
                desfecho = f"Rejeitado pelo validador: {val_result.get('error', 'erro desconhecido')}"
        except LLMFormatError as e:
            desfecho = f"Rejeitado corretamente (LLMFormatError): {e}"
        except Exception as e:
            desfecho = f"EXCEÇÃO VAZADA: {type(e).__name__}: {e}"
            
        print(f"{cid:<4} | {cat:<35} | {desfecho}")

if __name__ == "__main__":
    avaliar_corpus()