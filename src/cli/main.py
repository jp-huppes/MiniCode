import sys
import argparse
from pathlib import Path
from typing import List, Optional, Callable, Any

from src.cli.renderer import HumanRenderer, JsonRenderer, QuietRenderer, Renderer
from src.agent.orchestrator import AgentOrchestrator
from src.llm.client import LLMClient
from src.agent.state import State

class RunResult:
    def __init__(self, final_state: str, success: bool, run_dir: Path, error: Optional[str] = None):
        self.final_state = final_state
        self.success = success
        self.run_dir = run_dir
        self.error = error

def run_task(
    task: str,
    project_dir: Path,
    config: dict,
    on_event: Callable[[str, dict], None]
) -> RunResult:
    minicode_dir = project_dir / ".minicode"
    llm_client = LLMClient() # usa variaveis de ambiente para API key / base URL
    
    orchestrator = AgentOrchestrator(project_dir, minicode_dir, llm_client)
    
    # simula emissao de eventos iniciais e disparar a FSM
    on_event("state_changed", {"old": "RECEIVED", "new": "SPECIFIED"})
    
    # geracao simplificada de spec/plan mock ou real se integrada aos model spec
    spec_content = f"# Especificação para: {task}\n\n## Requisitos\n- {task}"
    plan_content = f"# Plano\n\n1. Implementar a tarefa solicitada."
    
    spec_path = project_dir / "specification.md"
    plan_path = project_dir / "plan.md"
    spec_path.write_text(spec_content, encoding="utf-8")
    plan_path.write_text(plan_content, encoding="utf-8")
    
    on_event("spec_ready", {"path": str(spec_path), "content": spec_content})
    on_event("plan_ready", {"path": str(plan_path), "content": plan_content})
    
    on_event("state_changed", {"old": "SPECIFIED", "new": "IMPLEMENTING"})
    
    try:
        final_state_enum = orchestrator.run(spec_content, plan_content)
        final_state_str = final_state_enum.value if hasattr(final_state_enum, "value") else str(final_state_enum)
        success = (final_state_str == "DONE")
        
        run_dir = orchestrator.executor.logger.run_dir if hasattr(orchestrator.executor, "logger") else minicode_dir
        
        result = RunResult(
            final_state=final_state_str,
            success=success,
            run_dir=run_dir,
            error=orchestrator.state_data.failure_reason
        )
        on_event("run_finished", {"final_state": final_state_str, "success": success})
        return result
    except Exception as e:
        result = RunResult(
            final_state="FAILED",
            success=False,
            run_dir=minicode_dir,
            error=str(e)
        )
        on_event("run_finished", {"final_state": "FAILED", "success": False, "error": str(e)})
        return result

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="minicode",
        description="Agente Autônomo de Programação MiniCode"
    )
    parser.add_argument("task", nargs="?", help="Tarefa a ser executada pelo agente (modo one-shot)")
    parser.add_argument("--project", default=".", help="Diretório de trabalho do projeto (default: diretório atual)")
    parser.add_argument("--max-iterations", type=int, default=25, help="Número máximo de iterações do loop")
    parser.add_argument("--max-repair-attempts", type=int, default=3, help="Número máximo de tentativas de reparo")
    parser.add_argument("--test-cmd", default="pytest", help="Comando de teste customizado")
    parser.add_argument("--json", action="store_true", help="Saída estrita em formato JSON")
    parser.add_argument("--verbose", action="store_true", help="Exibe detalhes completos")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso (apenas resultado final)")
    parser.add_argument("--no-color", action="store_true", help="Desativa cores ANSI no terminal")
    return parser.parse_args(argv)

def repl(config: dict, renderer: Renderer) -> int:
    print("MiniCode REPL Interativo. Digite sua tarefa ou /exit para sair.")
    while True:
        try:
            line = input("minicode> ").strip()
            if not line: continue
            if line == "/exit":
                print("Saindo...")
                break
            if line == "/help":
                print("Comandos: /exit (sai), /help (ajuda), ou digite uma descrição de tarefa.")
                continue
            
            result = run_task(line, config["project_dir"], config, renderer.handle)
            renderer.finish(result)
        except (KeyboardInterrupt, EOFError):
            print("\nInterrompido. Saindo...")
            return 130
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    if argv is None: argv = sys.argv[1:]
        
    try: args = parse_args(argv)
    except SystemExit: return 2  # erro de uso de argumentos

    project_dir = Path(args.project).resolve()
    if not project_dir.exists():
        try: project_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            print(f"Erro: O diretório --project '{args.project}' não existe e não pôde ser criado.", file=sys.stderr)
            return 2
            
    if not project_dir.is_dir():
        print(f"Erro: O caminho '{args.project}' informado em --project não é um diretório válido.", file=sys.stderr)
        return 2

    if args.json and not args.task:
        print("Erro: A flag --json só pode ser utilizada no modo direto com uma tarefa (--project + task).", file=sys.stderr)
        return 2

    # selecao do renderer
    if args.json: renderer = JsonRenderer()
    elif args.quiet: renderer = QuietRenderer()
    else: renderer = HumanRenderer(verbose=args.verbose, no_color=args.no_color)

    config = {
        "project_dir": project_dir,
        "max_iterations": args.max_iterations,
        "max_repair_attempts": args.max_repair_attempts,
        "test_cmd": args.test_cmd
    }

    if args.task:
        if not args.task.strip():
            print("Erro: A tarefa não pode ser vazia.", file=sys.stderr)
            return 2
        try:
            result = run_task(args.task, project_dir, config, renderer.handle)
            renderer.finish(result)
            return 0 if result.success else 1
        except KeyboardInterrupt:
            print("\nExecução interrompida por Ctrl+C.", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Erro de infraestrutura: {str(e)}", file=sys.stderr)
            return 3
    else: return repl(config, renderer)

if __name__ == "__main__":
    sys.exit(main())