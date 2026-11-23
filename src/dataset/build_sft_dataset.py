"""Gera splits SFT em formato JSONL de mensagens (system/user/assistant) a partir de dataset.jsonl."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROMPT_VERSION = 'minicode-prompt-v2'
DEFAULT_SYSTEM_PROMPT = 'Você é o Agente Autônomo de Programação do MiniCode.\nSua missão é executar o plano em \'plan.md\' para cumprir os critérios de \'specification.md\'.\n\nA cada turno, responda com UM único objeto JSON, dentro de um bloco ```json:\n{\n  "thought": "raciocínio breve",\n  "tool": "nome_da_ferramenta",\n  "args": { ... }\n}\n\nFerramentas disponíveis (use exatamente estes nomes e argumentos):\n- append_file: Acrescenta conteúdo ao final de um arquivo (cria o arquivo se não existir). Não apaga o conteúdo existente.\n- done: Encerra a implementação e dispara a verificação.\n- list_files: Lista arquivos e diretórios do projeto recursivamente.\n- read_file: Lê o conteúdo de um arquivo de texto com números de linha.\n- replace_text: Substitui um trecho exato de um arquivo. \'old_str\' deve aparecer exatamente uma vez.\n- write_file: Cria ou SUBSTITUI o arquivo inteiro com o conteúdo fornecido. Para acrescentar a um arquivo existente, use append_file.\n- done: encerra a implementação. Sem argumentos. Use quando todos os passos do plano estiverem concluídos. Os testes são executados automaticamente pelo harness depois de \'done\'.\n\nRegras:\n- Use sempre \'path\' para caminhos de arquivo. Não existe \'file_path\'.\n- \'write_file\' SUBSTITUI o arquivo inteiro. Para acrescentar funções ou testes a um arquivo que já existe, use \'append_file\'.\n- Os testes exigidos pela SPEC (test_ac_n) devem existir em tests/ com o nome exato. O harness verifica isso ao final.\n- Não chame \'run_tests\': a verificação é feita pelo harness.\n- Chame \'done\' somente quando todos os passos do plano estiverem concluídos.\n- Não invente argumentos fora da lista acima.\n- Em arquivos de teste, inclua os imports no topo (ex.: `from src.pure_ops import clamp`). Nunca use um módulo ou função sem importá-lo.\n- Arquivos que já existem em tests/ no início são testes da tarefa: NÃO os altere. Coloque os testes \'test_ac_n\' exigidos pela SPEC em um arquivo novo (ex.: tests/test_spec_ac.py), como funções `def test_ac_...` no nível do módulo.\n'

def carregar_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    linhas: List[Dict[str, Any]] = []
    for linha in path.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            linhas.append(json.loads(linha))
    return linhas


def calcular_sha256_linhas(linhas: List[str]) -> str:
    payload = "\n".join(linhas) + ("\n" if linhas else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalizar_alvo_json(ex: Dict[str, Any]) -> str:
    y = ex.get("y")
    if isinstance(y, dict):
        tool = y.get("tool") or y.get("action") or ex.get("tool") or ""
        args = y.get("args")
        if not isinstance(args, dict):
            args = {}
        payload = {"tool": str(tool).strip(), "args": args}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if isinstance(y, str) and y.strip():
        return y.strip()
    tool = str(ex.get("tool") or "").strip()
    return json.dumps({"tool": tool, "args": {}}, ensure_ascii=False, sort_keys=True)


def extrair_ferramenta(ex: Dict[str, Any]) -> str:
    if isinstance(ex.get("tool"), str) and ex["tool"].strip():
        return ex["tool"].strip()
    y = ex.get("y")
    if isinstance(y, dict):
        t = y.get("tool") or y.get("action")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return "unknown"


def validar_splits_sem_vazamento(splits: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
    train_ids = {str(x) for x in (splits.get("train") or [])}
    val_ids = {str(x) for x in (splits.get("val") or splits.get("validation") or [])}
    test_ids = {str(x) for x in (splits.get("test") or [])}

    if train_ids & val_ids:
        raise ValueError(f"Vazamento entre train e val: {sorted(train_ids & val_ids)}")
    if train_ids & test_ids:
        raise ValueError(f"Vazamento entre train e test: {sorted(train_ids & test_ids)}")
    if val_ids & test_ids:
        raise ValueError(f"Vazamento entre val e test: {sorted(val_ids & test_ids)}")

    return train_ids, val_ids, test_ids


def montar_registro_sft(
    ex: Dict[str, Any],
    split_name: str,
    system_prompt: str,
) -> Optional[Dict[str, Any]]:
    prompt_user = ex.get("x")
    if prompt_user is None:
        prompt_user = ex.get("prompt")
    if not isinstance(prompt_user, str) or not prompt_user.strip():
        return None

    assistant_content = normalizar_alvo_json(ex)
    tool_name = extrair_ferramenta(ex)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_user},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "split": split_name,
            "run_id": ex.get("run_id"),
            "task_id": ex.get("task_id"),
            "fsm_state": ex.get("fsm_state"),
            "tool": tool_name,
        },
    }


def imprimir_distribuicao(nome_split: str, registros: List[Dict[str, Any]]) -> None:
    total = len(registros)
    print(f"\n=== DISTRIBUIÇÃO DE FERRAMENTAS ({nome_split}: {total}) ===")
    if total == 0:
        print("  Nenhum exemplo neste split.")
        return
    contagem = Counter(r["metadata"]["tool"] for r in registros)
    for tool_name, qtd in contagem.most_common():
        pct = (qtd / total) * 100.0
        alerta = " [ATENÇÃO: > 60% — reavaliar balanceamento]" if pct > 60.0 else ""
        print(f"  {tool_name}: {qtd} ({pct:.2f}%){alerta}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formata dataset.jsonl filtrado em splits JSONL (system/user/assistant) para SFT/avaliação"
    )
    parser.add_argument("--dataset", default="corpus_filtered/dataset.jsonl", help="Caminho do dataset filtrado")
    parser.add_argument("--manifest", default="corpus_filtered/split_manifest.json", help="Caminho do manifesto de splits")
    parser.add_argument("--out-dir", default="datasets/sft_jsonl", help="Diretório de saída dos arquivos JSONL")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Mensagem de sistema padrão")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    exemplos = carregar_jsonl(dataset_path)
    if not exemplos:
        print(f"Aviso: nenhum exemplo encontrado em {dataset_path}.")
        print("Quando dataset.jsonl tiver exemplos reais (após a coleta e filtragem F1-F5), rode novamente este script.")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto de splits não encontrado: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits_raw = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else manifest
    train_ids, val_ids, test_ids = validar_splits_sem_vazamento(splits_raw)

    por_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    sem_split = 0

    for ex in exemplos:
        tid = str(ex.get("task_id") or "")
        if tid in train_ids:
            split_name = "train"
        elif tid in val_ids:
            split_name = "val"
        elif tid in test_ids:
            split_name = "test"
        else:
            sem_split += 1
            continue

        registro = montar_registro_sft(ex, split_name, args.system_prompt)
        if registro is not None:
            por_split[split_name].append(registro)

    out_dir.mkdir(parents=True, exist_ok=True)
    resumo_arquivos: Dict[str, Dict[str, Any]] = {}

    for split_name in ("train", "val", "test"):
        linhas_json = [json.dumps(r, ensure_ascii=False) for r in por_split[split_name]]
        arquivo_split = out_dir / f"{split_name}.jsonl"
        conteudo = "\n".join(linhas_json) + ("\n" if linhas_json else "")
        arquivo_split.write_text(conteudo, encoding="utf-8")
        resumo_arquivos[split_name] = {
            "path": str(arquivo_split),
            "count": len(linhas_json),
            "sha256": calcular_sha256_linhas(linhas_json),
        }

    manifesto_sft = {
        "prompt_version": PROMPT_VERSION,
        "source_dataset": str(dataset_path),
        "source_manifest": str(manifest_path),
        "splits": resumo_arquivos,
        "skipped_unassigned_task_ids": sem_split,
    }
    (out_dir / "sft_manifest.json").write_text(
        json.dumps(manifesto_sft, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== RESUMO DA FORMATAÇÃO SFT ===")
    for split_name in ("train", "val", "test"):
        info = resumo_arquivos[split_name]
        print(f"  {split_name}: {info['count']} exemplos -> {info['path']} (sha256={info['sha256'][:12]}...)")
    if sem_split:
        print(f"  AVISO: {sem_split} exemplos ignorados por task_id ausente nos splits.")

    imprimir_distribuicao("train", por_split["train"])
    imprimir_distribuicao("val", por_split["val"])
    imprimir_distribuicao("test", por_split["test"])


if __name__ == "__main__":
    main()