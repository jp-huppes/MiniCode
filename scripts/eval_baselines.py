import argparse
import json
import os
import random
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


PROMPT_VERSION = 'minicode-prompt-v2'
DEFAULT_SYSTEM_PROMPT = 'Você é o Agente Autônomo de Programação do MiniCode.\nSua missão é executar o plano em \'plan.md\' para cumprir os critérios de \'specification.md\'.\n\nA cada turno, responda com UM único objeto JSON, dentro de um bloco ```json:\n{\n  "thought": "raciocínio breve",\n  "tool": "nome_da_ferramenta",\n  "args": { ... }\n}\n\nFerramentas disponíveis (use exatamente estes nomes e argumentos):\n- append_file: Acrescenta conteúdo ao final de um arquivo (cria o arquivo se não existir). Não apaga o conteúdo existente.\n- done: Encerra a implementação e dispara a verificação.\n- list_files: Lista arquivos e diretórios do projeto recursivamente.\n- read_file: Lê o conteúdo de um arquivo de texto com números de linha.\n- replace_text: Substitui um trecho exato de um arquivo. \'old_str\' deve aparecer exatamente uma vez.\n- write_file: Cria ou SUBSTITUI o arquivo inteiro com o conteúdo fornecido. Para acrescentar a um arquivo existente, use append_file.\n- done: encerra a implementação. Sem argumentos. Use quando todos os passos do plano estiverem concluídos. Os testes são executados automaticamente pelo harness depois de \'done\'.\n\nRegras:\n- Use sempre \'path\' para caminhos de arquivo. Não existe \'file_path\'.\n- \'write_file\' SUBSTITUI o arquivo inteiro. Para acrescentar funções ou testes a um arquivo que já existe, use \'append_file\'.\n- Os testes exigidos pela SPEC (test_ac_n) devem existir em tests/ com o nome exato. O harness verifica isso ao final.\n- Não chame \'run_tests\': a verificação é feita pelo harness.\n- Chame \'done\' somente quando todos os passos do plano estiverem concluídos.\n- Não invente argumentos fora da lista acima.\n- Em arquivos de teste, inclua os imports no topo (ex.: `from src.pure_ops import clamp`). Nunca use um módulo ou função sem importá-lo.\n- Arquivos que já existem em tests/ no início são testes da tarefa: NÃO os altere. Coloque os testes \'test_ac_n\' exigidos pela SPEC em um arquivo novo (ex.: tests/test_spec_ac.py), como funções `def test_ac_...` no nível do módulo.\n'


def parse_action_response(raw_text: Any) -> Optional[Dict[str, Any]]:
    """Parser congelado: extrai objeto JSON com 'tool' (ou 'action') e 'args'."""
    if not isinstance(raw_text, str):
        return None
    texto = raw_text.strip()
    if not texto:
        return None

    # remove cercas markdown ```json ... ``` se existirem
    if texto.startswith("```"):
        linhas = texto.splitlines()
        if len(linhas) >= 3:
            linhas = linhas[1:-1] if linhas[-1].strip().startswith("```") else linhas[1:]
            texto = "\n".join(linhas).strip()

    try:
        obj = json.loads(texto)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("action")
    if not tool or not isinstance(tool, str):
        return None
    args = obj.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"tool": tool.strip(), "args": args}


def normalizar_args(args: Dict[str, Any]) -> Dict[str, Any]:
    norm = {}
    for k, v in args.items():
        if k in {"path", "file_path", "target_file"} and isinstance(v, str):
            norm[k] = v.replace("\\", "/").lstrip("./").strip()
        else:
            norm[k] = v
    return norm


def compute_f1_macro(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


def evaluate_predictions(
    examples: List[Dict[str, Any]],
    raw_responses: List[Optional[str]],
) -> Dict[str, Any]:
    assert len(examples) == len(raw_responses)
    n = len(examples)
    if n == 0:
        return {
            "n": 0,
            "per_example_correct": [],
            "acc": 0.0,
            "json_validity": 0.0,
            "exact_action_acc": 0.0,
            "f1_macro": 0.0,
            "by_stratum": {},
        }

    y_true_tools: List[str] = []
    y_pred_tools: List[str] = []
    per_example_correct: List[int] = []
    parsed_count = 0
    exact_full_count = 0

    by_stratum_hits: Dict[str, List[int]] = defaultdict(list)

    for ex, raw in zip(examples, raw_responses):
        ref_action = ex.get("y") or {"tool": ex.get("tool", ""), "args": {}}
        ref_tool = str(ref_action.get("tool", "")).strip()
        ref_args = normalizar_args(ref_action.get("args") or {})
        stratum = str(ex.get("fsm_state") or "UNKNOWN").upper()

        parsed = parse_action_response(raw)
        if parsed is not None:
            parsed_count += 1
            pred_tool = parsed["tool"]
            pred_args = normalizar_args(parsed["args"])
        else:
            pred_tool = "__INVALID_JSON__"
            pred_args = {}

        hit = 1 if pred_tool == ref_tool else 0
        per_example_correct.append(hit)
        by_stratum_hits[stratum].append(hit)

        if hit == 1 and pred_args == ref_args:
            exact_full_count += 1

        y_true_tools.append(ref_tool)
        y_pred_tools.append(pred_tool)

    stratum_acc = {
        st: round(sum(vals) / len(vals), 4)
        for st, vals in sorted(by_stratum_hits.items())
        if vals
    }

    return {
        "n": n,
        "per_example_correct": per_example_correct,
        "acc": round(sum(per_example_correct) / n, 4),
        "json_validity": round(parsed_count / n, 4),
        "exact_action_acc": round(exact_full_count / n, 4),
        "f1_macro": round(compute_f1_macro(y_true_tools, y_pred_tools), 4),
        "by_stratum": stratum_acc,
    }


def paired_bootstrap_delta_ci(
    examples: List[Dict[str, Any]],
    correct_a: List[int],
    correct_b: List[int],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap pareado por run_id (10.000 reamostragens por padrão)."""
    by_run_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        rid = str(ex.get("run_id") or f"idx_{idx}")
        by_run_indices[rid].append(idx)

    runs = sorted(by_run_indices.keys())
    if not runs:
        return {"delta_mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_gt_zero": 0.0}

    rng = random.Random(seed)
    deltas: List[float] = []

    for _ in range(n_resamples):
        sampled_runs = [rng.choice(runs) for _ in range(len(runs))]
        hits_a: List[int] = []
        hits_b: List[int] = []
        for rid in sampled_runs:
            for idx in by_run_indices[rid]:
                hits_a.append(correct_a[idx])
                hits_b.append(correct_b[idx])
        acc_a = sum(hits_a) / len(hits_a)
        acc_b = sum(hits_b) / len(hits_b)
        deltas.append(acc_a - acc_b)

    deltas.sort()
    low_idx = int(0.025 * (n_resamples - 1))
    high_idx = int(0.975 * (n_resamples - 1))
    p_gt_zero = sum(1 for d in deltas if d > 0.0) / n_resamples

    return {
        "delta_mean": round(sum(deltas) / n_resamples, 4),
        "ci95_low": round(deltas[low_idx], 4),
        "ci95_high": round(deltas[high_idx], 4),
        "p_gt_zero": round(p_gt_zero, 4),
    }


def majority_class_baseline(
    train_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> List[str]:
    counts = Counter(
        str((ex.get("y") or {}).get("tool") or ex.get("tool") or "")
        for ex in train_examples
        if ((ex.get("y") or {}).get("tool") or ex.get("tool"))
    )
    majority_tool = counts.most_common(1)[0][0] if counts else "read_file"
    payload = json.dumps({"tool": majority_tool, "args": {}}, ensure_ascii=False)
    return [payload for _ in test_examples]


def build_few_shot_prefix(train_examples: List[Dict[str, Any]], k: int = 3) -> str:
    if k <= 0 or not train_examples:
        return ""
    escolhidos = train_examples[:k]
    blocos = ["Exemplos de referência (do split de treino):"]
    for i, ex in enumerate(escolhidos, start=1):
        y_obj = ex.get("y") or {"tool": ex.get("tool", ""), "args": {}}
        blocos.append(
            f"--- Exemplo {i} ---\n"
            f"Estado/Prompt:\n{ex.get('x', '')}\n"
            f"Resposta esperada:\n{json.dumps(y_obj, ensure_ascii=False)}"
        )
    blocos.append("--- Agora resolva o próximo caso ---\n")
    return "\n\n".join(blocos)


def run_llm_baseline(
    test_examples: List[Dict[str, Any]],
    chat_fn: Callable[[List[Dict[str, str]], float], str],
    few_shot_prefix: str = "",
    system_prompt: str = "",
) -> List[str]:
    respostas: List[str] = []
    tag = "few_shot" if few_shot_prefix else "zero_shot"
    for idx, ex in enumerate(test_examples, 1):
        print(f"[{tag}] {idx}/{len(test_examples)}...", flush=True)
        prompt_x = str(ex.get("x", ""))
        conteudo = f"{few_shot_prefix}{prompt_x}" if few_shot_prefix else prompt_x
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": conteudo})
        # Temperatura congelada em 0.0 para avaliação de baselines
        raw = chat_fn(msgs, 0.0)
        respostas.append(raw)
    return respostas


def build_chat_fn(
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 30.0,
) -> Callable[[List[Dict[str, str]], float], str]:
    """Cria função chat(messages, temperature) compatível com endpoints OpenAI-compatible (ex.: vLLM)."""
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    def _chat_fn(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return f"__CHAT_ERROR__: {exc}"

        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "")

    return _chat_fn


def run_synthetic_smoke_test(few_shot_k: int = 2, resamples: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Valida métricas, few-shot prefix, zero-shot e bootstrap pareado com dados sintéticos."""
    train_ex = [
        {"task_id": "t1", "run_id": "r1", "fsm_state": "IMPLEMENTING", "x": "Ler src/app.py", "y": {"tool": "read_file", "args": {"path": "src/app.py"}}},
        {"task_id": "t2", "run_id": "r2", "fsm_state": "IMPLEMENTING", "x": "Escrever src/app.py", "y": {"tool": "write_file", "args": {"path": "src/app.py", "content": "x = 1"}}},
        {"task_id": "t3", "run_id": "r3", "fsm_state": "IMPLEMENTING", "x": "Ler README.md", "y": {"tool": "read_file", "args": {"path": "README.md"}}},
    ]
    test_ex = [
        {"task_id": "t4", "run_id": "r4", "fsm_state": "IMPLEMENTING", "x": "Ler src/main.py", "y": {"tool": "read_file", "args": {"path": "src/main.py"}}},
        {"task_id": "t5", "run_id": "r4", "fsm_state": "IMPLEMENTING", "x": "Escrever src/main.py", "y": {"tool": "write_file", "args": {"path": "src/main.py", "content": "pass"}}},
        {"task_id": "t6", "run_id": "r5", "fsm_state": "VERIFYING", "x": "Concluir tarefa", "y": {"tool": "done", "args": {"summary": "ok"}}},
        {"task_id": "t7", "run_id": "r5", "fsm_state": "IMPLEMENTING", "x": "Listar arquivos", "y": {"tool": "list_files", "args": {"path": "."}}},
    ]

    def fake_zero_shot(msgs: List[Dict[str, str]], temp: float) -> str:
        assert temp == 0.0
        prompt = msgs[-1]["content"]
        if "Escrever" in prompt:
            return '{"tool": "read_file", "args": {"path": "src/main.py"}}'
        return '{"tool": "read_file", "args": {"path": "src/main.py"}}'

    def fake_few_shot(msgs: List[Dict[str, str]], temp: float) -> str:
        assert temp == 0.0
        prompt = msgs[-1]["content"]
        if "Escrever" in prompt:
            return '{"tool": "write_file", "args": {"path": "src/main.py", "content": "pass"}}'
        if "Concluir" in prompt:
            return '{"tool": "done", "args": {"summary": "ok"}}'
        if "Listar" in prompt:
            return '{"tool": "list_files", "args": {"path": "."}}'
        return '{"tool": "read_file", "args": {"path": "src/main.py"}}'

    maj_preds = majority_class_baseline(train_ex, test_ex)
    zero_preds = run_llm_baseline(test_ex, fake_zero_shot, few_shot_prefix="")
    few_prefix = build_few_shot_prefix(train_ex, k=few_shot_k)
    few_preds = run_llm_baseline(test_ex, fake_few_shot, few_shot_prefix=few_prefix)

    m_maj = evaluate_predictions(test_ex, maj_preds)
    m_zero = evaluate_predictions(test_ex, zero_preds)
    m_few = evaluate_predictions(test_ex, few_preds)

    ci_few_vs_zero = paired_bootstrap_delta_ci(
        test_ex,
        m_few["per_example_correct"],
        m_zero["per_example_correct"],
        n_resamples=resamples,
        seed=seed,
    )

    return {
        "majority": {k: v for k, v in m_maj.items() if k != "per_example_correct"},
        "zero_shot": {k: v for k, v in m_zero.items() if k != "per_example_correct"},
        "few_shot": {k: v for k, v in m_few.items() if k != "per_example_correct"},
        "bootstrap_few_vs_zero": ci_few_vs_zero,
    }


def carregar_jsonl(caminho: Path) -> List[Dict[str, Any]]:
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa G6: Avaliação de baselines e bootstrap pareado por run")
    parser.add_argument("--dataset", default="corpus_filtered/dataset.jsonl")
    parser.add_argument("--manifest", default="corpus_filtered/split_manifest.json")
    parser.add_argument("--base-url", default=os.getenv("BASELINE_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("BASELINE_MODEL", ""))
    parser.add_argument("--api-key", default=os.getenv("BASELINE_API_KEY", ""))
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--few-shot-k", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-report", default="corpus_filtered/baselines_report.json")
    parser.add_argument("--synthetic-test", action="store_true", help="Roda validação local com dados sintéticos")
    args = parser.parse_args()

    if args.synthetic_test:
        res = run_synthetic_smoke_test(few_shot_k=args.few_shot_k, resamples=min(args.bootstrap_resamples, 500), seed=args.seed)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    dataset = carregar_jsonl(Path(args.dataset))
    manifest_path = Path(args.manifest)

    if not dataset or not manifest_path.exists():
        print("Aviso: dataset.jsonl vazio ou split_manifest.json ausente.")
        print("Para testar a lógica de métricas e bootstrap agora, use --synthetic-test ou aguarde os dados reais.")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    train_tasks = set(splits.get("train", []))
    test_tasks = set(splits.get("test", []))

    train_ex = [ex for ex in dataset if str(ex.get("task_id")) in train_tasks]
    test_ex = [ex for ex in dataset if str(ex.get("task_id")) in test_tasks]

    report: Dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "temperature": 0.0,
        "few-shot-k": args.few_shot_k,
        "bootstrap_resamples": args.bootstrap_resamples,
        "test_examples_count": len(test_ex),
        "systems": {},
        "comparisons": {},
    }

    maj_preds = majority_class_baseline(train_ex, test_ex)
    maj_metrics = evaluate_predictions(test_ex, maj_preds)
    report["systems"]["majority_class"] = {k: v for k, v in maj_metrics.items() if k != "per_example_correct"}

    if args.base_url and args.model:
        chat_fn = build_chat_fn(base_url=args.base_url, model=args.model, api_key=args.api_key)

        zero_preds = run_llm_baseline(
            test_ex,
            chat_fn=chat_fn,
            few_shot_prefix="",
            system_prompt=args.system_prompt,
        )
        zero_metrics = evaluate_predictions(test_ex, zero_preds)
        report["systems"]["zero_shot"] = {k: v for k, v in zero_metrics.items() if k != "per_example_correct"}

        few_prefix = build_few_shot_prefix(train_ex, k=args.few_shot_k)
        few_preds = run_llm_baseline(
            test_ex,
            chat_fn=chat_fn,
            few_shot_prefix=few_prefix,
            system_prompt=args.system_prompt,
        )
        few_metrics = evaluate_predictions(test_ex, few_preds)
        report["systems"]["few_shot"] = {k: v for k, v in few_metrics.items() if k != "per_example_correct"}

        report["comparisons"]["zero_shot_vs_majority"] = paired_bootstrap_delta_ci(
            test_ex,
            zero_metrics["per_example_correct"],
            maj_metrics["per_example_correct"],
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        report["comparisons"]["few_shot_vs_majority"] = paired_bootstrap_delta_ci(
            test_ex,
            few_metrics["per_example_correct"],
            maj_metrics["per_example_correct"],
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        report["comparisons"]["few_shot_vs_zero_shot"] = paired_bootstrap_delta_ci(
            test_ex,
            few_metrics["per_example_correct"],
            zero_metrics["per_example_correct"],
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    else:
        print("Aviso: --base-url e --model não informados; apenas majority_class foi avaliada (execução final fica para após o dataset).")

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()