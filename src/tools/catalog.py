from dataclasses import dataclass
import json
import copy
from pydantic import TypeAdapter
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Type, Union
from pydantic import BaseModel, ConfigDict, Field


class StrictArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ListFilesArgs(StrictArgsModel):
    path: str = "."
    max_entries: Optional[int] = 200

class ReadFileArgs(StrictArgsModel):
    path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None

class WriteFileArgs(StrictArgsModel):
    path: str
    content: str

class AppendFileArgs(StrictArgsModel):
    path: str
    content: str

class ReplaceTextArgs(StrictArgsModel):
    path: str
    old_str: str
    new_str: str

class RunTestsArgs(StrictArgsModel):
    path: Optional[str] = None
    args: Optional[List[str]] = None
    command: Optional[str] = None

class DoneArgs(StrictArgsModel): pass

class EnvelopeListFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["list_files"]
    args: ListFilesArgs = Field(default_factory=ListFilesArgs)

class EnvelopeReadFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["read_file"]
    args: ReadFileArgs

class EnvelopeWriteFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["write_file"]
    args: WriteFileArgs

class EnvelopeAppendFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["append_file"]
    args: AppendFileArgs

class EnvelopeReplaceText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["replace_text"]
    args: ReplaceTextArgs

class EnvelopeRunTests(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["run_tests"]
    args: RunTestsArgs = Field(default_factory=RunTestsArgs)

class EnvelopeDone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thought: Optional[str] = None
    tool: Literal["done"]
    args: DoneArgs = Field(default_factory=DoneArgs)

AcaoEnvelope = Annotated[
    Union[EnvelopeListFiles, EnvelopeReadFile, EnvelopeWriteFile, EnvelopeAppendFile, EnvelopeReplaceText, EnvelopeRunTests, EnvelopeDone,],
    Field(discriminator="tool"),]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: Type[BaseModel]
    descricao: str
    escreve_arquivos: bool = False
    so_harness: bool = False


TOOL_CATALOG: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_files",
        args_model=ListFilesArgs,
        descricao="Lista arquivos e diretórios do projeto recursivamente.",
        escreve_arquivos=False,
        so_harness=False,
    ),
    ToolSpec(
        name="read_file",
        args_model=ReadFileArgs,
        descricao="Lê o conteúdo de um arquivo de texto com números de linha.",
        escreve_arquivos=False,
        so_harness=False,
    ),
    ToolSpec(
        name="write_file",
        args_model=WriteFileArgs,
        descricao="Cria ou SUBSTITUI o arquivo inteiro com o conteúdo fornecido. Para acrescentar a um arquivo existente, use append_file.",
        escreve_arquivos=True,
        so_harness=False,
    ),
    ToolSpec(
        name="append_file",
        args_model=AppendFileArgs,
        descricao="Acrescenta conteúdo ao final de um arquivo (cria o arquivo se não existir). Não apaga o conteúdo existente.",
        escreve_arquivos=True,
        so_harness=False,
    ),
    ToolSpec(
        name="replace_text",
        args_model=ReplaceTextArgs,
        descricao="Substitui um trecho exato de um arquivo. 'old_str' deve aparecer exatamente uma vez.",
        escreve_arquivos=True,
        so_harness=False,
    ),
    ToolSpec(
        name="run_tests",
        args_model=RunTestsArgs,
        descricao="Executa a suíte de testes utilizando pytest e retorna o resultado.",
        escreve_arquivos=False,
        so_harness=True,
    ),
    ToolSpec(
        name="done",
        args_model=DoneArgs,
        descricao="Encerra a implementação e dispara a verificação.",
        escreve_arquivos=False,
        so_harness=False,
    ),
)

CATALOG_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_CATALOG}

ALLOWED_TOOLS: Set[str] = {spec.name for spec in TOOL_CATALOG}

LLM_TOOLS: List[str] = [spec.name for spec in TOOL_CATALOG if not spec.so_harness]

WRITE_TOOLS: Set[str] = {spec.name for spec in TOOL_CATALOG if spec.escreve_arquivos}

FILE_PATH_TOOLS: Set[str] = {spec.name for spec in TOOL_CATALOG if "path" in spec.args_model.model_fields}
CONTENT_TOOLS: Set[str] = {spec.name for spec in TOOL_CATALOG if "content" in spec.args_model.model_fields}


def export_openai_tools(include_harness_only: bool = False) -> List[Dict[str, Any]]:
    openai_tools: List[Dict[str, Any]] = []
    for spec in TOOL_CATALOG:
        if spec.so_harness and not include_harness_only: continue
        schema = spec.args_model.model_json_schema()
        schema.pop("title", None)
        openai_tools.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.descricao,
                "parameters": schema,
            },
        })

    return openai_tools


def export_prompt_text(include_harness_only: bool = False) -> str:
    lines: List[str] = []
    for spec in TOOL_CATALOG:
        if spec.so_harness and not include_harness_only: continue
        schema = spec.args_model.model_json_schema()
        schema.pop("title", None)
        lines.append(f"- Ferramenta: `{spec.name}`")
        lines.append(f"  Descrição: {spec.descricao}")
        lines.append(f"  Parâmetros JSON Schema: {json.dumps(schema, ensure_ascii=False)}")
        
    return "\n".join(lines)

ENVELOPE_ADAPTER = TypeAdapter(AcaoEnvelope)

def export_envelope_schema() -> Dict[str, Any]:
    """Gera o JSON Schema bruto da união discriminada AcaoEnvelope."""
    return ENVELOPE_ADAPTER.json_schema()


def _inline_refs(node: Any, defs: Dict[str, Any], seen: set[str] | None = None) -> Any:
    seen = seen or set()

    if isinstance(node, list):
        return [_inline_refs(item, defs, set(seen)) for item in node]

    if not isinstance(node, dict):
        return node

    # Remove 'discriminator' (backends como vLLM costumam ter suporte parcial)
    cleaned = {k: v for k, v in node.items() if k != "discriminator"}

    if "$ref" in cleaned:
        ref_str = cleaned.pop("$ref")
        prefix = "#/$defs/"
        if ref_str.startswith(prefix):
            def_name = ref_str[len(prefix):]
            if def_name in defs and def_name not in seen:
                next_seen = set(seen)
                next_seen.add(def_name)
                target = _inline_refs(copy.deepcopy(defs[def_name]), defs, next_seen)
                # Mescla eventuais chaves irmãs (exceto 'title') se houver
                cleaned.pop("title", None)
                if isinstance(target, dict):
                    merged = copy.deepcopy(target)
                    for k, v in cleaned.items():
                        if k != "title":
                            merged[k] = v
                    return merged
                return target

    # Remove 'title' e 'default' supérfluos
    cleaned.pop("title", None)
    cleaned.pop("default", None)

    result: Dict[str, Any] = {}
    for k, v in cleaned.items():
        if k == "$defs":
            continue
        result[k] = _inline_refs(v, defs, set(seen))

    return result


def normalize_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza um JSON Schema para decodificação restrita:
    - expande $ref a partir de $defs e remove $defs
    - remove 'discriminator' mantendo 'oneOf' e 'const'
    - remove 'title' e 'default' supérfluos
    """
    working = copy.deepcopy(schema)
    defs = working.get("$defs", {})
    normalized = _inline_refs(working, defs)
    if isinstance(normalized, dict):
        normalized.pop("$defs", None)
    return normalized


def export_normalized_envelope_schema() -> Dict[str, Any]:
    """Exporta o schema do envelope normalizado para Camada 1."""
    return normalize_json_schema(export_envelope_schema())

