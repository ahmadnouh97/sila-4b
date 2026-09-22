from dataclasses import dataclass
from math import isfinite

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_json(value: object, field: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
    elif isinstance(value, list):
        for item in value:
            _require_json(item, field)
        return
    elif isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            for item in value.values():
                _require_json(item, field)
            return
    raise ValueError(f"{field} must contain only valid JSON values")


def _require_call(tool_name: object, arguments: object) -> None:
    _require_text(tool_name, "tool_name")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    _require_json(arguments, "arguments")


def _require_non_negative_number(value: object, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field} must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.description, "description")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be an object")
        _require_json(self.parameters, "parameters")
        if self.parameters.get("type") != "object":
            raise ValueError("parameters.type must be 'object'")
        if not isinstance(self.parameters.get("properties"), dict):
            raise ValueError("parameters.properties must be an object")


@dataclass(frozen=True, slots=True)
class ExpectedToolCall:
    tool_name: str
    arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _require_call(self.tool_name, self.arguments)


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    tool_name: str
    arguments: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _require_call(self.tool_name, self.arguments)


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    id: str
    source: str
    language: str
    domain: str
    user_utterance: str
    available_tools: tuple[ToolDefinition, ...]
    should_call_tool: bool
    expected_tool_name: str | None
    expected_arguments: dict[str, JsonValue] | None
    dialect: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("id", "source", "language", "domain", "user_utterance"):
            _require_text(getattr(self, field), field)
        if self.dialect is not None:
            _require_text(self.dialect, "dialect")
        if not isinstance(self.available_tools, tuple) or not self.available_tools:
            raise ValueError("available_tools must be a non-empty tuple")
        if not all(isinstance(tool, ToolDefinition) for tool in self.available_tools):
            raise ValueError("available_tools must contain ToolDefinition values")
        tool_names = [tool.name for tool in self.available_tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("available tool names must be unique")
        if not isinstance(self.should_call_tool, bool):
            raise ValueError("should_call_tool must be a boolean")
        if self.should_call_tool:
            _require_call(self.expected_tool_name, self.expected_arguments)
            if self.expected_tool_name not in tool_names:
                raise ValueError("expected tool must be present in available_tools")
        elif self.expected_tool_name is not None or self.expected_arguments is not None:
            raise ValueError("negative examples cannot include an expected tool call")
        if not isinstance(self.tags, tuple):
            raise ValueError("tags must be a tuple")
        for tag in self.tags:
            _require_text(tag, "tag")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("tags must be unique")


@dataclass(frozen=True, slots=True)
class RawPrediction:
    example_id: str
    generated_text: str
    model_id: str
    model_revision: str
    prompt_version: str
    generation_config: dict[str, JsonValue]
    latency_seconds: float
    gpu_memory_mb: float | None = None

    def __post_init__(self) -> None:
        for field in ("example_id", "model_id", "model_revision", "prompt_version"):
            _require_text(getattr(self, field), field)
        if not isinstance(self.generated_text, str):
            raise ValueError("generated_text must be a string")
        if not isinstance(self.generation_config, dict):
            raise ValueError("generation_config must be an object")
        _require_json(self.generation_config, "generation_config")
        _require_non_negative_number(self.latency_seconds, "latency_seconds")
        if self.gpu_memory_mb is not None:
            _require_non_negative_number(self.gpu_memory_mb, "gpu_memory_mb")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    example_id: str
    raw_prediction: RawPrediction
    expected_call: ExpectedToolCall | None
    parsed_call: ParsedToolCall | None
    tool_selection_correct: bool
    arguments_correct: bool | None
    exact_match: bool
    parse_error: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.example_id, "example_id")
        if not isinstance(self.raw_prediction, RawPrediction):
            raise ValueError("raw_prediction must be a RawPrediction")
        if self.raw_prediction.example_id != self.example_id:
            raise ValueError("result and prediction example IDs must match")
        if self.expected_call is not None and not isinstance(
            self.expected_call, ExpectedToolCall
        ):
            raise ValueError("expected_call must be an ExpectedToolCall or None")
        if self.parsed_call is not None and not isinstance(
            self.parsed_call, ParsedToolCall
        ):
            raise ValueError("parsed_call must be a ParsedToolCall or None")
        if not isinstance(self.tool_selection_correct, bool):
            raise ValueError("tool_selection_correct must be a boolean")
        if self.arguments_correct is not None and not isinstance(
            self.arguments_correct, bool
        ):
            raise ValueError("arguments_correct must be a boolean or None")
        if not isinstance(self.exact_match, bool):
            raise ValueError("exact_match must be a boolean")
        if self.exact_match and (
            not self.tool_selection_correct or self.arguments_correct is False
        ):
            raise ValueError(
                "exact_match requires correct tool selection and arguments"
            )
        if self.parse_error is not None:
            _require_text(self.parse_error, "parse_error")
