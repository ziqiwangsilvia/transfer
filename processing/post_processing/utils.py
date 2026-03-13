from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Tool names used to detect the "tool-as-key" JSON format emitted by some models,
# e.g. {"show_line_chart": {...}} instead of {"name": "show_line_chart", ...}.
KNOWN_TOOL_NAMES = {
    "show_pie_chart",
    "show_line_chart",
    "show_stacked_bar_chart",
}


class ToolCall:
    """
    Normalised representation of a single tool invocation.

    Handles both "name" and "tool" fields emitted by different model families.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tool: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.tool = tool
        self.parameters = parameters or {}

    def get_tool_name(self) -> str:
        """Return the tool name, preferring 'name' over 'tool'."""
        return self.name or self.tool or ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the ground-truth format used by the evaluator."""
        return {"name": self.get_tool_name(), "arguments": self.parameters}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from a dict that may use 'name'/'tool' and 'parameters'/'arguments'."""
        return cls(
            name=data.get("name"),
            tool=data.get("tool"),
            parameters=data.get("parameters") or data.get("arguments", {}),
        )


class ParsedResponse:
    """Parsed model output, ready for the evaluator."""

    def __init__(
        self,
        response_type: str,
        tools: Optional[List[ToolCall]] = None,
        response: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.type = response_type
        self.tools = tools or []
        self.response = response
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.type}
        if self.tools:
            result["tools"] = [tc.to_dict() for tc in self.tools]
        if self.response:
            result["response"] = self.response
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class ParsedInferenceResult:
    """Normalised output of one inference call, ready for the evaluator."""

    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""
    parse_error: Optional[str] = None
    # Raw output fields — preserved for eval metrics but not used during parsing.
    raw_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_content: str = ""

    def to_eval_dict(self) -> Dict[str, Any]:
        """Return the dict format expected by the tool-calling evaluator."""
        if self.tool_calls:
            return {"type": "tool", "tools": self.tool_calls}
        if self.content:
            return {"type": "nlp", "response": self.content}
        return {"type": "error", "error": self.parse_error or "empty output"}
