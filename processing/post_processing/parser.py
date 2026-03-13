import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from processing.post_processing.json_parser import _parse_arguments, parse_json_output
from processing.post_processing.pythonic_parser import parse_pythonic_output
from processing.post_processing.utils import (
    ParsedInferenceResult,
    ParsedResponse,
    ToolCall,
)

log = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base — subclasses implement _parse_content() for the text path."""

    def __init__(self, template_has_tool_token: bool = False):
        self.template_has_tool_token = template_has_tool_token

    def parse(self, result: Dict[str, Any]) -> ParsedInferenceResult:
        """Normalise one result dict from InferenceEngine."""
        if "error" in result:
            return ParsedInferenceResult(parse_error=result["error"])

        has_tool_calls = "tool_calls" in result and result["tool_calls"]
        text = result.get("content", "")

        # Try tool_calls first; if absent, try to extract from content.
        try:
            if has_tool_calls:
                return self._parse_native_tool_calls(result["tool_calls"])
            if text:
                return self._parse_content(text)
            return ParsedInferenceResult(content=text, raw_content=text)

        except Exception:
            # Fallback: unknown mode — behave like original logic
            if has_tool_calls:
                return self._parse_native_tool_calls(result["tool_calls"])
            if self.template_has_tool_token:
                return ParsedInferenceResult(content=text, raw_content=text)
            return self._parse_content(text)

    def _parse_native_tool_calls(
        self, tool_calls_raw: List[Dict[str, Any]]
    ) -> ParsedInferenceResult:
        """Normalise the tool_calls list returned by the inference API."""
        try:
            tool_calls = []
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = _parse_arguments(fn.get("arguments", {}))
                tool_calls.append({"name": name, "arguments": args})

            if tool_calls:
                return ParsedInferenceResult(
                    tool_calls=tool_calls,
                    raw_tool_calls=tool_calls_raw,
                )

            return ParsedInferenceResult(
                parse_error="tool_call list was empty after parsing",
                raw_tool_calls=tool_calls_raw,
            )

        except Exception as exc:
            log.warning("Failed to parse native tool_calls: %s", exc)
            return ParsedInferenceResult(
                parse_error=str(exc),
                raw_tool_calls=tool_calls_raw,
            )

    @abstractmethod
    def _parse_content(self, text: str) -> ParsedInferenceResult:
        """Try to extract a tool call from text, falling back to NLP."""
        ...


class JsonParser(BaseParser):
    """Handles models that emit tool calls as JSON inside the content field.

    Only called when template_has_tool_token=False. Uses parse_json_output()
    which handles:
      - {"name": "...", "arguments": {...}}
      - markdown-fenced JSON blocks
      - plain NLP text fallback
    """

    def _parse_content(self, text: str) -> ParsedInferenceResult:
        if not text.strip():
            return ParsedInferenceResult()

        try:
            parsed: ParsedResponse = parse_json_output(text)
            if parsed.type == "tool" and parsed.tools:
                return ParsedInferenceResult(
                    tool_calls=[tc.to_dict() for tc in parsed.tools],
                    raw_content=text,
                )
            return ParsedInferenceResult(
                content=parsed.response or text,
                raw_content=text,
            )
        except Exception as exc:
            log.warning("JsonParser failed: %s", exc)
            return ParsedInferenceResult(
                content=text,
                raw_content=text,
                parse_error=str(exc),
            )


class PythonicParser(BaseParser):
    """Handles models that emit tool calls in Pythonic function-call syntax.

    Only called when template_has_tool_token=False.

    e.g. [show_pie_chart(title="Spending", data_source={...})]

    Falls back to JSON parsing if the pythonic parser finds nothing.
    """

    def _parse_content(self, text: str) -> ParsedInferenceResult:
        if not text.strip():
            return ParsedInferenceResult()

        # 1. Pythonic syntax — the primary format for this parser
        pythonic_calls: Optional[List[ToolCall]] = parse_pythonic_output(text)
        if pythonic_calls:
            return ParsedInferenceResult(
                tool_calls=[tc.to_dict() for tc in pythonic_calls],
                raw_content=text,
            )

        # 2. JSON fallback — model may have used JSON format instead
        try:
            parsed: ParsedResponse = parse_json_output(text)
            if parsed.type == "tool" and parsed.tools:
                return ParsedInferenceResult(
                    tool_calls=[tc.to_dict() for tc in parsed.tools],
                    raw_content=text,
                )
            return ParsedInferenceResult(
                content=parsed.response or text,
                raw_content=text,
            )
        except Exception as exc:
            log.warning("PythonicParser JSON fallback failed: %s", exc)

        # 3. Plain NLP
        return ParsedInferenceResult(content=text, raw_content=text)


_PARSERS: Dict[str, type] = {
    "json": JsonParser,
    "pythonic": PythonicParser,
}


def get_parser(
    parser_type: str = "json",
    template_has_tool_token: bool = False,
) -> BaseParser:
    """Return a parser instance by name."""
    cls = _PARSERS.get(parser_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown parser type '{parser_type}'. Choose from: {list(_PARSERS)}"
        )
    return cls(template_has_tool_token=template_has_tool_token)


def parse_outputs(
    outputs: List[Dict[str, Any]],
    parser_type: str = "json",
    template_has_tool_token: bool = False,
) -> List[ParsedInferenceResult]:
    """Parse a flat list of inference result dicts into ParsedInferenceResults."""
    parser = get_parser(parser_type, template_has_tool_token=template_has_tool_token)
    results = []
    for output in outputs:
        try:
            results.append(parser.parse(output))
        except Exception as exc:
            log.error("Unexpected parse error for output %r: %s", output, exc)
            results.append(ParsedInferenceResult(parse_error=str(exc)))
    return results
