import json
import re
from typing import Any, Dict, List

from json_repair import repair_json

from processing.post_processing.utils import KNOWN_TOOL_NAMES, ParsedResponse, ToolCall


def _parse_arguments(raw_args: Any) -> Dict[str, Any]:
    """Parse an arguments field that may be a JSON string (OpenAI) or already a dict (vLLM/Llama)."""
    return json.loads(raw_args) if isinstance(raw_args, str) else raw_args


def _tool_calls_from_json_list(items: List[Any]) -> List[ToolCall]:
    """Extract ToolCalls from a parsed JSON list, handling all known formats."""
    tool_calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Tool-as-key format: {"show_line_chart": {...}}
        tool_key = next((k for k in item if k in KNOWN_TOOL_NAMES), None)
        if tool_key:
            tool_calls.append(ToolCall(name=tool_key, parameters=item[tool_key]))
        # Standard name/tool field
        elif "name" in item or "tool" in item:
            tool_calls.append(ToolCall.from_dict(item))
        # OpenAI function object: {"type": "function", "function": {"name": ..., "parameters": ...}}
        elif "function" in item:
            try:
                fn = item["function"]
                raw_args = fn.get("arguments") or fn.get("parameters") or {}
                tool_calls.append(
                    ToolCall(name=fn["name"], parameters=_parse_arguments(raw_args))
                )
            except Exception:
                continue
    return tool_calls


def parse_json_output(output: str) -> ParsedResponse:
    """
    Parse model output into a ParsedResponse using JSON strategies.

    Attempts in order:
    0. Structured outputs (WIP)
    1. Direct JSON parse (clean JSON output)
    2. Fence-stripped JSON parse (```json ... ``` wrapped output)
    3. Substring extraction — find the first JSON array or object in mixed content
    """
    output = output.strip()

    # --- Attempt 0: structured output ---
    # Attempt to fix truncated or repetitive JSON
    try:
        # repair_json returns a string; json.loads turns it into a dict
        repaired_string = repair_json(output, return_objects=False)
        parsed = json.loads(repaired_string)

        if isinstance(parsed, dict):
            # Check for tool_calls structure
            if "tool_calls" in parsed:
                tc = parsed["tool_calls"]
                # Handle cases where tool_calls might be a list or a dict
                if isinstance(tc, dict):
                    return ParsedResponse(
                        "tool",
                        tools=[
                            ToolCall(
                                name=tc.get("name", ""),
                                parameters=tc.get("arguments", tc.get("args", {})),
                            )
                        ],
                    )

            # If it's just a text response wrapped in JSON
            if "content" in parsed:
                return ParsedResponse("nlp", response=parsed["content"])

    except Exception:
        pass

    # --- Attempt 1 & 2: direct parse, then fence-stripped ---
    for text in [output, re.sub(r"```[\w_]*", "", output).replace("```", "").strip()]:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue

        if isinstance(parsed, dict):
            if "tool_calls" in parsed:
                tool_calls = []
                for tc in parsed["tool_calls"]:
                    try:
                        args = _parse_arguments(tc["function"]["arguments"])
                        tool_calls.append(
                            ToolCall(name=tc["function"]["name"], parameters=args)
                        )
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                if tool_calls:
                    return ParsedResponse("tool", tools=tool_calls)

            elif ("tool" in parsed or "name" in parsed) and "parameters" in parsed:
                return ParsedResponse("tool", tools=[ToolCall.from_dict(parsed)])

            # Tool-as-key format: {"show_line_chart": {...}}
            tool_key = next((k for k in parsed if k in KNOWN_TOOL_NAMES), None)
            if tool_key:
                return ParsedResponse(
                    "tool", tools=[ToolCall(name=tool_key, parameters=parsed[tool_key])]
                )

        elif isinstance(parsed, list):
            tool_calls = _tool_calls_from_json_list(parsed)
            if tool_calls:
                return ParsedResponse("tool", tools=tool_calls)

    # --- Attempt 3: extract JSON structure from mixed content ---
    try:
        cleaned = re.sub(r"```[\w_]*", "", output).replace("```", "").strip()
        arr_start, obj_start = cleaned.find("["), cleaned.find("{")
        candidates = [(i, c) for i, c in [(arr_start, "["), (obj_start, "{")] if i >= 0]
        if candidates:
            start_idx, start_char = min(candidates, key=lambda x: x[0])
            end_idx = cleaned.rfind("]" if start_char == "[" else "}") + 1
            if end_idx > start_idx:
                parsed = json.loads(cleaned[start_idx:end_idx])
                if isinstance(parsed, list):
                    tool_calls = _tool_calls_from_json_list(parsed)
                    if tool_calls:
                        return ParsedResponse("tool", tools=tool_calls)
                elif isinstance(parsed, dict):
                    # Recurse with the extracted object as clean JSON
                    result = parse_json_output(json.dumps(parsed))
                    if result.type == "tool":
                        return result
    except Exception:
        pass

    return ParsedResponse("nlp", response=output)
