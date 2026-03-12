import json
import re
from typing import Any, Dict, List

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
    try:
        output_structured = output.strip().replace("\\/", "/")
        parsed = json.loads(output_structured)
        if isinstance(parsed, dict):
            if "tool_calls" in parsed and isinstance(parsed, dict):
                tc = parsed["tool_calls"]
                return ParsedResponse(
                    "tool",
                    tools = [ToolCall(
                        name=tc.get("name", ""),
                        parameters=tc.get("arguments", tc.get("args", {})),
                    )],
                )
            if "content" in parsed and len(parsed) == 1:
                return ParsedResponse("nlp", response=parsed["content"])        

    except (json.JSONDecodeError, ValueError):
        pass



def get_schema_reliability(
    ground_truth: Dict[str, Any],
    processed_output: Optional[Dict[str, Any]] = None,
    raw_output_text: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates schema reliability for both raw and processed outputs.
    
    Returns:
        (schema_reliability_raw, schema_reliability_processed)
    """
    # Only calculate for tool calls
    if ground_truth.get("type") != "tool":
        return None, None

    # 1. Schema Reliability for Raw Output (Before any cleanup/stripping)
    schema_reliability_raw = None
    if raw_output_text is not None:
        text = raw_output_text.strip()
        if not text:
            # Reverted: assume native tool path with no text is 'clean'
            schema_reliability_raw = 1.0
        else:
            try:
                # Valid only if it parses without .replace('\\/', '/') or \n stripping
                json.loads(text)
                schema_reliability_raw = 1.0
            except (json.JSONDecodeError, ValueError):
                # Fallback check for pythonic format
                schema_reliability_raw = 1.0 if _is_valid_tool_call_pythonic(text) else 0.0

    # 2. Schema Reliability for Processed Output (After your cleanup logic)
    schema_reliability_processed = None
    if processed_output is not None:
        # 1.0 if the final processed dictionary is a valid tool call
        schema_reliability_processed = 1.0 if processed_output.get("type") == "tool" else 0.0

    return schema_reliability_raw, schema_reliability_processed


"output": {
      "role": "assistant",
      "content": "{\"tool_calls\": {\"name\": \"show_pie_chart\", \"args\": {\"data_type\": \"spending\", \"time_range\": \"this_month\", \"group_by\": \"payee\", \"categories\": [\"Restaurants\", \"Taxi & Car sharing\", \"Public transport\", \"Groceries\", \"Events\", \"Rent\\/Mortgage\", \"Utilities\", \"Insurance\", \"Subscriptions\", \"Hobbies\", \"Clothing & Shoes\", \"Electronics\", \"Home & Garden\", \"Medical\", \"Groceries\", \"Fuel\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groc"
    },
    "parsed_output": {
      "type": "nlp",
      "response": "{\"tool_calls\": {\"name\": \"show_pie_chart\", \"args\": {\"data_type\": \"spending\", \"time_range\": \"this_month\", \"group_by\": \"payee\", \"categories\": [\"Restaurants\", \"Taxi & Car sharing\", \"Public transport\", \"Groceries\", \"Events\", \"Rent\\/Mortgage\", \"Utilities\", \"Insurance\", \"Subscriptions\", \"Hobbies\", \"Clothing & Shoes\", \"Electronics\", \"Home & Garden\", \"Medical\", \"Groceries\", \"Fuel\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groceries\", \"Groc"
    },


"output": {
      "role": "assistant",
      "content": "{\"tool_calls\": {\"name\": \"show_line_chart\", \"args\": {\"chart_type\": \"cash_flow\", \"time_range\": \"this_year\"\n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n    \n"
    },
    "parsed_output": {
      "type": "nlp",
      "response": "{\"tool_calls\": {\"name\": \"show_line_chart\", \"args\": {\"chart_type\": \"cash_flow\", \"time_range\": \"this_year\""
    },