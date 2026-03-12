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


from json_repair import repair_json
import json

def parse_json_output(output: str) -> ParsedResponse:
    output = output.strip()
    
    # Attempt to fix truncated or repetitive JSON
    try:
        # repair_json returns a string; json.loads turns it into a dict
        repaired_string = repair_json(output)
        parsed = json.loads(repaired_string)
        
        if isinstance(parsed, dict):
            # Check for tool_calls structure
            if "tool_calls" in parsed:
                tc = parsed["tool_calls"]
                # Handle cases where tool_calls might be a list or a dict
                if isinstance(tc, dict):
                    return ParsedResponse(
                        "tool",
                        tools=[ToolCall(
                            name=tc.get("name", ""),
                            parameters=tc.get("arguments", tc.get("args", {})),
                        )],
                    )
            
            # If it's just a text response wrapped in JSON
            if "content" in parsed:
                return ParsedResponse("nlp", response=parsed["content"])
                
    except Exception:
        pass
    
    # ... your existing fallback logic
