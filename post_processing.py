import json

def clean_value(v):
    """Deep cleans newlines from strings, lists, or dicts."""
    if isinstance(v, str):
        return v.replace("\n", " ").strip()
    if isinstance(v, dict):
        return {k: clean_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [clean_value(i) for i in v]
    return v

def parse_response(output):
    try:
        # 1. Parse JSON (automatically handles \/ and escaped \n)
        parsed = json.loads(output.strip())
        
        # 2. Deep clean the entire object to remove actual \n characters
        parsed = clean_value(parsed)

        if not isinstance(parsed, dict): return None

        # 3. Handle Tool Calls
        if "tool_calls" in parsed:
            tc = parsed["tool_calls"]
            # Ensure it's a list for the ToolCall constructor
            items = tc if isinstance(tc, list) else [tc]
            return ParsedResponse(
                "tool",
                tools=[ToolCall(
                    name=t.get("name", ""),
                    parameters=t.get("arguments") or t.get("args") or {}
                ) for t in items]
            )

        # 4. Handle NLP Content
        if "content" in parsed:
            return ParsedResponse("nlp", response=parsed["content"])

    except (json.JSONDecodeError, TypeError):
        pass
    return None


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
