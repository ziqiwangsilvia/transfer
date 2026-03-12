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
