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

    except (jso