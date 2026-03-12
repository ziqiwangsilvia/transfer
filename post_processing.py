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
    prediction: Dict[str, Any],
    raw_output_text: str,
    post_processing: bool = False,
) -> Tuple[Optional[float], float]:
    """
    Measure how accurately the model formatted its output before parsing.

    Scores:
      raw:       1.0 if the output was parseable without any cleanup,
                 None if post_processing=False (native tool call path, no raw text).
                 For NLP ground truth: 1.0 if model correctly gave NLP, else 0.0.
      processed: 1.0 if parsing ultimately yielded a valid tool call,
                 0.0 otherwise (or NLP: mirrors raw).
    """
    if ground_truth.get("type") == "nlp":
        score = 1.0 if prediction.get("type") == "nlp" else 0.0
        if not post_processing:
            return None, score
        return score, score

    processed = 1.0 if prediction.get("type") == "tool" else 0.0

    if not post_processing:
        return None, processed

    text = raw_output_text.strip()
    raw = 0.0
    if text:
        if _is_valid_tool_call_pythonic(text):
            raw = 1.0
        else:
            try:
                parsed = json.loads(text)
                if _is_valid_tool_call_json(parsed):
                    raw = 1.0
            except (json.JSONDecodeError, ValueError):
                pass
    else:
        # Native tool_call path — no raw text to inspect; treat as clean
        raw = processed

    return raw, processed