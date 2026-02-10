import json
import re
from datasets import load_dataset


def parse_api_request(text: str):
    """Parse strings like:
    API-Request: [ModifyRegistration(appointment_id='34567890', new_appointment_date='2023-03-26', ...)]
    Returns (name, params_dict) or (None, None) if not matched.
    """
    if not text:
        return None, None
    m = re.search(r"\[([A-Za-z0-9_]+)\((.*)\)\]", text)
    if not m:
        return None, None
    name = m.group(1)
    params_str = m.group(2)
    params = {}
    for p in re.finditer(r"(\w+)=('([^']*)'|\"([^\"]*)\")", params_str):
        key = p.group(1)
        val = p.group(3) if p.group(3) is not None else p.group(4)
        params[key] = val
    return name, params


def load_and_format_hf_subset(repo="liminghao1630/API-Bank", sample_size=100, output_path="formatted_dataset.json"):
    ds = load_dataset(repo, streaming=True, split="test")
    formatted = []

    print(f"🚀 Streaming from {repo} and formatting {sample_size} examples...")

    for i, entry in enumerate(ds):
        if i >= sample_size:
            break

        # Common fields to use as query
        query = entry.get("query") or entry.get("instruction") or entry.get("input") or entry.get("prompt") or ""

        # Prefer explicit expected_output, fall back to answer/response
        expected = entry.get("expected_output") or entry.get("answer") or entry.get("output") or ""

        # Detect API-Request pattern and convert to tool format
        if isinstance(expected, str) and "API-Request" in expected:
            name, params = parse_api_request(expected)
            if name:
                ground_truth = {"type": "tool", "tools": [{"tool": name, "parameters": params}]}
            else:
                ground_truth = {"type": "tool", "tools": [{"tool": "unknown", "parameters": {"raw": expected}}]}
        else:
            # Treat as plain nlp response
            resp_text = expected if expected else entry.get("answer", "")
            ground_truth = {"type": "nlp", "response": resp_text}

        formatted.append({"query": query, "ground_truth": ground_truth})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(formatted)} formatted entries to {output_path}")


if __name__ == "__main__":
    load_and_format_hf_subset(sample_size=10)
