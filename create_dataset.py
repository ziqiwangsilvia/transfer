import json
import re
import yaml
from pathlib import Path
from datasets import load_dataset


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def load_and_format_hf_subset(config: dict = None):
    """Load HF dataset and format to match data_schema.json structure.
    
    Args:
        config: Configuration dict from YAML. If None, loads from config.yaml
    """
    if config is None:
        config = load_config()
    
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    parse_cfg = config.get("parsing", {})
    log_cfg = config.get("logging", {})
    
    repo = dataset_cfg.get("repo", "liminghao1630/API-Bank")
    split = dataset_cfg.get("split", "test")
    sample_size = dataset_cfg.get("sample_size", 100)
    streaming = dataset_cfg.get("streaming", True)
    
    output_path = output_cfg.get("output_path", "formatted_dataset.json")
    indent = output_cfg.get("indent", 2)
    ensure_ascii = output_cfg.get("ensure_ascii", False)
    
    verbose = log_cfg.get("verbose", True)
    print_samples = log_cfg.get("print_samples", False)
    
    ds = load_dataset(repo, streaming=streaming, split=split)
    formatted = []

    if verbose:
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
        
        if print_samples and i < 3:
            print(f"\nSample {i}:\n  Query: {query[:100]}\n  Type: {ground_truth.get('type')}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=indent, ensure_ascii=ensure_ascii)

    if verbose:
        print(f"✅ Wrote {len(formatted)} formatted entries to {output_path}")


if __name__ == "__main__":
    load_and_format_hf_subset()
