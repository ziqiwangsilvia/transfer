import json
import re
import yaml
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Args:
    """Configuration arguments for dataset loading and formatting."""
    repo: str = "liminghao1630/API-Bank"
    split: str = "test"
    sample_size: int = 100
    streaming: bool = True
    output_path: str = "formatted_dataset.json"
    indent: int = 2
    ensure_ascii: bool = False
    verbose: bool = True
    print_samples: bool = False


def parse_args(config: dict = None) -> Args:
    """Parse and extract arguments from configuration.
    
    Args:
        config: Configuration dict from YAML. If None, loads from config.yaml
        
    Returns:
        Args dataclass instance with parsed configuration values
    """
    if config is None:
        config = load_config()
    
    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    log_cfg = config.get("logging", {})
    
    # Map config keys to their source sections
    config_map = {
        "repo": dataset_cfg,
        "split": dataset_cfg,
        "sample_size": dataset_cfg,
        "streaming": dataset_cfg,
        "output_path": output_cfg,
        "indent": output_cfg,
        "ensure_ascii": output_cfg,
        "verbose": log_cfg,
        "print_samples": log_cfg,
    }
    
    kwargs = {key: cfg[key] for key, cfg in config_map.items() if key in cfg}
    
    return Args(**kwargs)


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
    args = parse_args(config)
    
    ds = load_dataset(args.repo, streaming=args.streaming, split=args.split)
    formatted = []

    if args.verbose:
        print(f"🚀 Streaming from {args.repo} and formatting {args.sample_size} examples...")

    for i, entry in enumerate(ds):
        if i >= args.sample_size:
            break

        # Common fields to use as query
        query = entry.get("input")
        instruction = entry.get("instruction")

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

        formatted.append({"query": query, "instruction": instruction, "ground_truth": ground_truth})

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=args.indent, ensure_ascii=args.ensure_ascii)

    if args.verbose:
        print(f"✅ Wrote {len(formatted)} formatted entries to {args.output_path}")


if __name__ == "__main__":
    load_and_format_hf_subset()
