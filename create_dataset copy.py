import json
import re
import yaml
from pathlib import Path
from src.benchmark_subset.create_subset import load_and_format_hf_subset

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

if __name__ == "__main__":
    load_and_format_hf_subset()