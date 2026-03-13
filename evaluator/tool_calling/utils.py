import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

log = logging.getLogger(__name__)

# Schema loading and formatting


def load_tools(tool_config: Any) -> List[Dict[str, Any]]:
    """Load tool definitions from config. Returns list of raw tool schemas."""
    if tool_config.tools:
        return tool_config.tools

    # Load from schema file
    if tool_config.schema_path:
        schema_path = Path(tool_config.schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Tool schema not found: {tool_config.schema_path}")

        with open(schema_path, "r") as f:
            if schema_path.suffix == ".json":
                tools = json.load(f)
            else:
                import yaml

                tools = yaml.safe_load(f)

        # Handle both {"tools": [...]} and [...] formats
        if isinstance(tools, dict) and "tools" in tools:
            return tools["tools"]
        return tools

    log.warning("No tools defined in configuration")
    return []


def _resolve_refs(obj: Any, defs: Dict[str, Any]) -> Any:
    """Recursively resolve $ref references in a JSON schema."""
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_path = obj["$ref"]
            def_name = ref_path.split("/")[-1]
            if def_name in defs:
                return _resolve_refs(defs[def_name], defs)
            return obj
        return {k: _resolve_refs(v, defs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(item, defs) for item in obj]
    return obj


def format_schemas_for_vllm(schemas: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format resolved tool schemas into the vLLM / OpenAI tools list format."""

    vllm_tools = []
    for tool_name, schema in schemas.items():
        vllm_tools.append(
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get(
                        "description",
                        f"Display {tool_name.replace('_', ' ')} visualisation",
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", []),
                    },
                },
            }
        )
    return vllm_tools


def load_schemas_from_json(
    schema_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Load and resolve tool schemas from a list of raw JSON schema dicts.

    Args:
        schema_list: List of schema dicts from financial_tools_schema.json

    Returns:
        Dict mapping tool names to resolved schemas.
    """
    schemas = {}
    for entry in schema_list:
        schema = entry.get("function", entry)

        name = schema["name"]
        params = schema.get("parameters", {})
        properties = params.get("properties", {})
        defs = schema.get("$defs", {})
        if defs:
            properties = _resolve_refs(properties, defs)

        schemas[name] = {
            "name": name,
            "description": schema.get("description", ""),
            "properties": properties,
            "required": params.get("required", list(properties.keys())),
        }
    return schemas


# Result Saving


def save_predictions(
    output_dir: Path,
    records: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    scores: List[Dict[str, float]],
    save_format: str = "json",
    include_scores: bool = True,
) -> None:
    """Save full predictions with context, output, and scores."""
    predictions_data = []
    for record, output, prediction, score in zip(records, outputs, predictions, scores):
        entry = {
            "uid": record.get("uid", ""),
            "category": record.get("category", ""),
            "turn_indices": record.get("turn_indices", []),
            "gt_turn_index": record.get("gt_turn_index", None),
            "context": record["context"],
            "ground_truth": record["ground_truth"],
            "output": output,
            "parsed_output": prediction,
        }
        if include_scores:
            entry["scores"] = score
        predictions_data.append(entry)

    predictions_file = output_dir / f"predictions.{save_format}"
    with open(predictions_file, "w") as f:
        json.dump(predictions_data, f, indent=2)
    log.info(f"Saved predictions to {predictions_file}")


def save_per_item_results(
    output_dir: Path,
    records: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    scores: List[Dict[str, float]],
    include_raw_outputs: bool = True,
) -> None:
    """Save per-item results as parquet and CSV for analysis.

    GT type derivation:
      - "tool_calls" in gt  -> gt_type "tool" (assistant predicting a tool call)
      - role "tool" ->  gt_type "nlp"     (model summarising a tool result - NLP)
      - role "assistant" with "content" ->  gt_type "nlp"  (plain text response)
    """
    per_item_results = []

    for record, output, prediction, score in zip(records, outputs, predictions, scores):
        gt = record["ground_truth"]

        # Derive GT type and value from message dict keys
        if "tool_calls" in gt:
            gt_type = "tool"
            gt_value = json.dumps(gt["tool_calls"])
        else:
            gt_type = "nlp"
            gt_value = gt.get("content", "")

        # Preserve original role analysis
        gt_role = gt.get("role", "")

        item_result = {
            "uid": record.get("uid", ""),
            "category": record.get("category", ""),
            "turn_indices": json.dumps(record.get("turn_indices", [])),
            "gt_turn_index": record.get("gt_turn_index", None),
            "gt_type": gt_type,
            "gt_role": gt_role,
            "gt_value": gt_value,
            "pred_type": prediction.get("type", ""),
            "pred_tools": json.dumps(prediction.get("tools", [])),
            "pred_response": prediction.get("response", ""),
            "pred_error": prediction.get("error", ""),
        }

        for metric, value in score.items():
            item_result[f"score_{metric}"] = value

        if include_raw_outputs:
            item_result["raw_output"] = json.dumps(output)

        per_item_results.append(item_result)

    df = pd.DataFrame(per_item_results)

    # Save as parquet
    parquet_file = output_dir / "per_item_results.parquet"
    df.to_parquet(parquet_file, index=False)
    log.info(f"Saved per-item results to {parquet_file}")

    # Save to CSV for debugging
    csv_file = output_dir / "per_item_results.csv"
    df.to_csv(csv_file, index=False)
    log.info(f"Saved per-item results to {csv_file}")
