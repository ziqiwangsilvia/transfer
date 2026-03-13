import ast
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def _filter_ignored_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ignored parameters from tool parameters before scoring."""
    IGNORED_PARAMS = {"title"}
    if not params:
        return params

    filtered = {}
    for key, value in params.items():
        if key in IGNORED_PARAMS:
            continue
        if isinstance(value, dict):
            filtered[key] = _filter_ignored_params(value)
        else:
            filtered[key] = value
    return filtered


def get_when2call(ground_truth: Dict[str, Any], prediction: Dict[str, Any]) -> float:
    """Measure whether the model correctly decided to call a tool or respond in text.

    Returns 1.0 for a correct decision, 0.0 for incorrect.
    """
    gt_type = ground_truth.get("type", "error")
    pred_type = prediction.get("type", "error")
    return 1.0 if gt_type == pred_type else 0.0


def _is_valid_tool_call_json(parsed: Any) -> bool:
    """Check if a parsed JSON object represents a valid tool call structure."""
    if isinstance(parsed, dict):
        has_name_field = "name" in parsed
        has_args = "arguments" in parsed
        has_tool_calls = "tool_calls" in parsed
        return (has_name_field and has_args) or has_tool_calls
    if isinstance(parsed, list):
        return all("name" in item and "arguments" in item for item in parsed)
    return False


def _is_valid_tool_call_pythonic(text: Any) -> bool:
    """Check if text is valid pythonic tool call syntax."""
    try:
        tree = ast.parse(text, mode="eval")
        body = tree.body
        if isinstance(body, ast.List):
            return any(isinstance(el, ast.Call) for el in body.elts)
        if isinstance(body, ast.Call):
            return True
        return False
    except SyntaxError:
        return False


def get_schema_reliability(
    ground_truth: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
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
            # when tool call token exists and content is empty
            schema_reliability_raw = 1.0
        else:
            try:
                # Valid only if it parses without .replace('\\/', '/') or \n stripping
                json.loads(text)
                schema_reliability_raw = 1.0
            except (json.JSONDecodeError, ValueError):
                # Fallback check for pythonic format
                schema_reliability_raw = (
                    1.0 if _is_valid_tool_call_pythonic(text) else 0.0
                )

    # 2. Schema Reliability for Processed Output (After your cleanup logic)
    schema_reliability_processed = None
    if prediction is not None:
        # 1.0 if the final processed dictionary is a valid tool call
        schema_reliability_processed = 1.0 if prediction.get("type") == "tool" else 0.0

    return schema_reliability_raw, schema_reliability_processed


def get_tool_pickup_and_hallucination(
    ground_truth: Dict[str, Any],
    prediction: Dict[str, Any],
    available_tools: List[str],
) -> Tuple[Any, Any, Any]:
    """
    Measure whether the model picks the right tool and doesn't hallucinate.

    Returns:
        tool_pick_up_rate: fraction of GT tools the model predicted
        tool_hallucination_rate: fraction of predicted tools not in available set
        tool_additional_rate: fraction of valid-but-unrequested tools added
    """
    # NLP responses
    if ground_truth.get("type") != "tool" or prediction.get("type") != "tool":
        return None, None, None

    gt_tools = ground_truth.get("tools", [])
    pred_tools = prediction.get("tools", [])

    gt_tool_names = set(t.get("name", "") for t in gt_tools)
    pred_tool_names = set(t.get("name", "") for t in pred_tools)
    available_tool_set = set(available_tools)

    correct_tools = gt_tool_names & pred_tool_names
    hallucinated_tools = pred_tool_names - available_tool_set
    additional_tools = (pred_tool_names & available_tool_set) - gt_tool_names

    n = len(gt_tools)
    return (
        len(correct_tools) / n,
        len(hallucinated_tools) / n,
        len(additional_tools) / n,
    )


def get_variable_parsing_and_hallucination(
    ground_truth: Dict[str, Any],
    prediction: Dict[str, Any],
    tool_schemas: Dict[str, Dict[str, Any]],
) -> Tuple[Any, Any, Any, Any]:
    """Measure whether the model extracts the right parameters with correct values.

    Returns:
        variable_pickup_rate: fraction of GT params the model attempted
        variable_correct_rate: fraction of GT params set with correct value
        variable_hallucination_rate: fraction of invalid params predicted
        variable_additional_rate: fraction of valid-but-unrequested params added
    """
    # NLP responses
    if ground_truth.get("type") != "tool" or prediction.get("type") != "tool":
        return None, None, None, None

    gt_tools = ground_truth.get("tools", [])
    pred_tools = prediction.get("tools", [])

    gt_by_name = {t.get("name"): t for t in gt_tools}
    pred_by_name = {t.get("name"): t for t in pred_tools}

    total_pickup = 0.0
    total_correct = 0.0
    total_hallucinated = 0.0
    total_additional = 0.0
    total_tools = len(gt_tools)

    for tool_name, gt_tool in gt_by_name.items():
        if tool_name not in pred_by_name:
            continue

        pred_tool = pred_by_name[tool_name]
        gt_params = _filter_ignored_params(gt_tool.get("arguments", {}))
        pred_params = _filter_ignored_params(pred_tool.get("arguments", {}))

        tool_schema = tool_schemas.get(tool_name, {})
        valid_params = set(tool_schema.get("properties", {}).keys()) - {"title"}

        # Flatten nested params
        gt_flat = _flatten_dict(gt_params)
        pred_flat = _flatten_dict(pred_params)

        n_pickup = 0
        n_correct = 0
        n_hallucinated = 0
        n_additional = 0
        total_params = len(gt_flat)

        for key, pred_value in pred_flat.items():
            # Get base param name
            base_key = key.split(".")[0] if "." in key else key

            # Hallucinated param
            if base_key not in valid_params:
                n_hallucinated += 1
                continue

            # Check if param is in ground truth
            if key in gt_flat:
                n_pickup += 1
                if _values_match(gt_flat[key], pred_value):
                    n_correct += 1
            else:
                n_additional += 1

        total_pickup += n_pickup / total_params
        total_correct += n_correct / total_params
        total_hallucinated += n_hallucinated / total_params
        total_additional += n_additional / total_params

    return (
        total_pickup / total_tools,
        total_correct / total_tools,
        total_hallucinated / total_tools,
        total_additional / total_tools,
    )


def get_exact_match(
    ground_truth: Dict[str, Any], prediction: Dict[str, Any]
) -> Optional[float]:
    """Check if prediction exactly matches ground truth tool calls."""
    if ground_truth.get("type") != "tool" or prediction.get("type") != "tool":
        return None

    gt_tools = ground_truth.get("tools", [])
    pred_tools = prediction.get("tools", [])

    if len(gt_tools) != len(pred_tools):
        return 0.0

    gt_sorted = sorted(gt_tools, key=lambda x: x.get("name", ""))
    pred_sorted = sorted(pred_tools, key=lambda x: x.get("name", ""))

    for gt, pred in zip(gt_sorted, pred_sorted):
        if not _tools_match_exactly(gt, pred):
            return 0.0

    return 1.0


def compute_all_metrics(
    ground_truth: Dict[str, Any],
    prediction: Dict[str, Any],
    raw_output: Dict[str, Any],
    available_tools: List[str],
    tool_schemas: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all tool-calling evaluation metrics for a single sample."""
    # Extract raw text for schema_reliability:
    # native tool_call path has no text to inspect; content path uses the string.
    raw_output_text = raw_output.get("content", "") if raw_output is not None else None

    scores: Dict[str, Any] = {}
    scores["when2call"] = get_when2call(ground_truth, prediction)
    scores["schema_reliability_raw"], scores["schema_reliability_processed"] = (
        get_schema_reliability(ground_truth, prediction, raw_output_text)
    )
    (
        scores["tool_pick_up_rate"],
        scores["tool_hallucination_rate"],
        scores["tool_additional_rate"],
    ) = get_tool_pickup_and_hallucination(ground_truth, prediction, available_tools)
    (
        scores["variable_pickup_rate"],
        scores["variable_correct_rate"],
        scores["variable_hallucination_rate"],
        scores["variable_additional_rate"],
    ) = get_variable_parsing_and_hallucination(ground_truth, prediction, tool_schemas)
    scores["exact_match"] = get_exact_match(ground_truth, prediction)

    return scores


# Helpers


def _tools_match_exactly(gt_tool: Dict[str, Any], pred_tool: Dict[str, Any]) -> bool:
    """Check if two tool calls match exactly, ignoring additional predicted parameters."""
    if gt_tool.get("name") != pred_tool.get("name"):
        return False
    gt_params = _filter_ignored_params(gt_tool.get("arguments", {}))
    pred_params = _filter_ignored_params(pred_tool.get("arguments", {}))
    gt_flat = _flatten_dict(gt_params)
    pred_flat = _flatten_dict(pred_params)

    # Ignore additional parameters in prediciton
    for key, gt_value in gt_flat.items():
        if key not in pred_flat:
            return False
        if not _values_match(gt_value, pred_flat[key]):
            return False

    return True


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dict for comparison."""
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to JSON strings for comparison
            items.append((new_key, json.dumps(v, sort_keys=True)))
        else:
            items.append((new_key, v))

    return dict(items)


def _values_match(gt_value: Any, pred_value: Any) -> bool:
    """Check if two parameter values match."""
    # Handle None
    if gt_value is None or pred_value is None:
        return gt_value == pred_value

    # String comparison (case-insensitive)
    if isinstance(gt_value, str) and isinstance(pred_value, str):
        return gt_value.lower() == pred_value.lower()

    # Numeric comparison
    if isinstance(gt_value, (int, float)) and isinstance(pred_value, (int, float)):
        return abs(gt_value - pred_value) < 1e-9

    # Exact match for other types
    return gt_value == pred_value
