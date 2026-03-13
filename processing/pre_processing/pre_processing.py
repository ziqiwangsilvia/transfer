import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


DATASET_CATEGORIES = ("complete", "incomplete", "nlp")


def convert_filter_format(
    data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Converts nested filter objects inside tool_call parameters to flat keys.

    Transforms:
      {"filter": {"categories": [...], "payees": [...]}}
    To:
      {"filter_categories": [...], "filter_payees": [...]}
    """
    return {
        category: [_convert_sample(sample) for sample in samples]
        for category, samples in data.items()
    }


def _convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Apply filter format conversion to all messages in a sample."""
    return {
        **sample,
        "messages": [_convert_message(m) for m in sample.get("messages", [])],
    }


def _convert_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Apply filter format conversion to a single message dict."""
    if "tool_calls" not in message:
        return message
    return {**message, "tool_calls": _convert_node(message["tool_calls"])}


def _convert_node(data: Any) -> Any:
    """Recursively convert filter dicts within a tool_call value."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "filter" and isinstance(value, dict):
                if "categories" in value:
                    result["filter_categories"] = value["categories"]
                if "payees" in value:
                    result["filter_payees"] = value["payees"]
            elif key == "data_source" and isinstance(value, dict):
                for ds_key, ds_value in value.items():
                    result[ds_key] = _convert_node(ds_value)
            else:
                result[key] = _convert_node(value)
        return result
    elif isinstance(data, list):
        return [_convert_node(item) for item in data]
    return data


def load_dataset(
    path: str,
    max_samples: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load the dataset from a JSON file.

    Returns:
        Dict mapping category name to list of raw sample dicts
        (each with uid and messages keys).
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(dataset_path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        log.warning(
            "Dataset root is %s, expected dict with keys %s",
            type(raw).__name__,
            DATASET_CATEGORIES,
        )
        raise TypeError(
            f"Dataset must be a JSON object with category keys, got {type(raw).__name__}"
        )

    wanted = set(categories) if categories else set(DATASET_CATEGORIES)
    result: Dict[str, List[Dict[str, Any]]] = {}

    for cat in DATASET_CATEGORIES:
        if cat not in wanted:
            continue
        samples = raw.get(cat, [])
        if not isinstance(samples, list):
            log.warning(
                "Category %r has type %s, expected list - skipping",
                cat,
                type(samples).__name__,
            )
            continue

        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]
        result[cat] = samples
        log.info(f"  {cat}: {len(samples)} samples")

    total = sum(len(v) for v in result.values())
    log.info(f"Loaded {total} samples total from {path}")
    return result


# Template rewriting


def _rewrite_messages(
    messages: List[Dict[str, Any]],
    template_has_tool_token: bool,
) -> List[Dict[str, Any]]:
    """Return a rewritten copy of messages for models without tool tokens.

    When template_has_tool_token is True the messages are returned as-is

    When False:
      - Assistant turns with a tool_call key are rewritten to
        {"role": "assistant", "content": <json string of tool_call>}
      - Tool turns are rewritten to
        {"role": "user", "content": <data summary>}
    """
    messages = deepcopy(messages)

    if template_has_tool_token:
        return messages

    rewritten = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and "tool_calls" in msg:
            rewritten.append(
                {
                    "role": "assistant",
                    "content": f"<tool_call>{json.dumps(msg['tool_calls'])}</tool_call>",
                }
            )
        elif role == "tool":
            rewritten.append(
                {
                    "role": "user",
                    "content": f"<tool_response>{msg.get('content', '')}</tool_response>",
                }
            )
        else:
            rewritten.append(msg)

    return rewritten


def prepare_records(
    dataset: Dict[str, List[Dict[str, Any]]],
    template_has_tool_token: bool = True,
    model_in_the_loop: bool = False,
) -> List[Dict[str, Any]]:
    """Convert the loaded dataset into a flat list of inference-ready records.

    Inference is triggered at:
      - Every user turn  (model predicts what comes next)
      - Every tool turn  (model summarises the tool result)

    Returns:
        Flat list of record dicts, each with keys:
        uid, category, context, ground_truth, turn_indices, gt_turn_index.
    """
    if model_in_the_loop:
        raise NotImplementedError(
            "model_in_the_loop=True is not yet implemented. "
            "End-to-end tool execution will be added in the future."
        )

    records: List[Dict[str, Any]] = []

    for category, samples in dataset.items():
        for sample in samples:
            uid = sample.get("uid", "")
            original_messages = sample.get("messages", [])

            if len(original_messages) < 2:
                log.warning(
                    f"Sample uid={uid!r} category={category!r} has fewer than "
                    "2 messages — skipping."
                )
                continue

            rewritten = _rewrite_messages(original_messages, template_has_tool_token)

            for i, original_turn in enumerate(original_messages):
                role = original_turn.get("role")

                if role == "user":
                    # Need at least one more message to use as GT
                    if i + 1 >= len(original_messages):
                        log.warning(
                            f"uid={uid!r} user turn at index {i} has no "
                            "following message — skipping."
                        )
                        continue

                    context = rewritten[: i + 1]
                    ground_truth = original_messages[i + 1]
                    turn_indices = list(range(i + 1))
                    gt_turn_index = i + 1

                elif role == "tool":
                    # Tool turn is both the last context item and the GT.
                    context = rewritten[: i + 1]
                    ground_truth = original_messages[i]
                    turn_indices = list(range(i + 1))
                    gt_turn_index = i

                else:
                    # Assistant turns: no inference, just part of context
                    continue

                records.append(
                    {
                        "uid": uid,
                        "category": category,
                        "context": context,
                        "ground_truth": ground_truth,
                        "turn_indices": turn_indices,
                        "gt_turn_index": gt_turn_index,
                    }
                )

    log.info(f"Prepared {len(records)} records for inference")
    return records


def get_contexts(records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Return just the context (input) lists, preserving order."""
    return [r["context"] for r in records]


def get_ground_truths(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return just the ground-truth message dicts, preserving order."""
    return [r["ground_truth"] for r in records]


def get_uids(records: List[Dict[str, Any]]) -> List[str]:
    """Return just the uid strings, preserving order."""
    return [r["uid"] for r in records]
