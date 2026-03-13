import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


def load_data(
    data_path: str,
    fields: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load guardrailing dataset from JSONL file.
    """
    df = pd.read_json(data_path, lines=True, nrows=max_samples)

    if fields:
        # Verify fields exist
        missing_fields = set(fields) - set(df.columns)
        if missing_fields:
            raise ValueError(f"Missing fields in dataset: {missing_fields}")
        df = df[fields]

    log.info(f"Loaded {len(df)} samples from {data_path}")
    return df


def load_system_prompt(filepath: str) -> str:
    """
    Load the system prompt/instructions from file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    word_count = len(prompt.split())
    log.info(f"Loaded system prompt ({word_count} words)")

    return prompt


def split_discarded(
    results: List[Dict[str, Any]], valid_labels: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split results into valid and discarded entries.
    """
    valid_entries = []
    discarded_entries = []

    valid_label_set = set(label.lower() for label in valid_labels)

    for result in results:
        # Normalise output
        pred = result.get("output", "").strip().lower()

        if pred in valid_label_set:
            valid_entries.append(result)
        else:
            discarded_entries.append(result)

    if discarded_entries:
        log.warning(
            f"{len(discarded_entries)}/{len(results)} items discarded "
            f"(invalid predictions)"
        )

    return valid_entries, discarded_entries


def dump_to_json(data: Any, output_path: str | Path) -> None:
    """
    Save data to JSON file.
    """
    file_path = Path(output_path)

    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved to {output_path}")


def save_predictions(
    output_dir: Path,
    dataset: pd.DataFrame,
    prompts: List[str],
    outputs: List[str],
    predictions: List[Dict[str, Any]],
    scores: Dict[str, float],
    save_format: str = "json",
    include_prompts: bool = True,
) -> None:
    """Save detailed predictions."""
    predictions_data = []

    for i, row in dataset.iterrows():
        idx = int(i)  # type: ignore[arg-type]
        pred = {
            "sample_id": i,
            "input": row.to_dict(),
            "output": outputs[idx],
            "parsed_output": predictions[idx],
        }

        if include_prompts:
            pred["prompt"] = prompts[idx]

        predictions_data.append(pred)

    # Add overall scores
    result = {"predictions": predictions_data, "overall_scores": scores}

    predictions_file = output_dir / f"predictions.{save_format}"
    with open(predictions_file, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"Saved predictions to {predictions_file}")


def save_per_item_results(
    output_dir: Path,
    dataset: pd.DataFrame,
    outputs: List[str],
    predictions: List[Dict[str, Any]],
    field_prompt: str = "prompt",
    field_label: str = "prompt_label",
    field_violated_categories: str = "violated_categories",
    include_raw_outputs: bool = True,
) -> None:
    """Save per-item results with ground truth vs prediction comparison."""
    per_item_results = []

    for i, row in dataset.iterrows():
        idx = int(i)  # type: ignore[arg-type]
        item_result = {
            "sample_id": idx,
            "prompt": row[field_prompt],
            "ground_truth_label": row[field_label],
            "predicted_label": predictions[idx].get("output", ""),
            "violated_categories": row.get(field_violated_categories, ""),
        }

        gt_label = str(item_result["ground_truth_label"])
        pred_label = str(item_result["predicted_label"])
        # Compute per-item correctness
        item_result["correct"] = gt_label.lower() == pred_label.lower()

        if include_raw_outputs:
            item_result["raw_output"] = outputs[idx]

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


def create_balanced_subset(
    dataframe: pd.DataFrame,
    target_col: str,
    output_filename: str,
    n_samples: int = 100,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Create a label-balanced dataset subset.
    """
    min_labelsize = dataframe[target_col].value_counts().min()
    n_samples = min(n_samples, min_labelsize * 2)

    # Even number for 50/50 split
    if n_samples % 2 != 0:
        n_samples -= 1

    balanced_df = dataframe.groupby(target_col).sample(
        n=n_samples // 2,
        random_state=random_seed,
    )

    assert n_samples == len(balanced_df), (
        f"Expected {n_samples}, got {len(balanced_df)}"
    )

    final_df = balanced_df.sample(n=n_samples, random_state=random_seed).reset_index(
        drop=True
    )

    # Save to JSONL
    final_df.to_json(output_filename, orient="records", lines=True)
    log.info(f"Created balanced subset: {n_samples} samples → {output_filename}")

    return final_df
