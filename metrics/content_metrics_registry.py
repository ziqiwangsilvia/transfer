import logging
from typing import Any, Dict, List, Optional

from evaluator.conversational_content.content_metrics import (
    compute_llm_judge_metrics_batch,
    get_bert_score_batch,
    get_levenshtein_distance_batch,
    get_rouge_score_batch,
)

log = logging.getLogger(__name__)

METRIC_FAMILY_FUNCS: Dict[str, Any] = {
    "rouge": get_rouge_score_batch,
    "bert": get_bert_score_batch,
    "levenshtein": get_levenshtein_distance_batch,
    "judge": compute_llm_judge_metrics_batch,
}

METRIC_FAMILY_OUTPUTS: Dict[str, List[str]] = {
    "rouge": ["rouge1", "rougeL"],
    "bert": ["bert_precision", "bert_recall", "bert_f1"],
    "levenshtein": ["levenshtein_distance", "levenshtein_ratio"],
    "judge": [
        "content_similarity",
        "context_accuracy",
        "content_politeness",
        "topic_accuracy",
        "content_helpfulness",
    ],
}


async def run_metric_families(
    family_list: List[str],
    predictions: List[str],
    references: List[str],
    **kwargs: Any,
) -> Dict[str, List[float]]:
    """Compute metric families and return a flattened dict of submetric -> values.

    The function is intentionally permissive with kwargs to support the judge
    metric which requires extra parameters (ground_truths, queries, judge_config,
    enabled_llm_judge_metrics). Non-judge metric functions are invoked with the
    common (predictions, references) signature.
    """
    results: Dict[str, List[float]] = {}

    enabled_llm_judge_metrics: List[str] = kwargs.get("enabled_llm_judge_metrics") or []

    for family in family_list:
        subkeys = METRIC_FAMILY_OUTPUTS.get(family, [])

        # Handle judge and non-judge families separately for type clarity
        if family == "judge":
            judge_fn = METRIC_FAMILY_FUNCS.get("judge")
            if judge_fn is None:
                log.warning("Judge metric family not found")
                continue
            out = await judge_fn(  # type: ignore[misc]
                ground_truths=kwargs.get("ground_truths"),
                predictions=predictions,  # type: ignore[arg-type]
                queries=kwargs.get("queries"),
                judge_config=kwargs.get("judge_config"),
                enabled_llm_judge_metrics=enabled_llm_judge_metrics,
            )
        else:
            fn = METRIC_FAMILY_FUNCS.get(family)
            if fn is None:
                log.warning("Unknown metric family '%s' — skipping", family)
                continue
            out = fn(predictions=predictions, references=references)  # type: ignore[misc]

        # Project only the subkeys we're interested in. For judge family we
        # only include the enabled judge submetrics.
        for subkey in subkeys:
            if family != "judge" or subkey in enabled_llm_judge_metrics:
                results[subkey] = out[subkey]

    return results


async def compute_all_metrics_batch(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    queries: List[str],
    scoring_config: Any,
    judge_config: Optional[Any] = None,
) -> List[Dict[str, float]]:
    """Compute configured content evaluation metrics in batch using registry pattern."""
    all_scores: List[Dict[str, float]] = [{} for _ in range(len(ground_truths))]

    # Collect NLP-only samples with non-empty texts and remember their indices
    nlp_indices: List[int] = []
    nlp_gts: List[str] = []
    nlp_preds: List[str] = []
    nlp_queries: List[str] = []

    for idx, (gt, pred, query) in enumerate(zip(ground_truths, predictions, queries)):
        if gt.get("type") == "nlp" and pred.get("type") == "nlp":
            gt_text = gt.get("response", "")
            pred_text = pred.get("response", "")
            if gt_text and pred_text:
                nlp_indices.append(idx)
                nlp_gts.append(gt_text)
                nlp_preds.append(pred_text)
                nlp_queries.append(query)

    if not nlp_indices:
        return all_scores

    # Resolve configured metric families with safe defaults
    enabled_metric_families = list(getattr(scoring_config, "metrics", []) or [])
    enabled_llm_judge_metrics = list(getattr(scoring_config, "llm_judge_metrics", []) or [])

    # Optionally include judge family
    families_to_compute: List[str] = list(enabled_metric_families)
    if judge_config and getattr(judge_config, "enabled", False) and enabled_llm_judge_metrics:
        families_to_compute.append("judge")

    # Compute all metrics using registry with pre-filtered NLP-only data
    if families_to_compute:
        metrics_dict = await run_metric_families(
            families_to_compute,
            predictions=nlp_preds,
            references=nlp_gts,
            ground_truths=nlp_gts,
            queries=nlp_queries,
            judge_config=judge_config,
            enabled_llm_judge_metrics=enabled_llm_judge_metrics,
        )

        # Distribute metrics to corresponding original indices
        for i, idx in enumerate(nlp_indices):
            all_scores[idx] = {}
            for metric_name, metric_values in metrics_dict.items():
                all_scores[idx][metric_name] = metric_values[i]

    return all_scores
