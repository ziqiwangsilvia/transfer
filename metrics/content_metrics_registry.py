from typing import Any, Dict, List, Optional

from evaluator.conversational_content.content_metrics import (
    compute_llm_judge_metrics_batch,
    get_bert_score_batch,
    get_levenshtein_distance_batch,
    get_rouge_score_batch,
)

METRIC_FAMILY_FUNCS = {
    "rouge": get_rouge_score_batch,
    "bert": get_bert_score_batch,
    "levenshtein": get_levenshtein_distance_batch,
    "judge": compute_llm_judge_metrics_batch,
}

METRIC_FAMILY_OUTPUTS = {
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


async def run_metric_families(family_list, predictions, references, **kwargs):
    """Compute metrics for specified families.

    Args:
        family_list: List of metric families to compute.
        predictions: List of predictions.
        references: List of references.
        **kwargs: Additional arguments (for judge: ground_truths, queries, judge_config, enabled_llm_judge_metrics).

    Returns:
        Dict mapping metric names to lists of scores.
    """
    results = {}

    for family in family_list:
        fn = METRIC_FAMILY_FUNCS[family]
        subkeys = METRIC_FAMILY_OUTPUTS[family]

        # Judge metrics have different signature
        if family != "judge":
            out = fn(predictions=predictions, references=references)
        else:
            out = await fn(
                ground_truths=kwargs.get("ground_truths"),
                predictions=predictions,
                queries=kwargs.get("queries"),
                judge_config=kwargs.get("judge_config"),
                enabled_llm_judge_metrics=kwargs.get("enabled_llm_judge_metrics"),
            )

        for subkey in subkeys:
            if family != "judge" or subkey in kwargs.get("enabled_llm_judge_metrics"):
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

    # Collect NLP-only samples and record their original positions so that
    # computed metrics can be placed back in the same order as the input
    nlp_indices, nlp_gts, nlp_preds, nlp_queries = [], [], [], []
    for idx, (gt, pred, query) in enumerate(zip(ground_truths, predictions, queries)):
        if gt.get("type") == "nlp" and pred.get("type") == "nlp":
            gt_text, pred_text = gt.get("response", ""), pred.get("response", "")
            if gt_text and pred_text:
                nlp_indices.append(idx)
                nlp_gts.append(gt_text)
                nlp_preds.append(pred_text)
                nlp_queries.append(query)

    if not nlp_indices:
        return all_scores

    # Determine which families to compute
    enabled_metric_families = scoring_config.metrics
    enabled_llm_judge_metrics = scoring_config.llm_judge_metrics

    # Include judge family if enabled
    families_to_compute = list(enabled_metric_families)
    if judge_config and judge_config.enabled and enabled_llm_judge_metrics:
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
