from evaluator.conversational_content.content_metrics import (
    get_rouge_score_batch,
    get_bert_score_batch,
    get_levenshtein_distance_batch,
    compute_llm_judge_metrics_batch,
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

    # Judge *subkeys only* (post-processed into content_<subkey>)
    "judge": ["similarity", "context_accuracy", "politeness", "topic_accuracy", "helpfulness"],
}


async def run_metric_families(
    family_list,
    predictions,
    references,
    **kwargs
):
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
        if family == "judge":
            out = await fn(
                ground_truths=kwargs.get("ground_truths"),
                predictions=predictions,
                queries=kwargs.get("queries"),
                judge_config=kwargs.get("judge_config"),
                enabled_llm_judge_metrics=kwargs.get("enabled_llm_judge_metrics"),
            )
        else:
            out = fn(predictions=predictions, references=references, **kwargs)

        for subkey in subkeys:
            metric_name = f"content_{subkey}"
            results[metric_name] = out[subkey]

    return results