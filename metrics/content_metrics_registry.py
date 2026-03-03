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


def run_metric_families(
    family_list,
    *,
    predictions,
    references,
    **kwargs
):
    results = {}

    for family in family_list:
        fn = METRIC_FAMILY_FUNCS[family]
        subkeys = METRIC_FAMILY_OUTPUTS[family]

        out = fn(predictions=predictions, references=references, **kwargs)

        for subkey in subkeys:
            metric_name = f"content_{subkey}"
            results[metric_name] = out[subkey]

    return results