import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import evaluate
import Levenshtein
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

log = logging.getLogger(__name__)


def get_levenshtein_distance_batch(
    predictions: List[str], references: List[str]
) -> Dict[str, List[float]]:
    """
    Compute Levenshtein distance and ratio in batch.
    Currently loops due to no native batch support
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length: "
            f"{len(predictions)} vs {len(references)}"
        )

    distances = []
    ratios = []
    for pred, ref in zip(predictions, references):
        distances.append(Levenshtein.distance(pred, ref))
        ratios.append(Levenshtein.ratio(pred, ref))
    return {"levenshtein_distance": distances, "levenshtein_ratio": ratios}


def get_rouge_score_batch(
    predictions: List[str], references: List[str]
) -> Dict[str, List[float]]:
    """Compute ROUGE-1 and ROUGE-L scores in batch."""
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length: "
            f"{len(predictions)} vs {len(references)}"
        )

    rouge_scorer = cast(Any, evaluate.load("rouge"))
    scores = rouge_scorer.compute(
        predictions=predictions, references=references, use_aggregator=False
    )
    rouge1 = (
        [scores["rouge1"]] * len(predictions)
        if isinstance(scores["rouge1"], float)
        else scores["rouge1"]
    )
    rougeL = (
        [scores["rougeL"]] * len(predictions)
        if isinstance(scores["rougeL"], float)
        else scores["rougeL"]
    )
    return {"rouge1": rouge1, "rougeL": rougeL}


def get_bert_score_batch(
    predictions: List[str], references: List[str]
) -> Dict[str, List[float]]:
    """Compute BERTScore precision, recall, and F1 in batch."""
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references must have same length: "
            f"{len(predictions)} vs {len(references)}"
        )

    bert_scorer = cast(Any, evaluate.load("bertscore"))
    results = bert_scorer.compute(
        predictions=predictions, references=references, lang="en"
    )
    return {
        "bert_precision": results["precision"],
        "bert_recall": results["recall"],
        "bert_f1": results["f1"],
    }


def _parse_judge_output(output: str, output_type: str) -> float:
    """Parse judge model output based on expected type.

    Returns raw scores:
    - binary: 0.0 or 1.0
    - score_1_5: 1.0 to 5.0
    """
    output = output.strip()
    if output_type == "binary":
        # Expected: 0 or 1
        try:
            return 1.0 if float(output) > 0.5 else 0.0
        except ValueError:
            log.warning(
                f"Failed to parse binary judge output: {output!r}, returning 0.0"
            )
            return 0.0
    elif output_type == "score_1_5":
        try:
            return max(1.0, min(5.0, float(output)))
        except ValueError:
            log.warning(
                f"Failed to parse score judge output: {output!r}, returning 3.0"
            )
            return 3.0
    return 0.0


async def _run_judge_openai_batch(
    prompts: List[Tuple[str, str]], judge_config: Any
) -> List[str]:
    """Run all judge prompts concurrently against the OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    client = AsyncOpenAI(api_key=api_key)
    try:
        semaphore = asyncio.Semaphore(judge_config.batch_size)

        async def call_single(system_prompt: str, user_prompt: str) -> str:
            async with semaphore:
                try:
                    response = await client.chat.completions.create(
                        model=judge_config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=judge_config.temperature,
                        max_tokens=judge_config.max_tokens,
                    )
                    return response.choices[0].message.content or ""
                except Exception as e:
                    log.error(f"OpenAI API error: {e}")
                    return ""

        return list(
            await tqdm_asyncio.gather(
                *[call_single(s, u) for s, u in prompts],
                desc="LLM Judge (Content Metrics)",
                unit="sample",
                colour="magenta",
            )
        )
    finally:
        await client.close()


async def compute_llm_judge_metrics_batch(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    queries: List[str],
    judge_config: Any,
    enabled_llm_judge_metrics: List[str],
) -> List[Dict[str, float]]:
    """Compute LLM-as-judge metrics via OpenAI for all enabled metrics."""
    all_scores: List[Dict[str, float]] = [{} for _ in range(len(ground_truths))]

    if not judge_config or not judge_config.enabled:
        return all_scores

    # Collect NLP-only samples
    nlp_indices, nlp_gts, nlp_preds, nlp_queries = [], [], [], []
    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        if gt.get("type") == "nlp" and pred.get("type") == "nlp":
            gt_text, pred_text = gt.get("response", ""), pred.get("response", "")
            if gt_text and pred_text:
                nlp_indices.append(idx)
                nlp_gts.append(gt_text)
                nlp_preds.append(pred_text)
                nlp_queries.append(queries[idx])

    if not nlp_indices:
        return all_scores

    async def run_single_metric(metric_name: str):
        if metric_name not in judge_config.prompts:
            log.warning(f"Judge prompt '{metric_name} not found in config, skipping")
            return

        prompt_config = judge_config.prompts[metric_name]
        log.info(f"Computing judge metric: {prompt_config.metric_name}")

        judge_prompts = [
            (
                prompt_config.system_prompt,
                prompt_config.user_prompt.format(
                    reference=gt, prediction=pred, query=q
                ),
            )
            for gt, pred, q in zip(nlp_gts, nlp_preds, nlp_queries)
        ]

        raw_outputs = await _run_judge_openai_batch(judge_prompts, judge_config)

        for i, raw_output in enumerate(raw_outputs):
            score = _parse_judge_output(raw_output, prompt_config.output_type)
            all_scores[nlp_indices[i]][prompt_config.metric_name] = score

    await asyncio.gather(*[run_single_metric(m) for m in enabled_llm_judge_metrics])

    return all_scores


async def compute_all_metrics_batch(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    queries: List[str],
    scoring_config: Any,
    judge_config: Optional[Any] = None,
) -> List[Dict[str, float]]:
    """Compute configured content evaluation metrics in batch using registry pattern."""
    from evaluator.conversational_content.content_metrics_registry import (
        run_metric_families,
    )

    all_scores: List[Dict[str, float]] = [{} for _ in range(len(ground_truths))]

    # Collect NLP-only samples
    nlp_indices, nlp_gts, nlp_preds = [], [], []
    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        if gt.get("type") == "nlp" and pred.get("type") == "nlp":
            gt_text, pred_text = gt.get("response", ""), pred.get("response", "")
            if gt_text and pred_text:
                nlp_indices.append(idx)
                nlp_gts.append(gt_text)
                nlp_preds.append(pred_text)

    if not nlp_indices:
        return all_scores

    # Compute non-judge metrics using registry pattern
    enabled_metric_families = scoring_config.metrics
    if enabled_metric_families:
        metrics_dict = run_metric_families(
            enabled_metric_families, predictions=nlp_preds, references=nlp_gts
        )

        # Distribute metrics to corresponding sample indices
        for i, idx in enumerate(nlp_indices):
            all_scores[idx] = {}
            for metric_name, metric_values in metrics_dict.items():
                all_scores[idx][metric_name] = metric_values[i]

    # Compute judge metrics separately (async)
    enabled_llm_judge_metrics = scoring_config.llm_judge_metrics
    if judge_config and judge_config.enabled and enabled_llm_judge_metrics:
        judge_scores = await compute_llm_judge_metrics_batch(
            ground_truths=ground_truths,
            predictions=predictions,
            queries=queries,
            judge_config=judge_config,
            enabled_llm_judge_metrics=enabled_llm_judge_metrics,
        )

        # Merge judge scores into all_scores
        for idx, judge_score in enumerate(judge_scores):
            all_scores[idx].update(judge_score)

    return all_scores
