import asyncio
import logging
import os
from typing import Any, Dict, List, Tuple, cast

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
        try:
            return 1.0 if float(output) > 0.5 else 0.0
        except ValueError:
            log.warning(
                f"Failed to parse binary judge output: {output!r}, returning 0.0"
            )
            return 0.0

    if output_type == "score_1_5":
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

        outputs = await tqdm_asyncio.gather(
            *[call_single(s, u) for s, u in prompts],
            desc="LLM Judge (Content Metrics)",
            unit="sample",
            colour="magenta",
        )
        return list(outputs)
    finally:
        await client.close()


async def compute_llm_judge_metrics_batch(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    queries: List[str],
    judge_config: Any,
    enabled_llm_judge_metrics: List[str],
) -> List[Dict[str, float]]:
    """Compute LLM-as-judge metrics via OpenAI for all enabled metrics.

    Note: Expects pre-filtered NLP-only predictions, ground_truths, and queries.
    """
    all_scores: Dict[str, List[float]] = {key: [] for key in enabled_llm_judge_metrics}

    if not judge_config or not judge_config.enabled:
        return all_scores

    async def run_single_metric(metric_name: str):
        if metric_name not in judge_config.prompts:
            log.warning(f"Judge prompt '{metric_name}' not found in config, skipping")
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
            for gt, pred, q in zip(ground_truths, predictions, queries)
        ]

        raw_outputs = await _run_judge_openai_batch(judge_prompts, judge_config)

        for i, raw_output in enumerate(raw_outputs):
            score = _parse_judge_output(raw_output, prompt_config.output_type)
            all_scores[prompt_config.metric_name].append(score)

    await asyncio.gather(*[run_single_metric(m) for m in enabled_llm_judge_metrics])

    return all_scores
