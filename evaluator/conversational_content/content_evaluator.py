import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluator.conversational_content.content_metrics_registry import (
    compute_all_metrics_batch,
)
from evaluator.tool_calling.tool_calling_metrics import (
    get_when2call,
)
from evaluator.tool_calling.utils import save_per_item_results, save_predictions
from processing.post_processing.parser import parse_outputs

log = logging.getLogger(__name__)


class ContentEvaluator:
    """Evaluator for conversational/NLP content responses."""

    def __init__(
        self,
        model_config,
        shared_output_dir: Path,
        save_predictions: bool,
        save_format: str,
        include_scores: bool,
        include_raw_outputs: bool,
        conversational_config,
        judge_config=None,
        post_processed: bool = False,
    ):
        self.model_config = model_config
        self.post_processed = post_processed
        self.conversational_config = conversational_config
        self.judge_config = judge_config

        self.save_predictions = save_predictions
        self.save_format = save_format
        self.include_scores = include_scores
        self.include_raw_outputs = include_raw_outputs

        self.output_dir = shared_output_dir

    async def evaluate_and_save(
        self, inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate NLP/content responses and save results"""
        records = inference_results["records"]
        outputs = inference_results["outputs"]

        log.info("Parsing outputs")
        if self.post_processed is False:
            parsed = parse_outputs(
                outputs,
                parser_type=self.model_config.parser_type,
                template_has_tool_token=self.model_config.template_has_tool_token,
            )
            predictions = [p.to_eval_dict() for p in parsed]
        else:
            import json

            predictions = [
                {
                    "type": "tool",
                    "tools": [
                        {
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"].get("arguments"), str)
                            else tc["function"].get("arguments", {}),
                        }
                        for tc in output["tool_calls"]
                    ],
                }
                if output.get("tool_calls")
                else {
                    "type": "nlp",
                    "response": output.get("content", "") if output is not None else "",
                }
                for output in outputs
            ]
            outputs = [None] * len(predictions)

        nlp_records, nlp_outputs, nlp_predictions = self._filter_nlp_samples(
            records, outputs, predictions
        )
        log.info(f"{len(nlp_records)} NLP samples (of {len(records)} total)")

        if not nlp_records:
            log.warning("No NLP samples found — skipping content evaluation")
            return {
                "benchmark": "content",
                "model": self.model_config.name,
                "backend": self.model_config.backend,
                "output_dir": str(self.output_dir),
                "num_samples": 0,
            }

        log.info("Computing content metrics")
        scores = await self._score_predictions(
            nlp_records, nlp_predictions, nlp_outputs
        )

        self._save_results(
            records=nlp_records,
            outputs=nlp_outputs,
            predictions=nlp_predictions,
            scores=scores,
        )

        benchmark_dir = self.output_dir / "content"
        results_file = benchmark_dir / "per_item_results.parquet"

        log.info(f"Content evaluation complete — {len(nlp_records)} samples")
        log.info(f"Results: {results_file}")

        return {
            "benchmark": "content",
            "model": self.model_config.name,
            "backend": self.model_config.backend,
            "output_dir": str(self.output_dir),
            "results_file": str(results_file),
            "num_samples": len(nlp_records),
        }

    def _filter_nlp_samples(
        self,
        records: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return only samples whose ground truth contains a ``content`` field.

        Keeps the ordering of the original lists and applies the same filter to
        ``records``, ``outputs``, and ``predictions``.  This isolates NLP
        examples from tool responses.
        """
        nlp_records: List[Dict[str, Any]] = []
        nlp_outputs: List[Dict[str, Any]] = []
        nlp_predictions: List[Dict[str, Any]] = []

        for rec, out, pred in zip(records, outputs, predictions):
            if "content" in rec.get("ground_truth", {}):
                nlp_records.append(rec)
                nlp_outputs.append(out)
                nlp_predictions.append(pred)

        return nlp_records, nlp_outputs, nlp_predictions

    async def _score_predictions(
        self,
        records: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score NLP predictions using content metrics and the judge model."""
        # Ground truth message dict -> eval dict format expected by metrics
        # {"role": "assistant", "content": "..."} -> {"type": "nlp", "response": "..."}
        ground_truths_for_metrics = []
        queries = []

        for record in records:
            gt = record["ground_truth"]
            ground_truths_for_metrics.append(
                {"type": "nlp", "response": gt.get("content", "")}
            )
            # Query is the last user turn in the context
            user_turns = [m for m in record["context"] if m["role"] == "user"]
            queries.append(user_turns[-1]["content"] if user_turns else "")

        if self.judge_config and self.judge_config.enabled:
            log.info("Scoring with judge model (batch)")
        else:
            log.info("Scoring without judge (LLM judge metrics skipped)")

        scores = await compute_all_metrics_batch(
            ground_truths=ground_truths_for_metrics,
            predictions=predictions,
            queries=queries,
            scoring_config=self.conversational_config.scoring,
            judge_config=self.judge_config,
        )

        # when2call and schema reliability
        for i, (gt_for_metrics, prediction, output) in enumerate(
            zip(ground_truths_for_metrics, predictions, outputs)
        ):
            scores[i]["when2call"] = get_when2call(gt_for_metrics, prediction)

        return scores

    def _save_results(
        self,
        records: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        scores: List[Dict[str, Any]],
    ) -> None:
        """Save evaluation results."""
        benchmark_dir = self.output_dir / "content"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if self.save_predictions:
            save_predictions(
                output_dir=benchmark_dir,
                records=records,
                outputs=outputs,
                predictions=predictions,
                scores=scores,
                save_format=self.save_format,
                include_scores=self.include_scores,
            )

        save_per_item_results(
            output_dir=benchmark_dir,
            records=records,
            outputs=outputs,
            predictions=predictions,
            scores=scores,
            include_raw_outputs=self.include_raw_outputs,
        )
