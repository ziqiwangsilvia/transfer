import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluator.tool_calling.tool_calling_metrics import compute_all_metrics
from evaluator.tool_calling.utils import (
    format_schemas_for_vllm,
    load_schemas_from_json,
    load_tools,
    save_per_item_results,
    save_predictions,
)
from processing.post_processing.parser import parse_outputs

log = logging.getLogger(__name__)


class ToolCallingEvaluator:
    """Evaluator for tool-calling metrics."""

    def __init__(
        self,
        benchmark_config,
        model_config,
        shared_output_dir: Path,
        save_predictions: bool,
        save_format: str,
        include_scores: bool,
        include_raw_outputs: bool,
        post_processed: bool = False,
    ):
        self.benchmark_config = benchmark_config
        self.model_config = model_config
        self.post_processed = post_processed

        self.save_predictions = save_predictions
        self.save_format = save_format
        self.include_scores = include_scores
        self.include_raw_outputs = include_raw_outputs

        self.output_dir = shared_output_dir

    def load_tools(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load and format tools for scoring. Returns (tools, tool_schemas)."""
        raw_tools = load_tools(self.benchmark_config.tool)
        tool_schemas = load_schemas_from_json(raw_tools)
        tools = format_schemas_for_vllm(tool_schemas)
        log.info(f"Loaded {len(tools)} tools")
        return tools, tool_schemas

    def evaluate_and_save(self, inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate tool-calling responses and save results."""
        records = inference_results["records"]
        outputs = inference_results["outputs"]

        tools, tool_schemas = self.load_tools()

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
                else {"type": "nlp", "response": output.get("content", "")}
                for output in outputs
            ]
            outputs = [None] * len(predictions)

        # Split records into tool-calling and NLP buckets.
        # Tool-calling samples get full metrics; NLP samples get only
        # when2call and schema_reliability (all other metrics set to None).

        # GT classification:
        #   "tool_calls" in gt -> tool bucket
        #   "content"   in gt -> NLP bucket

        tool_records, tool_outputs, tool_predictions = [], [], []
        nlp_records, nlp_outputs, nlp_predictions = [], [], []

        for record, output, prediction in zip(records, outputs, predictions):
            gt = record["ground_truth"]

            if "tool_calls" in gt:
                tool_records.append(record)
                tool_outputs.append(output)
                tool_predictions.append(prediction)
            else:
                nlp_records.append(record)
                nlp_outputs.append(output)
                nlp_predictions.append(prediction)

        log.info(
            f"{len(tool_records)} tool-calling samples, "
            f"{len(nlp_records)} NLP samples "
            f"(of {len(records)} total)"
        )

        if not tool_records:
            log.warning("No tool-calling samples found — skipping evaluation")
            return {
                "benchmark": "tool_calling",
                "model": self.model_config.name,
                "backend": self.model_config.backend,
                "output_dir": str(self.output_dir),
                "num_samples": 0,
            }

        tool_scores = self._score_predictions(
            tool_records, tool_predictions, tool_outputs, tools, tool_schemas
        )

        # Score NLP samples for when2call and schema_reliability only
        nlp_scores = self._score_nlp_behavioural(
            nlp_records, nlp_predictions, nlp_outputs
        )

        # Combine: tool rows first, then NLP rows
        all_records = tool_records + nlp_records
        all_outputs = tool_outputs + nlp_outputs
        all_predictions = tool_predictions + nlp_predictions
        all_scores = tool_scores + nlp_scores

        self._save_results(
            records=all_records,
            outputs=all_outputs,
            predictions=all_predictions,
            scores=all_scores,
        )

        benchmark_dir = self.output_dir / "tool_calling"
        results_file = benchmark_dir / "per_item_results.parquet"

        log.info(
            f"Tool calling evaluation complete — {len(tool_records)} tool samples, {len(nlp_records)} NLP samples"
        )
        log.info(f"Results: {results_file}")

        return {
            "benchmark": "tool_calling",
            "model": self.model_config.name,
            "backend": self.model_config.backend,
            "output_dir": str(self.output_dir),
            "results_file": str(results_file),
            "num_samples": len(tool_records),
            "num_nlp_samples": len(nlp_records),
        }

    def _score_nlp_behavioural(
        self,
        records: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """Compute when2call and schema_reliability for NLP-GT records."""
        from evaluator.tool_calling.tool_calling_metrics import (
            get_schema_reliability,
            get_when2call,
        )

        # Metrics present in tool rows — all set to None for NLP rows
        null_tool_metrics = {
            "tool_pick_up_rate": None,
            "tool_hallucination_rate": None,
            "tool_additional_rate": None,
            "variable_pickup_rate": None,
            "variable_correct_rate": None,
            "variable_hallucination_rate": None,
            "variable_additional_rate": None,
            "exact_match": None,
        }

        scores = []
        for record, prediction, output in zip(records, predictions, outputs):
            gt = record["ground_truth"]
            gt_for_metrics = {"type": "nlp", "response": gt.get("content", "")}
            raw_text = output.get("content", "") if output is not None else None

            score = {
                **null_tool_metrics,
                "when2call": get_when2call(gt_for_metrics, prediction),
                "schema_reliability_raw": get_schema_reliability(
                    gt_for_metrics, prediction, raw_text
                )[0],
                "schema_reliability_processed": get_schema_reliability(
                    gt_for_metrics, prediction, raw_text
                )[1],
            }
            scores.append(score)

        return scores

    def _score_predictions(
        self,
        records: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_schemas: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        """Score tool-calling predictions against ground truth."""
        scores = []
        available_tool_names = [t["function"]["name"] for t in tools]

        for record, prediction, output in zip(records, predictions, outputs):
            gt = record["ground_truth"]

            # Ground truth message dict -> eval dict format expected by metrics
            # {"role": "assistant", "tool_calls": {...}} → {"type": "tool", "tools": [...]}
            # {"role": "assistant", "content": "..."}   → {"type": "nlp", ...} (ambiguous)
            if "tool_calls" in gt:
                gt_for_metrics = {
                    "type": "tool",
                    "tools": gt["tool_calls"]
                    if isinstance(gt["tool_calls"], list)
                    else [gt["tool_calls"]],
                }
            else:
                gt_for_metrics = {"type": "nlp", "response": gt.get("content", "")}

            score = compute_all_metrics(
                ground_truth=gt_for_metrics,
                prediction=prediction,
                raw_output=output,
                available_tools=available_tool_names,
                tool_schemas=tool_schemas,
            )
            scores.append(score)

        return scores

    def _save_results(
        self,
        records: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        scores: List[Dict[str, float]],
    ) -> None:
        """Save evaluation results."""
        benchmark_dir = self.output_dir / "tool_calling"
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
