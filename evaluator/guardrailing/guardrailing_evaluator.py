import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from evaluator.guardrailing import guardrailing_metrics as metrics
from evaluator.guardrailing import utils

log = logging.getLogger(__name__)


class GuardrailingEvaluator:
    """Evaluator for guardrailing benchmark."""

    def __init__(
        self,
        benchmark_config,
        model_config,
        shared_output_dir: Path,
        save_predictions: bool,
        save_format: str,
        include_scores: bool,
        include_raw_outputs: bool,
        inference_engine,
    ):
        self.benchmark_config = benchmark_config
        self.model_config = model_config
        self.inference_engine = inference_engine

        self.save_predictions = save_predictions
        self.save_format = save_format
        self.include_scores = include_scores
        self.include_raw_outputs = include_raw_outputs

        self.output_dir = shared_output_dir
        self.system_prompt: str = ""

    def _load_resources(self) -> None:
        """Load system prompt/instructions."""
        self.system_prompt = utils.load_system_prompt(
            self.benchmark_config.prompt.instructions_path
        )

    async def run_inference_only(self) -> Dict[str, Any]:
        """Run inference phase only, return data for later metric computation."""
        self._load_resources()

        log.info("Loading dataset")
        dataset_df = utils.load_data(
            data_path=self.benchmark_config.data.path,
            fields=[
                self.benchmark_config.data.field_prompt,
                self.benchmark_config.data.field_label,
                self.benchmark_config.data.field_violated_categories,
            ],
            max_samples=self.benchmark_config.data.max_samples,
        )
        log.info(f"Loaded {len(dataset_df)} samples")

        dataset_list = dataset_df.to_dict("records")
        prompts = [row[self.benchmark_config.data.field_prompt] for row in dataset_list]

        # Use guardrailing inference wrapper
        from evaluator.guardrailing.guardrailing_inference import GuardrailingInference

        log.info(
            f"Running guardrailing inference (backend: {self.model_config.backend})"
        )
        guardrailing_inference = GuardrailingInference(self.inference_engine)

        outputs = await guardrailing_inference.run_guardrailing_inference(
            prompts=prompts,
            system_prompt=self.system_prompt,
            valid_labels=self.benchmark_config.prompt.valid_labels,
            batch_size=self.model_config.batch_size,
        )

        # outputs is a flat List[Dict] — extract text for downstream use
        raw_text_outputs = []
        results = []
        for idx, (row, output) in enumerate(zip(dataset_list, outputs)):
            # output is {"role": "assistant", "content": "..."} or {"role": "assistant", "error": "..."}
            output_text = output.get("content", output.get("error", "")).strip()
            raw_text_outputs.append(output_text)
            results.append(
                {
                    **row,
                    "output": output_text,
                    "sample_id": idx,
                }
            )

        log.info("Parsing outputs")
        valid_results = self._parse_outputs(results)

        return {
            "dataset_df": dataset_df,
            "prompts": prompts,
            "outputs": raw_text_outputs,
            "valid_results": valid_results,
        }

    def _parse_outputs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits into valid and discarded outputs based on valid_labels.
        """
        valid_labels = self.benchmark_config.prompt.valid_labels

        valid_results, discarded_results = utils.split_discarded(results, valid_labels)

        if discarded_results:
            benchmark_dir = self.output_dir / "guardrailing"
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            discarded_path = benchmark_dir / "discarded.json"
            utils.dump_to_json(discarded_results, str(discarded_path))
            log.warning(
                f"{len(discarded_results)} samples discarded to {discarded_path}"
            )

        return valid_results

    def compute_metrics_and_save(
        self, inference_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute metrics and save all results."""
        dataset_df = inference_results["dataset_df"]
        prompts = inference_results["prompts"]
        outputs = inference_results["outputs"]
        valid_results = inference_results["valid_results"]

        log.info("Computing metrics")
        scores = self._score_predictions(valid_results)

        # Filter dataset to valid results only
        valid_indices = [r["sample_id"] for r in valid_results]
        valid_dataset_df = dataset_df.iloc[valid_indices].reset_index(drop=True)
        valid_prompts = [prompts[i] for i in valid_indices]
        valid_outputs = [outputs[i] for i in valid_indices]

        self._save_results(
            dataset_df=valid_dataset_df,
            prompts=valid_prompts,
            outputs=valid_outputs,
            valid_results=valid_results,
            scores=scores,
        )

        benchmark_dir = self.output_dir / "guardrailing"
        results_file = benchmark_dir / "per_item_results.parquet"

        log.info("\nBenchmark: guardrailing")
        log.info(f"Model: {self.model_config.name}")
        log.info(f"Backend: {self.model_config.backend}")
        log.info(f"Output directory: {self.output_dir}")
        log.info(f"Per-item results saved to: {results_file}")
        log.info(f"Total samples evaluated: {len(valid_results)}/{len(dataset_df)}")

        return {
            "benchmark": "guardrailing",
            "model": self.model_config.name,
            "backend": self.model_config.backend,
            "output_dir": str(self.output_dir),
            "results_file": str(results_file),
            "num_samples": len(valid_results),
            "num_discarded": len(dataset_df) - len(valid_results),
            "scores": scores,
        }

    def _score_predictions(
        self, valid_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Score predictions against ground truth."""
        data_cfg = self.benchmark_config.data

        ground_truth_labels = [row[data_cfg.field_label] for row in valid_results]
        predicted_labels = [row["output"] for row in valid_results]

        scores = metrics.compute_all_metrics(
            ground_truth_labels=ground_truth_labels,
            predicted_labels=predicted_labels,
            negative_label=self.benchmark_config.scoring.negative_label,
        )

        detailed_report = metrics.get_detailed_classification_report(
            ground_truth_labels, predicted_labels
        )

        log.info("\nClassification Report:")
        log.info(json.dumps(detailed_report, indent=2))
        log.info("\nOverall Metrics:")
        log.info(f"  Accuracy:  {scores['accuracy']:.4f}")
        log.info(f"  Precision: {scores['precision']:.4f}")
        log.info(f"  Recall:    {scores['recall']:.4f}")
        log.info(f"  F1:        {scores['f1']:.4f}")

        return scores

    def _save_results(
        self,
        dataset_df: Any,
        prompts: List[str],
        outputs: List[str],
        valid_results: List[Dict[str, Any]],
        scores: Dict[str, float],
    ) -> None:
        """Save evaluation results."""
        benchmark_dir = self.output_dir / "guardrailing"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if self.save_predictions:
            utils.save_predictions(
                output_dir=benchmark_dir,
                dataset=dataset_df,
                prompts=prompts,
                outputs=outputs,
                predictions=valid_results,
                scores=scores,
                save_format=self.save_format,
            )

        utils.save_per_item_results(
            output_dir=benchmark_dir,
            dataset=dataset_df,
            outputs=outputs,
            predictions=valid_results,
            field_prompt=self.benchmark_config.data.field_prompt,
            field_label=self.benchmark_config.data.field_label,
            field_violated_categories=self.benchmark_config.data.field_violated_categories,
            include_raw_outputs=self.include_raw_outputs,
        )
