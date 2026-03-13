import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from params import EvalConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on saved inference outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inference-dir",
        type=str,
        required=True,
        help="Path to the directory produced by infer.py",
    )
    parser.add_argument(
        "--post-processed",
        type=lambda x: x.lower() == "true",
        default=False,
        help=("Override post_processed from saved config —(true/false)"),
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=["tool_calling", "conversational_content", "guardrailing"],
        help="Override which benchmarks to evaluate (defaults to those saved during inference)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config yaml (defaults to config.yaml in inference dir)",
    )

    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    # Load config saved by infer.py
    config_path = Path(args.config) if args.config else inference_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {config_path}")

    log.info(f"Loading config from {config_path}")
    config = EvalConfig.from_yaml(str(config_path))

    # Load inference outputs
    inference_output_path = inference_dir / "inference_outputs.json"
    if not inference_output_path.exists():
        raise FileNotFoundError(f"inference_outputs.json not found in {inference_dir}")

    log.info(f"Loading inference outputs from {inference_output_path}")
    with open(inference_output_path, "r") as f:
        saved = json.load(f)

    records = saved["records"]
    outputs = saved["outputs"]
    saved_config = saved["config"]
    benchmarks_from_inference = saved_config.get("benchmarks", [])

    # Restore saved inference config back onto model config so evaluators
    # see exactly what was used during inference
    config.model.parser_type = saved_config.get("parser_type", config.model.parser_type)
    config.model.template_has_tool_token = saved_config.get(
        "template_has_tool_token", config.model.template_has_tool_token
    )
    config.model.post_processed = saved_config.get(
        "post_processed", config.model.post_processed
    )

    # CLI --post-processed
    if args.post_processed is not None:
        config.model.post_processed = args.post_processed

    benchmarks_to_run = (
        args.benchmarks if args.benchmarks else benchmarks_from_inference
    )
    log.info(f"Evaluating benchmarks: {', '.join(benchmarks_to_run)}")
    log.info(f"Post-processed: {config.model.post_processed}")

    inference_results = {
        "records": records,
        "outputs": outputs,
    }

    final_results = {}

    log.info("PHASE 3: EVALUATION")

    # --- Guardrailing ---
    if "guardrailing" in benchmarks_to_run:
        guardrail_path = inference_dir / "guardrailing_inference.json"
        if not guardrail_path.exists():
            log.warning(
                f"guardrailing_inference.json not found in {inference_dir} — skipping guardrailing"
            )
        else:
            log.info("\n--- Guardrailing Evaluation ---")
            try:
                from evaluator.guardrailing.guardrailing_evaluator import (
                    GuardrailingEvaluator,
                )

                with open(guardrail_path, "r") as f:
                    gr_saved = json.load(f)

                gr_inference_results = {
                    "prompts": gr_saved["prompts"],
                    "outputs": gr_saved["outputs"],
                    "valid_results": gr_saved["valid_results"],
                    "dataset_df": pd.DataFrame(gr_saved["dataset_records"]),
                }

                guardrailing_evaluator = GuardrailingEvaluator(
                    benchmark_config=config.guardrailing,
                    model_config=config.model,
                    shared_output_dir=inference_dir,
                    save_predictions=config.save_predictions,
                    save_format=config.save_format,
                    include_scores=config.include_scores,
                    include_raw_outputs=config.include_raw_outputs,
                    inference_engine=None,
                )
                guardrail_results = guardrailing_evaluator.compute_metrics_and_save(
                    gr_inference_results
                )
                final_results["guardrailing"] = guardrail_results
                log.info("Guardrailing evaluation complete")

            except Exception as e:
                log.error(f"Failed to evaluate guardrailing: {e}")
                import traceback

                traceback.print_exc()

    # --- Tool Calling ---
    if "tool_calling" in benchmarks_to_run:
        log.info("\n--- Tool Calling Evaluation ---")
        try:
            from evaluator.tool_calling.tool_calling_evaluator import (
                ToolCallingEvaluator,
            )

            tool_evaluator = ToolCallingEvaluator(
                benchmark_config=config.tool_calling,
                model_config=config.model,
                shared_output_dir=inference_dir,
                save_predictions=config.save_predictions,
                save_format=config.save_format,
                include_scores=config.include_scores,
                include_raw_outputs=config.include_raw_outputs,
                post_processed=config.model.post_processed,
            )
            tool_results = tool_evaluator.evaluate_and_save(inference_results)
            final_results["tool_calling"] = tool_results
            log.info("Tool calling evaluation complete")

        except Exception as e:
            log.error(f"Failed to evaluate tool calling: {e}")
            import traceback

            traceback.print_exc()

    # --- Conversational Content ---
    if "conversational_content" in benchmarks_to_run:
        log.info("\n--- Content Evaluation ---")
        try:
            from evaluator.conversational_content.content_evaluator import (
                ContentEvaluator,
            )

            content_evaluator = ContentEvaluator(
                model_config=config.model,
                shared_output_dir=inference_dir,
                save_predictions=config.save_predictions,
                save_format=config.save_format,
                include_scores=config.include_scores,
                include_raw_outputs=config.include_raw_outputs,
                conversational_config=config.conversational,
                judge_config=config.judge,
                post_processed=config.model.post_processed,
            )
            content_results = await content_evaluator.evaluate_and_save(
                inference_results
            )
            final_results["content"] = content_results
            log.info("Content evaluation complete")

        except Exception as e:
            log.error(f"Failed to evaluate content: {e}")
            import traceback

            traceback.print_exc()

    log.info("\nEVALUATION SUMMARY")
    for benchmark, results in final_results.items():
        log.info(f"\n{benchmark.upper()}:")
        log.info(f"  Output: {results.get('output_dir', 'N/A')}")
        log.info(f"  Samples: {results.get('num_samples', 'N/A')}")
        if "results_file" in results:
            log.info(f"  Results: {results['results_file']}")

    log.info("\nEvaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
