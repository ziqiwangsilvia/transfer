import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from inference.inference import InferenceEngine
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
        description="Run pre-processing and inference, saving outputs for evaluate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config/evaluation_default.yaml")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=["tool_calling", "conversational_content", "guardrailing"],
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--backend", type=str, choices=["openai", "vllm-service"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-base", type=str)
    parser.add_argument("--prompt-sections", type=str, nargs="+")
    parser.add_argument("--schema-path", type=str)
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override the directory name (defaults to model name)",
    )

    args = parser.parse_args()

    log.info(f"Loading configuration from {args.config}")
    config = EvalConfig.from_yaml(args.config)

    # CLI overrides
    if args.model:
        config.model.name = args.model
    if args.backend:
        config.model.backend = args.backend
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.temperature is not None:
        config.model.temperature = args.temperature
    if args.max_tokens:
        config.model.max_tokens = args.max_tokens
    if args.api_base:
        config.model.api_base = args.api_base
    if args.max_samples:
        config.tool_calling.data.max_samples = args.max_samples
        config.guardrailing.data.max_samples = args.max_samples
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.prompt_sections:
        config.tool_calling.prompt.sections = args.prompt_sections
    if args.schema_path:
        config.tool_calling.tool.schema_path = args.schema_path
    if args.benchmarks:
        config.tool_calling.enabled = "tool_calling" in args.benchmarks
        config.guardrailing.enabled = "guardrailing" in args.benchmarks
        config.conversational.enabled = "conversational_content" in args.benchmarks

    benchmarks_to_run = []
    if config.tool_calling.enabled:
        benchmarks_to_run.append("tool_calling")
    if config.guardrailing.enabled:
        benchmarks_to_run.append("guardrailing")
    if config.conversational.enabled:
        benchmarks_to_run.append("conversational_content")

    if not benchmarks_to_run:
        log.warning("No benchmarks enabled. Exiting.")
        return

    log.info(f"Benchmarks: {', '.join(benchmarks_to_run)}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = args.run_name if args.run_name else config.model.name.replace("/", "_")
    output_dir = Path(config.output_dir) / timestamp / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    log.info("PHASE 1: PRE-PROCESSING")

    import prompts.prompts as prompt_module
    from processing.pre_processing.pre_processing import (
        convert_filter_format,
        load_dataset,
        prepare_records,
    )

    if not config.tool_calling.data.path:
        raise ValueError("tool_calling.data.path is required")
    if not config.tool_calling.tool.schema_path:
        raise ValueError("tool_calling.tool.schema_path is required")
    if not config.tool_calling.prompt.sections:
        raise ValueError("tool_calling.prompt.sections is required")

    dataset = load_dataset(
        path=config.tool_calling.data.path,
        max_samples=config.tool_calling.data.max_samples,
    )

    if "_simplified" in config.tool_calling.tool.schema_path:
        log.info("Converting ground truth to simplified schema format")
        dataset = convert_filter_format(dataset)

    system_message = "\n\n".join(
        getattr(prompt_module, section)
        for section in config.tool_calling.prompt.sections
    )

    records = prepare_records(
        dataset=dataset,
        template_has_tool_token=config.model.template_has_tool_token,
        model_in_the_loop=config.model.model_in_the_loop,
    )
    log.info(f"Prepared {len(records)} records")

    tools = None
    if config.model.template_has_tool_token:
        from evaluator.tool_calling.utils import (
            format_schemas_for_vllm,
            load_schemas_from_json,
            load_tools,
        )

        raw_tools = load_tools(config.tool_calling.tool)
        tool_schemas = load_schemas_from_json(raw_tools)
        tools = format_schemas_for_vllm(tool_schemas)

    system_turn = {"role": "system", "content": system_message}
    contexts = [[system_turn] + record["context"] for record in records]

    log.info("PHASE 2: INFERENCE")

    inference_engine = InferenceEngine(config.model)
    await inference_engine.initialize()

    # --- Guardrailing inference (if enabled) ---
    if "guardrailing" in benchmarks_to_run:
        log.info("Running guardrailing inference")
        try:
            from evaluator.guardrailing.guardrailing_evaluator import (
                GuardrailingEvaluator,
            )

            guardrailing_evaluator = GuardrailingEvaluator(
                benchmark_config=config.guardrailing,
                model_config=config.model,
                shared_output_dir=output_dir,
                save_predictions=config.save_predictions,
                save_format=config.save_format,
                include_scores=config.include_scores,
                include_raw_outputs=config.include_raw_outputs,
                inference_engine=inference_engine,
            )
            guardrail_inference_results = (
                await guardrailing_evaluator.run_inference_only()
            )

            # Save for evaluate.py — dataset_df is not JSON serialisable
            guardrail_output_path = output_dir / "guardrailing_inference.json"
            with open(guardrail_output_path, "w") as f:
                json.dump(
                    {
                        "prompts": guardrail_inference_results["prompts"],
                        "outputs": guardrail_inference_results["outputs"],
                        "valid_results": guardrail_inference_results["valid_results"],
                        "dataset_records": guardrail_inference_results[
                            "dataset_df"
                        ].to_dict("records"),
                    },
                    f,
                    indent=2,
                )
            log.info(f"Guardrailing inference saved to {guardrail_output_path}")

        except Exception as e:
            log.error(f"Guardrailing inference failed: {e}")
            import traceback

            traceback.print_exc()

    # Tool calling / conversational inference
    log.info("Running tool calling inference")
    outputs = await inference_engine.run_inference(
        contexts=contexts,
        batch_size=config.model.batch_size,
        stop_sequences=config.tool_calling.prompt.stop_sequences,
        tools=tools,
    )
    log.info("Inference complete")

    inference_engine.cleanup()

    # SAVE OUTPUTS

    inference_output_path = output_dir / "inference_outputs.json"
    with open(inference_output_path, "w") as f:
        json.dump(
            {
                "records": records,
                "outputs": outputs,
                "config": {
                    "model_name": config.model.name,
                    "backend": config.model.backend,
                    "schema_path": config.tool_calling.tool.schema_path,
                    "parser_type": config.model.parser_type,
                    "template_has_tool_token": config.model.template_has_tool_token,
                    "benchmarks": benchmarks_to_run,
                },
            },
            f,
            indent=2,
        )
    log.info(f"Inference outputs saved to {inference_output_path}")

    # Print output dir as last line — picked up by run_multi_eval.sh
    print(str(output_dir))


if __name__ == "__main__":
    asyncio.run(main())
