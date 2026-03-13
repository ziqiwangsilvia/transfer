import argparse
import sys

from analysis.content_analysis import ContentAnalyser
from analysis.guardrailing_analysis import GuardrailingAnalyser
from analysis.tool_calling_analysis import ToolCallingAnalyser

ANALYSERS = {
    "tool_calling": ToolCallingAnalyser,
    "guardrailing": GuardrailingAnalyser,
    "content": ContentAnalyser,
}


def detect_benchmark(results_path: str) -> str:
    path_str = results_path.lower()
    for name in ANALYSERS:
        if name in path_str:
            return name
    raise ValueError(
        "Could not auto-detect benchmark type from path. "
        "Please specify with --benchmark (tool_calling, guardrailing, or content)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation analysis CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyse_results.py --results outputs/eval/2024-01-27/.../per_item_results.parquet --benchmark tool_calling
  python analyse_results.py --compare outputs/eval --benchmark tool_calling --output tc.png
  python analyse_results.py --compare outputs/eval --benchmark guardrailing --output gr.png
  python analyse_results.py --compare outputs/eval --benchmark content --output content.png
  python analyse_results.py --compare outputs/eval --benchmark tool_calling --output tc.png --include-timestamp
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results",
        type=str,
        help="Path to per_item_results.parquet for single model analysis",
    )
    group.add_argument(
        "--compare",
        type=str,
        help="Path to eval output directory for multi-model comparison",
    )

    parser.add_argument(
        "--output", type=str, help="Output file path for comparison results"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for single model analysis"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(ANALYSERS),
        help="Benchmark type to analyze",
    )
    parser.add_argument(
        "--include-timestamp",
        action="store_true",
        help="Include timestamp in model names",
    )

    args = parser.parse_args()

    try:
        if args.results:
            benchmark = args.benchmark or detect_benchmark(args.results)
            ANALYSERS[benchmark]().analyse_single_model(args.results, args.output_dir)

        elif args.compare:
            if not args.benchmark:
                print(
                    "Error: --benchmark required for multi-model comparison",
                    file=sys.stderr,
                )
                sys.exit(1)
            ANALYSERS[args.benchmark]().compare_models(
                args.compare, args.output, args.include_timestamp
            )

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
