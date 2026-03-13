import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POSITIVE_METRICS = [
    "when2call",
    "schema_reliability_raw",
    "schema_reliability_processed",
    "tool_pick_up_rate",
    "variable_pickup_rate",
    "variable_correct_rate",
    "exact_match",
]

NEGATIVE_METRICS = [
    "tool_hallucination_rate",
    "tool_additional_rate",
    "variable_hallucination_rate",
    "variable_additional_rate",
]

METRIC_LABELS = {
    "when2call": "When to Call",
    "schema_reliability_raw": "Schema\nReliability Raw",
    "schema_reliability_processed": "Schema\nReliability Processed",
    "tool_pick_up_rate": "Tool\nPick-up",
    "variable_pickup_rate": "Var\nPick-up",
    "variable_correct_rate": "Var\nCorrect",
    "exact_match": "Exact\nMatch",
    "tool_hallucination_rate": "Tool\nHallucination",
    "tool_additional_rate": "Tool\nAdditional",
    "variable_hallucination_rate": "Var\nHallucination",
    "variable_additional_rate": "Var\nAdditional",
}


class ToolCallingAnalyser:
    def load_results(self, results_path: str) -> pd.DataFrame:
        path = Path(results_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    def load_multiple_models(
        self, output_dir: str, include_timestamp: bool = False
    ) -> Dict[str, pd.DataFrame]:
        model_results = {}

        for timestamp_dir in Path(output_dir).glob("*"):
            if not timestamp_dir.is_dir():
                continue
            for model_dir in timestamp_dir.glob("*"):
                if not model_dir.is_dir():
                    continue
                results_file = model_dir / "tool_calling" / "per_item_results.parquet"
                if not results_file.exists():
                    continue

                df = pd.read_parquet(results_file)
                key = (
                    f"{timestamp_dir.name}_{model_dir.name}"
                    if include_timestamp
                    else model_dir.name
                )
                model_results[key] = df
                print(f"Loaded {len(df)} samples for {key}")

        # Sort by timestamp (earliest first)
        return dict(
            sorted(model_results.items(), key=lambda item: item[1].iloc[0].name)
        )

    def compute_summary(self, df: pd.DataFrame) -> Dict:
        summary: Dict = {"num_samples": len(df), "metrics": {}}
        for col in df.columns:
            if col.startswith("score_"):
                metric_name = col.replace("score_", "")
                summary["metrics"][metric_name] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                }
        return summary

    def build_comparison_table(
        self, model_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        if not model_dfs:
            return pd.DataFrame()

        rows = []
        for model_name, df in model_dfs.items():
            row = {"model": model_name, "num_samples": len(df)}
            for col in df.columns:
                if col.startswith("score_"):
                    row[f"{col.replace('score_', '')}_mean"] = df[col].mean()
                    row[f"{col.replace('score_', '')}_std"] = df[col].std()
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def _plot_radar_chart(
        self,
        ax,
        model_dfs: Dict[str, pd.DataFrame],
        metrics: list,
        title: str,
        y_max: float = 1.0,
    ):
        first_df = next(iter(model_dfs.values()))
        available = [m for m in metrics if f"score_{m}" in first_df.columns]
        if not available:
            return

        angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
        angles += angles[:1]

        for model_name, df in model_dfs.items():
            scores = [
                df[f"score_{m}"].dropna().mean() if f"score_{m}" in df.columns else 0.0
                for m in available
            ]
            scores_plot = scores + scores[:1]
            ax.plot(angles, scores_plot, "o-", linewidth=2, label=model_name)
            ax.fill(angles, scores_plot, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in available], fontsize=11)
        ax.set_ylim(0, y_max)
        ax.set_title(title, size=16, pad=20, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

    def create_tool_calling_charts(
        self, model_dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ):
        if not model_dfs:
            print("No models to compare")
            return

        # Compute y_max for negative metrics
        max_negative = max(
            (
                df[f"score_{m}"].dropna().mean()
                for df in model_dfs.values()
                for m in NEGATIVE_METRICS
                if f"score_{m}" in df.columns
            ),
            default=0.0,
        )

        if max_negative == 0:
            negative_y_max = 0.1
        elif max_negative < 0.1:
            negative_y_max = ((max_negative * 1.1 // 0.02) + 1) * 0.02
        else:
            negative_y_max = ((max_negative * 1.1 // 0.05) + 1) * 0.05

        for metrics, title, y_max, suffix in [
            (
                POSITIVE_METRICS,
                "Tool Calling - Positive Metrics (Higher is Better)",
                1.0,
                "_positive",
            ),
            (
                NEGATIVE_METRICS,
                "Tool Calling - Negative Metrics (Lower is Better)",
                negative_y_max,
                "_negative",
            ),
        ]:
            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )
            self._plot_radar_chart(ax, model_dfs, metrics, title, y_max)
            plt.tight_layout()

            if output_path:
                out = Path(output_path)
                save_path = out.with_name(out.stem + suffix + out.suffix)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Chart saved to {save_path}")
            else:
                plt.show()
            plt.close()

    def analyse_single_model(self, results_path: str, output_dir: Optional[str] = None):
        print(f"\nAnalyzing: {results_path}")
        df = self.load_results(results_path)
        summary = self.compute_summary(df)

        print(f"\nTotal samples: {summary['num_samples']}")

        for label, metrics in [
            ("Positive Metrics (Higher is Better)", POSITIVE_METRICS),
            ("Negative Metrics (Lower is Better)", NEGATIVE_METRICS),
        ]:
            print(f"\n=== {label} ===")
            print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print("-" * 70)
            for metric in metrics:
                if metric in summary["metrics"]:
                    s = summary["metrics"][metric]
                    print(
                        f"{metric:<30} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}"
                    )

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            df.to_csv(out / "analysis.csv", index=False)
            print(f"\nResults saved to {output_dir}")

    def compare_models(
        self,
        eval_dir: str,
        output_file: Optional[str] = None,
        include_timestamp: bool = False,
    ):
        print(f"\nLoading tool calling models from: {eval_dir}")
        model_dfs = self.load_multiple_models(eval_dir, include_timestamp)

        if not model_dfs:
            print("No models found!")
            return

        print(f"\nFound {len(model_dfs)} models")
        comparison = self.build_comparison_table(model_dfs)
        print("\nModel Comparison:")
        print(comparison.to_string())

        if output_file:
            comparison.to_csv(Path(output_file).with_suffix(".csv"))
            self.create_tool_calling_charts(
                model_dfs, str(Path(output_file).with_suffix(".png"))
            )
        else:
            self.create_tool_calling_charts(model_dfs)
