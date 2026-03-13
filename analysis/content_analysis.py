import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTENT_METRICS = [
    "levenshtein_ratio",
    "rouge1",
    "rougeL",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "content_similarity",
    "context_accuracy",
    "content_politeness",
    "content_helpfulness",
]

LOWER_IS_BETTER = ["levenshtein_distance"]

METRIC_LABELS = {
    "levenshtein_ratio": "Levenshtein\nRatio ↑",
    "levenshtein_distance": "Levenshtein\nDistance ↓",
    "rouge1": "ROUGE-1 ↑",
    "rougeL": "ROUGE-L ↑",
    "bert_precision": "BERT\nPrecision ↑",
    "bert_recall": "BERT\nRecall ↑",
    "bert_f1": "BERT F1 ↑",
    "content_similarity": "Content_Similarity ↑",
    "context_accuracy": "Context_Accuracy ↑",
    "content_politeness": "Politeness ↑",
    "content_helpfulness": "Helpfulness ↑",
}

ALL_METRIC_COLS = [f"score_{m}" for m in CONTENT_METRICS + LOWER_IS_BETTER]


class ContentAnalyser:
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
                results_file = model_dir / "content" / "per_item_results.parquet"
                if not results_file.exists():
                    continue

                df = pd.read_parquet(results_file)
                existing_cols = [col for col in ALL_METRIC_COLS if col in df.columns]
                if not existing_cols:
                    continue

                df = df[df[existing_cols].notna().any(axis=1)].copy()
                if df.empty:
                    continue

                key = (
                    f"{timestamp_dir.name}_{model_dir.name}"
                    if include_timestamp
                    else model_dir.name
                )
                model_results[key] = df
                print(f"Loaded {len(df)} NLP samples for {key}")

        return model_results

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

    def _normalise(
        self, value: float, metric_name: str, max_levenshtein: float
    ) -> float:
        if metric_name in ["content_similarity", "content_helpfulness"]:
            return (value - 1.0) / 4.0
        if metric_name == "levenshtein_distance":
            return value / max_levenshtein if max_levenshtein > 0 else 0.0
        return value

    def create_content_chart(
        self, model_dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ):
        if not model_dfs:
            print("No models to compare")
            return

        first_df = next(iter(model_dfs.values()))
        available_metrics = [
            m
            for m in CONTENT_METRICS + LOWER_IS_BETTER
            if f"score_{m}" in first_df.columns
        ]
        if not available_metrics:
            print("No content metrics found")
            return

        max_levenshtein = max(
            (
                df["score_levenshtein_distance"].mean()
                for df in model_dfs.values()
                if "score_levenshtein_distance" in df.columns
            ),
            default=0.0,
        )

        angles = np.linspace(
            0, 2 * np.pi, len(available_metrics), endpoint=False
        ).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

        for model_name, df in model_dfs.items():
            scores = [
                self._normalise(df[f"score_{m}"].mean(), m, max_levenshtein)
                for m in available_metrics
            ]
            scores_plot = scores + scores[:1]
            ax.plot(angles, scores_plot, "o-", linewidth=2, label=model_name)
            ax.fill(angles, scores_plot, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [METRIC_LABELS.get(m, m) for m in available_metrics], fontsize=12
        )
        ax.set_ylim(0, 1.0)
        ax.set_title(
            "Conversational Content Quality Metrics (Normalised 0-1)\n(↑ = Higher is Better, ↓ = Lower is Better)",
            size=16,
            pad=20,
            fontweight="bold",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Content quality chart saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def analyse_single_model(self, results_path: str, output_dir: Optional[str] = None):
        print(f"\nAnalyzing content metrics: {results_path}")
        df = self.load_results(results_path)

        existing_cols = [col for col in ALL_METRIC_COLS if col in df.columns]
        if not existing_cols:
            print("No content metrics found in this file")
            return

        df = df[df[existing_cols].notna().any(axis=1)]
        if df.empty:
            print("No content metrics found (no NLP responses)")
            return

        summary = self.compute_summary(df)
        print(f"\nTotal NLP samples: {summary['num_samples']}")
        print(f"\n{'Metric':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 75)

        for metric in CONTENT_METRICS + LOWER_IS_BETTER:
            if metric in summary["metrics"]:
                s = summary["metrics"][metric]
                direction = "↓" if metric in LOWER_IS_BETTER else "↑"
                label = f"{metric} {direction}"
                print(
                    f"{label:<40} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}"
                )

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "content_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            df.to_csv(out / "content_analysis.csv", index=False)
            print(f"\nResults saved to {output_dir}")

    def compare_models(
        self,
        eval_dir: str,
        output_file: Optional[str] = None,
        include_timestamp: bool = False,
    ):
        print(f"\nLoading content metrics from: {eval_dir}")
        model_dfs = self.load_multiple_models(eval_dir, include_timestamp)

        if not model_dfs:
            print("No models with content metrics found!")
            return

        print(f"\nFound {len(model_dfs)} models")

        rows = []
        for model_name, df in model_dfs.items():
            row = {"model": model_name, "num_samples": len(df)}
            for col in df.columns:
                if col.startswith("score_"):
                    row[f"{col.replace('score_', '')}_mean"] = df[col].mean()
                    row[f"{col.replace('score_', '')}_std"] = df[col].std()
            rows.append(row)

        comparison = pd.DataFrame(rows).set_index("model")
        print("\nModel Comparison:")
        print(comparison.to_string())

        if output_file:
            comparison.to_csv(Path(output_file).with_suffix(".csv"))
            self.create_content_chart(
                model_dfs, str(Path(output_file).with_suffix(".png"))
            )
        else:
            self.create_content_chart(model_dfs)
