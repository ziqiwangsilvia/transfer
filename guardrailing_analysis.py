import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class GuardrailingAnalyser:
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
                results_file = model_dir / "guardrailing" / "per_item_results.parquet"
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

        return model_results

    def compute_classification_metrics(self, df: pd.DataFrame) -> Dict:
        y_true = df["ground_truth_label"].values
        y_pred = df["predicted_label"].values
        pos_label = "unsafe"

        cm = confusion_matrix(y_true, y_pred, labels=["safe", pos_label])
        tn, fp, fn, tp = cm.ravel()

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
            "total_samples": len(df),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    def compute_summary(self, df: pd.DataFrame) -> Dict:
        return {
            "num_samples": len(df),
            "metrics": self.compute_classification_metrics(df),
            "ground_truth_distribution": df["ground_truth_label"]
            .value_counts()
            .to_dict(),
            "predicted_distribution": df["predicted_label"].value_counts().to_dict(),
        }

    def build_comparison_table(
        self, model_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        rows = []
        for model_name, df in model_dfs.items():
            m = self.compute_classification_metrics(df)
            rows.append(
                {
                    "model": model_name,
                    "num_samples": m["total_samples"],
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "true_positives": m["true_positives"],
                    "false_positives": m["false_positives"],
                    "true_negatives": m["true_negatives"],
                    "false_negatives": m["false_negatives"],
                }
            )
        return pd.DataFrame(rows).set_index("model")

    def create_radar_chart(
        self, model_dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ):
        metrics = ["accuracy", "precision", "recall", "f1"]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        for model_name, df in model_dfs.items():
            m = self.compute_classification_metrics(df)
            scores = [m[k] for k in metrics]
            scores_plot = scores + scores[:1]
            ax.plot(angles, scores_plot, "o-", linewidth=2, label=model_name)
            ax.fill(angles, scores_plot, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score"], fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(
            "Guardrailing - Classification Metrics", size=16, pad=20, fontweight="bold"
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Radar chart saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def create_bar_chart(
        self, model_dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ):
        comparison = self.build_comparison_table(model_dfs)
        metrics = ["accuracy", "precision", "recall", "f1"]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comparison.index))
        width = 0.2

        for i, metric in enumerate(metrics):
            offset = (i - len(metrics) / 2) * width + width / 2
            ax.bar(x + offset, comparison[metric], width, label=metric.capitalize())

        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Guardrailing - Model Comparison", fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison.index, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Bar chart saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def create_confusion_matrix_plot(
        self, model_dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ):
        n_models = len(model_dfs)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = [axes] if n_models == 1 else axes.flatten()

        for idx, (model_name, df) in enumerate(model_dfs.items()):
            m = self.compute_classification_metrics(df)
            cm = np.array(
                [
                    [m["true_negatives"], m["false_positives"]],
                    [m["false_negatives"], m["true_positives"]],
                ]
            )

            ax = axes[idx]
            im = ax.imshow(cm, cmap="Blues", aspect="auto")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=16,
                        fontweight="bold",
                    )

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Safe", "Unsafe"], fontsize=11)
            ax.set_yticklabels(["Safe", "Unsafe"], fontsize=11)
            ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
            ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
            ax.set_title(
                f"{model_name}\nAcc: {m['accuracy']:.3f}, F1: {m['f1']:.3f}",
                fontsize=12,
                fontweight="bold",
            )
            plt.colorbar(im, ax=ax)

        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix plot saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def create_all_charts(
        self, model_dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ):
        base = str(Path(output_path).with_suffix("")) if output_path else None
        self.create_radar_chart(model_dfs, f"{base}_radar.png" if base else None)
        self.create_bar_chart(model_dfs, f"{base}_bar.png" if base else None)
        self.create_confusion_matrix_plot(
            model_dfs, f"{base}_confusion.png" if base else None
        )

    def analyse_single_model(self, results_path: str, output_dir: Optional[str] = None):
        print(f"\nAnalyzing: {results_path}")
        df = self.load_results(results_path)
        summary = self.compute_summary(df)
        m = summary["metrics"]

        print(f"\nTotal samples: {summary['num_samples']}")
        print(f"\n{'Metric':<20} {'Value':>10}")
        print("-" * 32)
        print(f"{'Accuracy':<20} {m['accuracy']:>10.4f}")
        print(f"{'Precision':<20} {m['precision']:>10.4f}")
        print(f"{'Recall':<20} {m['recall']:>10.4f}")
        print(f"{'F1 Score':<20} {m['f1']:>10.4f}")

        print("\nConfusion Matrix:")
        print(f"{'':>15} {'Predicted Safe':>15} {'Predicted Unsafe':>17}")
        print(
            f"{'Actual Safe':<15} {m['true_negatives']:>15} {m['false_positives']:>17}"
        )
        print(
            f"{'Actual Unsafe':<15} {m['false_negatives']:>15} {m['true_positives']:>17}"
        )

        print(f"\nGround Truth: {summary['ground_truth_distribution']}")
        print(f"Predictions:  {summary['predicted_distribution']}")

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
        print(f"\nLoading models from: {eval_dir}")
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
            self.create_all_charts(model_dfs, output_file)
        else:
            self.create_all_charts(model_dfs)
