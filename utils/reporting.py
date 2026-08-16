"""CSV tables, paired Wilcoxon tests, and report-style figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


FIGURE_NUMBERS = {
    "bci_errp": (6, 7, 8),
    "seizure_detection": (9, 10, 11),
    "attention_state": (12, 13, 14),
}
TABLE_NUMBERS = {
    "bci_errp": (5, 6),
    "seizure_detection": (7, 8),
    "attention_state": (9, 10),
}
METHOD_ORDER = ("raw", "bandpass", "ica", "asr", "asr20", "ic_unet")
CLASSIFIER_ORDER = (
    "logistic_regression", "svm", "random_forest", "lightgbm", "mlp", "eegnet",
    "vit", "mobilenet",
)
DISPLAY_METHOD = {
    "raw": "Raw", "bandpass": "Filter (1–50 Hz)", "asr": "ASR (k=5)",
    "asr20": "ASR (k=20)",
    "ic_unet": "IC-U-Net", "ica": "ICA"
}
DISPLAY_CLASSIFIER = {
    "logistic_regression": "LR", "svm": "SVM", "random_forest": "RF",
    "lightgbm": "LightGBM", "mlp": "MLP", "eegnet": "EEGNet",
    "vit": "ViT", "mobilenet": "MobileNet",
}
METHOD_COLORS = {
    "Raw": "#ff9999", "Filter (1–50 Hz)": "#8fc5f4",
    "ASR (k=5)": "#91e693", "ASR (k=20)": "#4daf4a",
    "IC-U-Net": "#ffd29b",
    "ICA": "#c7a6e8",
}
TASK_DISPLAY = {
    "bci_errp": "BCI", "seizure_detection": "Seizure", "attention_state": "Attention",
}


def write_outputs(results: pd.DataFrame, output_dir: Path, manifest: dict) -> None:
    if results.empty:
        raise ValueError("No experiment results were produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "all_runs.csv", index=False)
    summary = results.groupby(["task", "method", "classifier"], as_index=False).agg(
        primary_mean=("primary", "mean"),
        primary_std=("primary", lambda values: values.std(ddof=0)),
        auc_mean=("auc", "mean"),
        auc_std=("auc", lambda values: values.std(ddof=0)),
        accuracy_mean=("accuracy", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    two_sided = _wilcoxon_vs_raw(results, alternative="two-sided")
    two_sided.to_csv(output_dir / "wilcoxon_two_sided_vs_raw.csv", index=False)
    one_sided = _wilcoxon_vs_raw(results, alternative="less")
    one_sided.to_csv(output_dir / "wilcoxon_one_sided_vs_raw.csv", index=False)
    (output_dir / "mann_whitney_vs_raw.csv").unlink(missing_ok=True)
    for task_name in results["task"].unique():
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_runs = results[results["task"] == task_name]
        task_summary = summary[summary["task"] == task_name]
        _write_tables(task_summary, task_dir, task_name)
        _write_figures(task_runs, task_dir, task_name)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def checkpoint_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "all_runs.partial.csv", index=False)


def _wilcoxon_vs_raw(results: pd.DataFrame, alternative: str) -> pd.DataFrame:
    """Compare paired classifier means and apply Holm within each task/metric."""
    classifier_means = results.groupby(
        ["task", "method", "classifier"], as_index=False
    )[["primary", "auc"]].mean()
    rows: list[dict] = []
    for task, group in classifier_means.groupby("task"):
        for metric in ("primary", "auc"):
            for row in _paired_method_tests(group, metric, alternative):
                rows.append({"task": task, **row})
    return pd.DataFrame(rows, columns=(
        "task", "method", "metric", "alternative", "n_classifiers",
        "w_statistic", "mean_difference", "p_value",
        "holm_adjusted_p_value", "significant_0_05",
    ))


def _paired_method_tests(data: pd.DataFrame, metric: str,
                         alternative: str) -> list[dict]:
    raw = data[data["method"] == "raw"][["classifier", metric]]
    rows = []
    for method in (name for name in _method_names(data) if name != "raw"):
        denoised = data[data["method"] == method][["classifier", metric]]
        paired = raw.merge(
            denoised, on="classifier", suffixes=("_raw", "_denoised")
        ).dropna()
        if paired.empty:
            continue
        raw_values = paired[f"{metric}_raw"]
        denoised_values = paired[f"{metric}_denoised"]
        differences = denoised_values.to_numpy() - raw_values.to_numpy()
        if (differences == 0).all():
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                denoised_values,
                raw_values,
                alternative=alternative,
                method="auto",
            )
        rows.append({
            "method": method,
            "metric": metric,
            "alternative": (
                "denoised < raw" if alternative == "less" else "denoised != raw"
            ),
            "n_classifiers": len(paired),
            "w_statistic": float(statistic),
            "mean_difference": float(differences.mean()),
            "p_value": float(p_value),
        })

    adjusted = _holm_adjust([row["p_value"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = adjusted_p
        row["significant_0_05"] = adjusted_p < 0.05
    return rows


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original comparison order."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _write_tables(summary: pd.DataFrame, task_dir: Path, task_name: str) -> None:
    numbers = TABLE_NUMBERS.get(task_name)
    if numbers is None:
        detailed_path = task_dir / "classifier_results.csv"
        average_path = task_dir / "method_average.csv"
    else:
        detailed_path = task_dir / f"table_{numbers[0]:02d}_classifier_results.csv"
        average_path = task_dir / f"table_{numbers[1]:02d}_method_average.csv"

    detailed = _sort_components(summary.copy())
    detailed["method"] = detailed["method"].map(
        lambda value: DISPLAY_METHOD.get(value, value)
    )
    detailed["classifier"] = detailed["classifier"].map(
        lambda value: DISPLAY_CLASSIFIER.get(value, value)
    )
    detailed["primary (mean ± std)"] = detailed.apply(
        lambda row: f"{row.primary_mean:.4f} ± {row.primary_std:.4f}", axis=1)
    detailed["AUC (mean ± std)"] = detailed.apply(
        lambda row: f"{row.auc_mean:.4f} ± {row.auc_std:.4f}", axis=1)
    detailed[["method", "classifier", "primary (mean ± std)", "AUC (mean ± std)"]].to_csv(
        detailed_path, index=False)

    averaged = summary.groupby("method", as_index=False).agg(
        primary_mean=("primary_mean", "mean"),
        primary_std=("primary_mean", lambda values: values.std(ddof=0)),
        auc_mean=("auc_mean", "mean"),
        auc_std=("auc_mean", lambda values: values.std(ddof=0)))
    averaged = _sort_components(averaged)
    averaged["method"] = averaged["method"].map(
        lambda value: DISPLAY_METHOD.get(value, value)
    )
    averaged.to_csv(average_path, index=False)


def _write_figures(results: pd.DataFrame, task_dir: Path, task_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    numbers = FIGURE_NUMBERS.get(task_name)
    if numbers is None:
        primary_path = task_dir / "figure_primary_bar.png"
        auc_path = task_dir / "figure_auc_bar.png"
        box_path = task_dir / "figure_metric_boxplots.png"
    else:
        primary_path = task_dir / f"figure_{numbers[0]:02d}_primary_bar.png"
        auc_path = task_dir / f"figure_{numbers[1]:02d}_auc_bar.png"
        box_path = task_dir / f"figure_{numbers[2]:02d}_metric_boxplots.png"

    plot_data = results.copy()
    plot_data["method_display"] = plot_data["method"].map(
        lambda value: DISPLAY_METHOD.get(value, value)
    )
    plot_data["classifier_display"] = plot_data["classifier"].map(
        lambda value: DISPLAY_CLASSIFIER.get(value, value)
    )
    classifier_means = plot_data.groupby(
        ["method", "method_display", "classifier", "classifier_display"], as_index=False
    )[["primary", "auc"]].mean()
    method_names = _method_names(results)
    method_order = [DISPLAY_METHOD.get(name, name) for name in method_names]
    classifier_names = _classifier_names(results)
    classifier_order = [DISPLAY_CLASSIFIER.get(name, name) for name in classifier_names]
    palette = _palette(method_order, sns)
    task_title = TASK_DISPLAY.get(task_name, task_name.replace("_", " ").title())

    for metric, path in (("primary", primary_path), ("auc", auc_path)):
        figure, axis = plt.subplots(figsize=(12, 6))
        sns.barplot(data=classifier_means, x="classifier_display", y=metric,
                    order=classifier_order, hue="method_display", hue_order=method_order,
                    palette=palette, errorbar=None, ax=axis)
        metric_label = _metric_label(metric, results)
        axis.set_title(f"{metric_label} of Different Denoising Methods - {task_title}")
        axis.set_xlabel("Classifier")
        axis.set_ylabel(metric_label)

        # Adapt the bar-chart Y axis to the displayed values:
        # minimum value - 0.1 and maximum value + 0.1, constrained to [0, 1].
        metric_values = classifier_means[metric].dropna()
        if metric_values.empty:
            y_lower, y_upper = 0.0, 1.0
        else:
            y_lower = max(0.0, float(metric_values.min()) - 0.1)
            y_upper = min(1.0, float(metric_values.max()) + 0.1)

        axis.set_ylim(y_lower, y_upper)

        # Move the bar-chart legend below the plot and arrange it in one row.
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=max(len(method_order), 1),
            frameon=True,
        )
        figure.subplots_adjust(bottom=0.24)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for axis, metric in zip(axes, ("primary", "auc")):
        sns.boxplot(data=classifier_means, x="method_display", y=metric,
                    order=method_order, hue="method_display", hue_order=method_order,
                    palette=palette, legend=False, ax=axis)
        metric_label = _metric_label(metric, results)
        axis.set_title(f"{metric_label} of Different Denoising Methods - {task_title}")
        axis.set_xlabel("")
        axis.set_ylabel(metric_label)
        axis.tick_params(axis="x", rotation=15)
        _add_significance(axis, classifier_means, metric, method_names, method_order)
    figure.tight_layout()
    figure.savefig(box_path, dpi=300)
    plt.close(figure)


def _add_significance(axis, data: pd.DataFrame, metric: str,
                      method_names: list[str], method_order: list[str]) -> None:
    values = data[metric]
    span = max(float(values.max() - values.min()), 0.05)
    lower = max(0.0, float(values.min()) - 0.12 * span)
    step = 0.13 * span
    top = float(values.max()) + step * max(len(method_names), 2)
    axis.set_ylim(lower, top + step)
    tests = {
        row["method"]: row for row in _paired_method_tests(
            data, metric, alternative="two-sided"
        )
    }
    if not tests:
        return
    for level, method in enumerate((name for name in method_names if name != "raw"), start=1):
        if method not in tests:
            continue
        p_value = tests[method]["holm_adjusted_p_value"]
        label = "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        right = method_order.index(DISPLAY_METHOD.get(method, method))
        y = float(values.max()) + step * level
        cap = step * 0.18
        axis.plot([0, 0, right, right], [y, y + cap, y + cap, y], color="0.25", linewidth=1)
        axis.text(right / 2, y + cap * 1.2, label, ha="center", va="bottom", fontsize=9,
                  fontweight="bold")


def _metric_label(metric: str, results: pd.DataFrame) -> str:
    if metric == "auc":
        return "AUC"
    return str(results["primary_name"].iloc[0]).replace("_", " ").title()


def _method_names(frame: pd.DataFrame) -> list[str]:
    encountered = list(dict.fromkeys(frame["method"].tolist()))
    return [name for name in METHOD_ORDER if name in encountered] + [
        name for name in encountered if name not in METHOD_ORDER
    ]


def _classifier_names(frame: pd.DataFrame) -> list[str]:
    encountered = list(dict.fromkeys(frame["classifier"].tolist()))
    return [name for name in CLASSIFIER_ORDER if name in encountered] + [
        name for name in encountered if name not in CLASSIFIER_ORDER
    ]


def _sort_components(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "method" in result:
        method_rank = {name: rank for rank, name in enumerate(_method_names(result))}
        result["_method_rank"] = result["method"].map(method_rank)
    if "classifier" in result:
        classifier_rank = {name: rank for rank, name in enumerate(_classifier_names(result))}
        result["_classifier_rank"] = result["classifier"].map(classifier_rank)
    rank_columns = [name for name in ("_method_rank", "_classifier_rank") if name in result]
    result = result.sort_values(rank_columns).drop(columns=rank_columns)
    return result


def _palette(method_order: list[str], sns) -> dict[str, object]:
    fallback = sns.color_palette("pastel", n_colors=len(method_order))
    return {
        name: METHOD_COLORS.get(name, fallback[index]) for index, name in enumerate(method_order)
    }
