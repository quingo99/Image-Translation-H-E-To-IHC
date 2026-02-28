"""Phase 2 report generator with paired, direction-aware comparison plots.

Usage:
    python report.py --base outputs/base/run_001/metrics_<split>.csv \
                     --expr outputs/expr/run_001/metrics_<split>.csv \
                     --output outputs/report
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_SPECS = [
    {"label": "PSNR", "col": "psnr", "higher_is_better": True},
    {"label": "SSIM", "col": "ssim", "higher_is_better": True},
    {"label": "LPIPS", "col": "lpips", "higher_is_better": False},
    {"label": "IOD Rel Err", "col": "iod_rel_err", "higher_is_better": False},
    {"label": "mIOD Rel Err", "col": "miod_rel_err", "higher_is_better": False},
    {"label": "Nuclei Density Err", "col": "nuclei_density_error", "higher_is_better": False},
    {"label": "Membrane Intensity Err", "col": "membrane_intensity_error", "higher_is_better": False},
]

KEY_METRIC_COLS = [
    "psnr",
    "ssim",
    "lpips",
    "miod_rel_err",
    "nuclei_density_error",
    "membrane_intensity_error",
]


def load_metrics(path):
    return pd.DataFrame(pd.read_csv(path))


def build_paired_frame(df_base, df_expr, key="filename"):
    """Align base/expr rows by filename so per-sample deltas are valid."""
    if key in df_base.columns and key in df_expr.columns:
        paired = df_base.merge(df_expr, on=key, how="inner", suffixes=("_base", "_expr"))
        if paired.empty:
            raise ValueError("No shared filenames between base and expr CSVs.")
        missing_base = len(df_base) - len(paired)
        missing_expr = len(df_expr) - len(paired)
        if missing_base or missing_expr:
            print(
                f"[warn] Paired {len(paired)} samples by filename "
                f"(dropped {missing_base} base, {missing_expr} expr)."
            )
        return paired

    n = min(len(df_base), len(df_expr))
    if n == 0:
        raise ValueError("One of the metric CSVs is empty.")
    print(f"[warn] '{key}' not found in one/both CSVs; pairing by row order for first {n} samples.")
    left = df_base.iloc[:n].reset_index(drop=True).add_suffix("_base")
    right = df_expr.iloc[:n].reset_index(drop=True).add_suffix("_expr")
    return pd.concat([left, right], axis=1)


def get_metric_spec(col):
    for spec in METRIC_SPECS:
        if spec["col"] == col:
            return spec
    return None


def get_paired_values(df_paired, col):
    bcol = f"{col}_base"
    ecol = f"{col}_expr"
    if bcol not in df_paired.columns or ecol not in df_paired.columns:
        return None, None
    vals = df_paired[[bcol, ecol]].dropna()
    if vals.empty:
        return None, None
    return vals[bcol], vals[ecol]


def signed_delta(base_vals, expr_vals, higher_is_better):
    """Positive means expr is better; negative means base is better."""
    raw = expr_vals.to_numpy() - base_vals.to_numpy()
    return raw if higher_is_better else -raw


def aggregate_improvement_pct(base_mean, expr_mean, higher_is_better):
    """Mean-level percent change, positive meaning expr is better."""
    den = max(abs(base_mean), 1e-8)
    if higher_is_better:
        return 100.0 * (expr_mean - base_mean) / den
    return 100.0 * (base_mean - expr_mean) / den


def make_summary_table(df_paired, output_dir):
    """Table 1: Metric means + direction-aware deltas and sample win rates."""
    rows = []
    for spec in METRIC_SPECS:
        base_vals, expr_vals = get_paired_values(df_paired, spec["col"])
        if base_vals is None:
            continue

        base_mean = float(base_vals.mean())
        base_std = float(base_vals.std())
        expr_mean = float(expr_vals.mean())
        expr_std = float(expr_vals.std())
        mean_improve_pct = float(
            aggregate_improvement_pct(base_mean, expr_mean, spec["higher_is_better"])
        )
        win_rate = float((signed_delta(base_vals, expr_vals, spec["higher_is_better"]) > 0).mean() * 100.0)

        if abs(mean_improve_pct) < 0.5:
            winner = "Tie"
        else:
            winner = "M_expr" if mean_improve_pct > 0 else "M_base"

        rows.append({
            "Metric": spec["label"],
            "Direction": "higher is better" if spec["higher_is_better"] else "lower is better",
            "M_base (mean)": round(base_mean, 4),
            "M_base (std)": round(base_std, 4),
            "M_expr (mean)": round(expr_mean, 4),
            "M_expr (std)": round(expr_std, 4),
            "Mean Improvement % (expr vs base)": round(mean_improve_pct, 2),
            "Expr Better % Samples": round(win_rate, 2),
            "Winner": winner,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    print("\n=== Summary Table ===")
    print(table.to_string(index=False))
    return table


def make_hypothesis_table(df_paired, output_dir):
    """Table 2: Hypothesis checks based on direction-aware mean improvement."""
    hypotheses = [
        {
            "Hypothesis": "M_expr improves DAB expression agreement (lower mIOD relative error)",
            "Expected": "up",
            "col": "miod_rel_err",
            "higher_is_better": False,
        },
        {
            "Hypothesis": "Membrane structure metrics improve without explicit membrane loss",
            "Expected": "neutral",
            "col": "membrane_intensity_error",
            "higher_is_better": False,
        },
        {
            "Hypothesis": "PSNR may not improve (possible trade-off)",
            "Expected": "neutral/down",
            "col": "psnr",
            "higher_is_better": True,
        },
    ]

    rows = []
    for h in hypotheses:
        base_vals, expr_vals = get_paired_values(df_paired, h["col"])
        if base_vals is None:
            rows.append({
                "Hypothesis": h["Hypothesis"],
                "Expected": h["Expected"],
                "Observed": "metric missing",
                "Status": "Inconclusive",
            })
            continue

        base_mean = float(base_vals.mean())
        expr_mean = float(expr_vals.mean())
        improve_pct = float(
            aggregate_improvement_pct(base_mean, expr_mean, h["higher_is_better"])
        )
        win_rate = float((signed_delta(base_vals, expr_vals, h["higher_is_better"]) > 0).mean() * 100.0)

        if abs(improve_pct) < 1.0:
            observed_state = "neutral"
        elif improve_pct > 0:
            observed_state = "up"
        else:
            observed_state = "down"

        observed = {
            "up": "up (improved)",
            "down": "down (degraded)",
            "neutral": "neutral",
        }[observed_state]

        expected_set = {tok.strip() for tok in h["Expected"].split("/")}
        if observed_state in expected_set:
            status = "Supported"
        elif observed_state == "neutral" and ("up" in expected_set or "down" in expected_set):
            status = "Inconclusive"
        else:
            status = "Contradicted"

        rows.append({
            "Hypothesis": h["Hypothesis"],
            "Expected": h["Expected"],
            "Observed": (
                f"{observed} (base={base_mean:.4f}, expr={expr_mean:.4f}, "
                f"mean_improve={improve_pct:+.2f}%, win_rate={win_rate:.1f}%)"
            ),
            "Status": status,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "hypothesis_table.csv"), index=False)
    print("\n=== Hypothesis Check ===")
    for _, row in table.iterrows():
        print(f"  [{row['Status']}] {row['Hypothesis']}")
        print(f"    Expected: {row['Expected']}, Observed: {row['Observed']}")
    return table


def plot_metric_comparison(df_paired, output_dir):
    """Direction-aware overview: mean improvement and per-sample win rate."""
    labels = []
    mean_improvement = []
    win_rates = []
    for col in KEY_METRIC_COLS:
        spec = get_metric_spec(col)
        if spec is None:
            continue
        base_vals, expr_vals = get_paired_values(df_paired, col)
        if base_vals is None:
            continue
        base_mean = float(base_vals.mean())
        expr_mean = float(expr_vals.mean())
        labels.append(spec["label"])
        mean_improvement.append(
            float(aggregate_improvement_pct(base_mean, expr_mean, spec["higher_is_better"]))
        )
        win_rates.append(float((signed_delta(base_vals, expr_vals, spec["higher_is_better"]) > 0).mean() * 100.0))

    if not labels:
        return

    order = np.argsort(mean_improvement)
    labels = [labels[i] for i in order]
    mean_improvement = [mean_improvement[i] for i in order]
    win_rates = [win_rates[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = np.arange(len(labels))

    colors_improve = ["seagreen" if v >= 0 else "firebrick" for v in mean_improvement]
    axes[0].barh(y, mean_improvement, color=colors_improve, alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    span = max(5.0, float(np.nanmax(np.abs(mean_improvement))) * 1.25)
    axes[0].set_xlim(-span, span)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Mean Improvement % (positive = M_expr better)")
    axes[0].set_title("Average Improvement")
    for yi, val in zip(y, mean_improvement):
        xpos = val + (0.8 if val >= 0 else -0.8)
        axes[0].text(xpos, yi, f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=9)

    colors_win = ["seagreen" if v >= 50.0 else "firebrick" for v in win_rates]
    axes[1].barh(y, win_rates, color=colors_win, alpha=0.85)
    axes[1].axvline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_xlim(0, 100)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Samples Where M_expr is Better (%)")
    axes[1].set_title("Per-Sample Win Rate")
    for yi, val in zip(y, win_rates):
        axes[1].text(min(val + 1.2, 99.0), yi, f"{val:.1f}%", va="center", ha="left", fontsize=9)

    fig.suptitle("Base vs Expr Test Metrics (paired by sample)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison.png"), dpi=180)
    plt.close()


def _plot_ecdf(ax, values, label, color):
    x = np.sort(values)
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size
    ax.plot(x, y, label=label, color=color, linewidth=2)


def plot_expression_comparison(df_paired, output_dir):
    """Expression-focused plots: error CDFs + paired scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, label in zip(
        axes[:2],
        ["iod_rel_err", "miod_rel_err"],
        ["IOD Relative Error", "mIOD Relative Error"],
    ):
        base_vals, expr_vals = get_paired_values(df_paired, col)
        if base_vals is None:
            ax.set_axis_off()
            continue
        base_np = base_vals.to_numpy()
        expr_np = expr_vals.to_numpy()
        _plot_ecdf(ax, base_np, "M_base", "steelblue")
        _plot_ecdf(ax, expr_np, "M_expr", "coral")
        med_base = float(np.median(base_np))
        med_expr = float(np.median(expr_np))
        improve = 100.0 * (med_base - med_expr) / max(abs(med_base), 1e-8)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.set_title(f"{label} Distribution\nMedian improvement: {improve:+.1f}%")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    scatter_metric = "miod_rel_err"
    scatter_label = "mIOD Relative Error"
    base_vals, expr_vals = get_paired_values(df_paired, scatter_metric)
    if base_vals is None:
        axes[2].set_axis_off()
    else:
        base_np = base_vals.to_numpy()
        expr_np = expr_vals.to_numpy()
        improved = expr_np <= base_np
        axes[2].scatter(base_np[improved], expr_np[improved], s=11, alpha=0.35, c="seagreen", label="Improved")
        axes[2].scatter(base_np[~improved], expr_np[~improved], s=11, alpha=0.35, c="firebrick", label="Worse")
        lim = float(np.nanpercentile(np.concatenate([base_np, expr_np]), 99.0))
        lim = max(lim, 1e-6)
        axes[2].plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.7, label="Equal")
        axes[2].set_xlim(0, lim)
        axes[2].set_ylim(0, lim)
        win_rate = 100.0 * float(improved.mean())
        axes[2].set_xlabel(f"{scatter_label} (M_base)")
        axes[2].set_ylabel(f"{scatter_label} (M_expr)")
        axes[2].set_title(f"Paired Sample Comparison\nBelow diagonal = better, win rate: {win_rate:.1f}%")
        axes[2].grid(alpha=0.2)
        axes[2].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expression_comparison.png"), dpi=180)
    plt.close()


def plot_structure_metrics(df_paired, output_dir):
    """Structure-focused plots: paired scatter and signed deltas."""
    structure_specs = [
        {"label": "Nuclei Density Error", "col": "nuclei_density_error"},
        {"label": "Membrane Intensity Error", "col": "membrane_intensity_error"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, spec in zip(axes, structure_specs):
        base_vals, expr_vals = get_paired_values(df_paired, spec["col"])
        if base_vals is None:
            ax.set_axis_off()
            continue

        base_np = base_vals.to_numpy()
        expr_np = expr_vals.to_numpy()
        improved = expr_np <= base_np
        ax.scatter(base_np[improved], expr_np[improved], s=11, alpha=0.35, c="seagreen", label="Improved")
        ax.scatter(base_np[~improved], expr_np[~improved], s=11, alpha=0.35, c="firebrick", label="Worse")
        lim = float(np.nanpercentile(np.concatenate([base_np, expr_np]), 99.0))
        lim = max(lim, 1e-6)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.7)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

        win_rate = 100.0 * float(improved.mean())
        mean_reduce = 100.0 * (base_np.mean() - expr_np.mean()) / max(abs(base_np.mean()), 1e-8)
        ax.set_title(f"{spec['label']}\nMean reduction: {mean_reduce:+.1f}% | Win rate: {win_rate:.1f}%")
        ax.set_xlabel("M_base")
        ax.set_ylabel("M_expr")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Structure Metrics (lower is better)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "structure_metrics.png"), dpi=180)
    plt.close()


def _strip_eval_prefix(stem):
    """Convert eval sample stem '000_00_<name>' -> '<name>' when applicable."""
    parts = stem.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]
    return stem


def _index_eval_images(sample_dir):
    """Return dict: original sample stem -> evaluation sample image path."""
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        return {}
    index = {}
    for path in sorted(sample_dir.glob("*.png")):
        key = _strip_eval_prefix(path.stem)
        index[key] = str(path)
    return index


def plot_eval_side_by_side(df_paired, base_csv, expr_csv, output_dir, n_samples=4):
    """Create qualitative side-by-side panel of base vs expr eval images."""
    base_sample_dir = Path(base_csv).parent / "evaluation" / "samples"
    expr_sample_dir = Path(expr_csv).parent / "evaluation" / "samples"
    base_index = _index_eval_images(base_sample_dir)
    expr_index = _index_eval_images(expr_sample_dir)

    if not base_index or not expr_index:
        print(
            "[warn] Could not find evaluation sample images for base/expr. "
            "Skipping qualitative comparison figure."
        )
        return

    if "filename" in df_paired.columns:
        filenames = df_paired["filename"].astype(str).tolist()
    elif "filename_base" in df_paired.columns:
        filenames = df_paired["filename_base"].astype(str).tolist()
    else:
        filenames = []

    ordered_stems = [Path(name).stem for name in filenames]
    selected = []
    for stem in ordered_stems:
        if stem in base_index and stem in expr_index:
            selected.append(stem)
        if len(selected) >= n_samples:
            break

    if len(selected) < n_samples:
        common = sorted(set(base_index.keys()) & set(expr_index.keys()))
        for stem in common:
            if stem not in selected:
                selected.append(stem)
            if len(selected) >= n_samples:
                break

    selected = selected[:n_samples]
    if not selected:
        print("[warn] No overlapping base/expr evaluation images found. Skipping qualitative panel.")
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    for row, stem in enumerate(selected):
        img_base = plt.imread(base_index[stem])
        img_expr = plt.imread(expr_index[stem])

        axes[row, 0].imshow(img_base)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{stem} | M_base", fontsize=9)

        axes[row, 1].imshow(img_expr)
        axes[row, 1].axis("off")
        axes[row, 1].set_title(f"{stem} | M_expr", fontsize=9)

    fig.suptitle(f"Evaluation Samples: M_base vs M_expr (n={len(selected)})", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "eval_side_by_side_4.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved qualitative comparison figure: {out_path}")


def generate_report(base_csv, expr_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_base = load_metrics(base_csv)
    df_expr = load_metrics(expr_csv)
    df_paired = build_paired_frame(df_base, df_expr, key="filename")

    print(f"Base model rows: {len(df_base)}")
    print(f"Expr model rows: {len(df_expr)}")
    print(f"Paired rows used for plots: {len(df_paired)}")

    make_summary_table(df_paired, output_dir)
    make_hypothesis_table(df_paired, output_dir)
    plot_metric_comparison(df_paired, output_dir)
    plot_expression_comparison(df_paired, output_dir)
    plot_structure_metrics(df_paired, output_dir)
    plot_eval_side_by_side(df_paired, base_csv, expr_csv, output_dir, n_samples=4)

    print(f"\nReport saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to base model metrics CSV")
    parser.add_argument("--expr", type=str, required=True, help="Path to expr model metrics CSV")
    parser.add_argument("--output", type=str, default="outputs/report")
    args = parser.parse_args()
    generate_report(args.base, args.expr, args.output)
