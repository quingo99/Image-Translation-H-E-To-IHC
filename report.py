"""Phase 2 report generator: loads metrics CSVs, produces comparison tables and plots.

Usage:
    python report.py --base outputs/base/run_001/metrics_<split>.csv \
                     --expr outputs/expr/run_001/metrics_<split>.csv \
                     --output outputs/report
"""

import argparse
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path):
    return pd.DataFrame(pd.read_csv(path))


def make_summary_table(df_base, df_expr, output_dir):
    """Table 1: Core Phase 2 metrics (mean +/- std)."""
    metrics = {
        "PSNR": "psnr",
        "SSIM": "ssim",
        "LPIPS": "lpips",
        "DAB Pearson-r": "dab_pearson_r",
        "IOD Rel Err": "iod_rel_err",
        "mIOD Rel Err": "miod_rel_err",
        "Nuclei Density Err": "nuclei_density_error",
        "Membrane Intensity Err": "membrane_intensity_error",
    }

    rows = []
    for label, col in metrics.items():
        base_vals = df_base[col] if col in df_base.columns else pd.Series([float("nan")])
        expr_vals = df_expr[col] if col in df_expr.columns else pd.Series([float("nan")])
        rows.append({
            "Metric": label,
            "M_base (mean)": f"{base_vals.mean():.4f}",
            "M_base (std)": f"{base_vals.std():.4f}",
            "M_expr (mean)": f"{expr_vals.mean():.4f}",
            "M_expr (std)": f"{expr_vals.std():.4f}",
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    print("\n=== Summary Table ===")
    print(table.to_string(index=False))
    return table


def make_hypothesis_table(df_base, df_expr, output_dir):
    """Table 2: Hypothesis checks."""
    hypotheses = [
        {
            "Hypothesis": "M_expr improves DAB expression agreement (IOD/mIOD/Pearson-r)",
            "Expected": "up",
            "base_col": "dab_pearson_r",
            "expr_col": "dab_pearson_r",
            "higher_is_better": True,
        },
        {
            "Hypothesis": "Membrane structure metrics improve without explicit membrane loss",
            "Expected": "neutral",
            "base_col": "membrane_intensity_error",
            "expr_col": "membrane_intensity_error",
            "higher_is_better": False,
        },
        {
            "Hypothesis": "PSNR may not improve (possible trade-off)",
            "Expected": "neutral/down",
            "base_col": "psnr",
            "expr_col": "psnr",
            "higher_is_better": True,
        },
    ]

    rows = []
    for h in hypotheses:
        base_mean = df_base[h["base_col"]].mean() if h["base_col"] in df_base.columns else float("nan")
        expr_mean = df_expr[h["expr_col"]].mean() if h["expr_col"] in df_expr.columns else float("nan")
        diff = expr_mean - base_mean

        if abs(diff) < 0.01 * max(abs(base_mean), 1e-8):
            observed_state = "neutral"
        elif (diff > 0 and h["higher_is_better"]) or (diff < 0 and not h["higher_is_better"]):
            observed_state = "up"
        else:
            observed_state = "down"

        observed = {
            "up": "up (improved)",
            "down": "down (degraded)",
            "neutral": "neutral",
        }[observed_state]

        # Status
        expected = h["Expected"]
        expected_set = {tok.strip() for tok in expected.split("/")}
        if observed_state in expected_set:
            status = "Supported"
        elif observed_state == "neutral" and ("up" in expected_set or "down" in expected_set):
            status = "Inconclusive"
        else:
            status = "Contradicted"

        rows.append({
            "Hypothesis": h["Hypothesis"],
            "Expected": expected,
            "Observed": f"{observed} (base={base_mean:.4f}, expr={expr_mean:.4f})",
            "Status": status,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "hypothesis_table.csv"), index=False)
    print("\n=== Hypothesis Check ===")
    for _, row in table.iterrows():
        print(f"  [{row['Status']}] {row['Hypothesis']}")
        print(f"    Expected: {row['Expected']}, Observed: {row['Observed']}")
    return table


def plot_metric_comparison(df_base, df_expr, output_dir):
    """Bar chart comparing key metrics between models."""
    metrics = {
        "PSNR": "psnr",
        "SSIM": "ssim",
        "LPIPS": "lpips",
        "DAB Pearson-r": "dab_pearson_r",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    for ax, (label, col) in zip(axes, metrics.items()):
        base_vals = df_base[col] if col in df_base.columns else pd.Series([float("nan")])
        expr_vals = df_expr[col] if col in df_expr.columns else pd.Series([float("nan")])
        base_mean = base_vals.mean()
        base_std = base_vals.std()
        expr_mean = expr_vals.mean()
        expr_std = expr_vals.std()

        bars = ax.bar(
            ["M_base", "M_expr"],
            [base_mean, expr_mean],
            yerr=[base_std, expr_std],
            capsize=5,
            color=["steelblue", "coral"],
            alpha=0.8,
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        for bar, val in zip(bars, [base_mean, expr_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison.png"), dpi=150)
    plt.close()


def plot_expression_comparison(df_base, df_expr, output_dir):
    """Scatter plots for DAB expression metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (xcol, ycol, title) in zip(axes, [
        ("iod_real", "iod_gen", "IOD: Real vs Generated"),
        ("miod_real", "miod_gen", "mIOD: Real vs Generated"),
        ("iod_rel_err", "miod_rel_err", "Relative Errors: IOD vs mIOD"),
    ]):
        if xcol in df_base.columns and ycol in df_base.columns:
            ax.scatter(df_base[xcol], df_base[ycol], alpha=0.3, s=10, label="M_base")
        if xcol in df_expr.columns and ycol in df_expr.columns:
            ax.scatter(df_expr[xcol], df_expr[ycol], alpha=0.3, s=10, label="M_expr")

        if "Relative" not in title:
            lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", alpha=0.5, label="ideal")
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expression_comparison.png"), dpi=150)
    plt.close()


def plot_structure_metrics(df_base, df_expr, output_dir):
    """Box plots for structure metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, title in zip(axes, [
        "nuclei_density_error", "membrane_intensity_error",
    ], [
        "Nuclei Density Error", "Membrane Intensity Error",
    ]):
        data = []
        labels = []
        if col in df_base.columns:
            data.append(df_base[col].dropna().values)
            labels.append("M_base")
        if col in df_expr.columns:
            data.append(df_expr[col].dropna().values)
            labels.append("M_expr")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ["steelblue", "coral"]
            for patch, color in zip(bp["boxes"], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(title)
        ax.set_ylabel("Error")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "structure_metrics.png"), dpi=150)
    plt.close()


def generate_report(base_csv, expr_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_base = load_metrics(base_csv)
    df_expr = load_metrics(expr_csv)

    print(f"Base model: {len(df_base)} evaluation samples")
    print(f"Expr model: {len(df_expr)} evaluation samples")

    make_summary_table(df_base, df_expr, output_dir)
    make_hypothesis_table(df_base, df_expr, output_dir)
    plot_metric_comparison(df_base, df_expr, output_dir)
    plot_expression_comparison(df_base, df_expr, output_dir)
    plot_structure_metrics(df_base, df_expr, output_dir)

    print(f"\nReport saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Path to base model metrics CSV")
    parser.add_argument("--expr", type=str, required=True, help="Path to expr model metrics CSV")
    parser.add_argument("--output", type=str, default="outputs/report")
    args = parser.parse_args()
    generate_report(args.base, args.expr, args.output)
