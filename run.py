"""Phase 2 single entry point.

Usage:
    python run.py train --config configs/base.yaml
    python run.py train --config configs/expr.yaml
    python run.py eval  --config configs/base.yaml --checkpoint outputs/base/run_001/generator_best.pth
    python run.py eval  --config configs/expr.yaml --checkpoint outputs/expr/run_001/generator_best.pth
    python run.py report --base outputs/base/run_001/metrics_<split>.csv \
                         --expr outputs/expr/run_001/metrics_<split>.csv
    python run.py all    (runs train-base, train-expr, eval-both, report)
"""

import argparse
import glob
import os


def find_latest_run(base_dir):
    """Find the latest run_NNN directory."""
    runs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    if not runs:
        return None
    return runs[-1]


def find_best_checkpoint(run_dir):
    """Find generator_best.pth in a run directory."""
    if not run_dir:
        return None
    best = os.path.join(run_dir, "generator_best.pth")
    if os.path.exists(best):
        return best
    final = os.path.join(run_dir, "generator_final.pth")
    if os.path.exists(final):
        return final
    return None


def find_metrics_csv(run_dir):
    """Find metrics CSV, preferring test split and falling back to val."""
    if not run_dir:
        return None
    for name in ("metrics_test.csv", "metrics_val.csv"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Pipeline Runner")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", type=str, required=True)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a model")
    p_eval.add_argument("--config", type=str, required=True)
    p_eval.add_argument("--checkpoint", type=str, required=True)

    # report
    p_report = sub.add_parser("report", help="Generate comparison report")
    p_report.add_argument("--base", type=str, required=True)
    p_report.add_argument("--expr", type=str, required=True)
    p_report.add_argument("--output", type=str, default="outputs/report")

    # all
    p_all = sub.add_parser("all", help="Run full pipeline: train both, eval both, report")

    args = parser.parse_args()

    if args.command == "train":
        from train import train, load_config
        cfg = load_config(args.config)
        train(cfg)

    elif args.command == "eval":
        from eval import evaluate, load_config
        cfg = load_config(args.config)
        evaluate(cfg, args.checkpoint)

    elif args.command == "report":
        from report import generate_report
        generate_report(args.base, args.expr, args.output)

    elif args.command == "all":
        from train import train, load_config as load_train_config
        from eval import evaluate, load_config as load_eval_config
        from report import generate_report

        # Step 1: Train baseline
        print("=" * 60)
        print("  STEP 1: Training M_base (Pyramid Pix2Pix)")
        print("=" * 60)
        cfg_base = load_train_config("configs/base.yaml")
        train(cfg_base)

        # Step 2: Train expr
        print("=" * 60)
        print("  STEP 2: Training M_expr (Pyramid Pix2Pix + L_expr)")
        print("=" * 60)
        cfg_expr = load_train_config("configs/expr.yaml")
        train(cfg_expr)

        # Step 3: Evaluate baseline
        print("=" * 60)
        print("  STEP 3: Evaluating M_base")
        print("=" * 60)
        base_run = find_latest_run("outputs/base")
        base_ckpt = find_best_checkpoint(base_run)
        if base_ckpt:
            cfg_base = load_eval_config("configs/base.yaml")
            evaluate(cfg_base, base_ckpt)
        else:
            print("ERROR: No checkpoint found for base model!")

        # Step 4: Evaluate expr
        print("=" * 60)
        print("  STEP 4: Evaluating M_expr")
        print("=" * 60)
        expr_run = find_latest_run("outputs/expr")
        expr_ckpt = find_best_checkpoint(expr_run)
        if expr_ckpt:
            cfg_expr = load_eval_config("configs/expr.yaml")
            evaluate(cfg_expr, expr_ckpt)
        else:
            print("ERROR: No checkpoint found for expr model!")

        # Step 5: Report
        print("=" * 60)
        print("  STEP 5: Generating comparison report")
        print("=" * 60)
        base_csv = find_metrics_csv(base_run)
        expr_csv = find_metrics_csv(expr_run)
        if base_csv and expr_csv and os.path.exists(base_csv) and os.path.exists(expr_csv):
            generate_report(base_csv, expr_csv, "outputs/report")
        else:
            print("ERROR: Could not find metrics CSVs for report generation.")

        print("\nPipeline complete!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
