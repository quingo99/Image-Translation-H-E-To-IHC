#!/bin/bash
#SBATCH --job-name=her2_expr_single
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=granite-gpu-guest
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@your.domain

set -euo pipefail

# Optional module setup (uncomment if your cluster uses environment modules)
# module purge
# module load cuda/12.1

cd "${SLURM_SUBMIT_DIR}"

# Ensure log directory exists before Slurm writes logs.
# If your cluster opens log files before running this script,
# run `mkdir -p logs` once in your submit directory before `sbatch`.
mkdir -p logs

# Activate Python environment
source venv/bin/activate

echo "HOST: $(hostname)"
echo "JOBID: ${SLURM_JOB_ID}"
nvidia-smi || true
python -u -c "import torch; print('PyTorch:', torch.__version__); print('CUDA avail:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"

# Project-specific training command
python -u train.py --config configs/expr.yaml --dataset her2match
