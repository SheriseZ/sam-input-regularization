#!/bin/bash
# =============================================================
# Condition 5: SAM + Mixup  (NOVEL — not covered by SAM paper Table 1)
# Backbone: WideResNet-28-10
# Augmentation: basic (random crop + horizontal flip) + Mixup (alpha=1.0)
# Optimizer: SAM, rho = 0.05 (paper default)
#
# Compare with: C3 (SGD+Mixup) and C8 (SAM+Cutout).
#   - vs C3 -> isolates SAM's marginal contribution on top of Mixup
#   - vs C8 -> compares the size of SAM's gain across different augs
# =============================================================

#SBATCH --job-name=c5_sam_mixup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --account=yzhao010_1531
#SBATCH --output=logs/c5_sam_mixup_%j.out
#SBATCH --error=logs/c5_sam_mixup_%j.err

set -uo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load cudnn/8.4.0.27-11.6
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam-env || true
export PATH="$HOME/.conda/envs/sam-env/bin:$PATH"
hash -r

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_CPP_MIN_LOG_LEVEL=2

export TFDS_DATA_DIR=/scratch1/$USER/tfds
mkdir -p "$TFDS_DATA_DIR"

export OUTDIR=/scratch1/$USER/sam_experiments/c5_sam_mixup
mkdir -p "$OUTDIR"

echo "=========================================="
echo "Condition 5: SAM + Mixup (rho=0.05, alpha=1.0)"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "OUTDIR    : $OUTDIR"
echo "Start     : $(date)"
echo "=========================================="

python -c "import sys, jax, absl, flax; print('python:', sys.executable); print('jax:', jax.__version__, jax.devices()); print('flax:', flax.__version__)" \
  || { echo "ENV BROKEN — imports failed, aborting"; exit 1; }
echo "=========================================="

cd "$HOME"
python -m sam.sam_jax.train \
  --dataset cifar10 \
  --model_name WideResnet28x10 \
  --output_dir "$OUTDIR" \
  --image_level_augmentations basic \
  --batch_level_augmentations mixup \
  --num_epochs 200 \
  --batch_size 128 \
  --learning_rate 0.1 \
  --weight_decay 5e-4 \
  --sam_rho 0.05 \
  --evaluate_every 5

echo "=========================================="
echo "End       : $(date)"
echo "=========================================="
