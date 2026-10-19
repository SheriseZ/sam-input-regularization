#!/bin/bash
# =============================================================
# Condition 10: SAM + mixcut  (= cutout(mixup(x)))
# Backbone: WideResNet-28-10
# Augmentation: basic (random crop + horizontal flip) + mixcut
# Optimizer: SAM, rho = 0.05
#
# SAM version of C9 (SGD + mixcut, 96.97%).
# Compare with:
#   - C10 vs C9  -> SAM's marginal contribution on top of mixcut
#                   (expect ~+0.42, in line with other augs)
#   - C10 vs C5  -> does adding Cutout on top of Mixup buy anything more?
#   - C10 vs C6  -> compared to the overall winner SAM+CutMix (97.68%),
#                   is mixcut weaker?
#   - C10 vs C8  -> compared to SAM+Cutout (97.43%), does the Mixup blend
#                   contribute negatively?
#
# NOTE: mixcut = cutout(mixup(x)), NOT Yun 2019 CutMix. Uses
#        google-research/sam's native `mixcut` flag; the label is the
#        Mixup soft label (Cutout does not change the label).
# =============================================================

#SBATCH --job-name=c10_sam_mixcut
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --account=yzhao010_1531
#SBATCH --output=logs/c10_sam_mixcut_%j.out
#SBATCH --error=logs/c10_sam_mixcut_%j.err

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

export OUTDIR=/scratch1/$USER/sam_experiments/c10_sam_mixcut
mkdir -p "$OUTDIR"

echo "=========================================="
echo "Condition 10: SAM + mixcut = cutout(mixup(x))"
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
  --batch_level_augmentations mixcut \
  --num_epochs 200 \
  --batch_size 128 \
  --learning_rate 0.1 \
  --weight_decay 5e-4 \
  --sam_rho 0.05 \
  --evaluate_every 5

echo "=========================================="
echo "End       : $(date)"
echo "=========================================="
