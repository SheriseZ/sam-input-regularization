#!/bin/bash
# =============================================================
# Run inference on the CIFAR-10 test set for the final checkpoint of
# every one of the 10 conditions and dump confusion.json per run.
#
# Outputs:
#   /scratch1/$USER/sam_experiments/<cond>/confusion.json
#
# After this finishes, rsync the JSON files back locally and run
# scripts/error_analysis_compare.py to compare e.g. C1 vs C6.
#
# Time: ~30-60s of forward pass per checkpoint, total < 15 min plus
#        a few minutes of startup / module load.
# =============================================================

#SBATCH --job-name=eval_confusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --account=yzhao010_1531
#SBATCH --output=logs/eval_confusion_%j.out
#SBATCH --error=logs/eval_confusion_%j.err

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

# One-shot sanity check: confirm the env can import everything we need
python -c "import jax, flax; from sam.sam_jax.models import load_model; from sam.sam_jax.datasets import dataset_source; print('imports OK; jax=', jax.__version__, 'flax=', flax.__version__, 'devices=', jax.devices())" \
  || { echo "ENV BROKEN — imports failed before any ckpt"; exit 1; }
echo "=========================================="

cd "$HOME"

CONDITIONS=(
  c1_sgd_baseline
  c2_sam_only
  c3_mixup_only
  c4_cutmix_only
  c5_sam_mixup
  c6_sam_cutmix
  c7_sgd_cutout
  c8_sam_cutout
  c9_sgd_mixcut
  c10_sam_mixcut
)

PASS=0
FAIL=0
FAILED_LIST=()

for cond in "${CONDITIONS[@]}"; do
  CKPT="/scratch1/$USER/sam_experiments/${cond}"
  if [ ! -d "$CKPT" ]; then
    echo "[skip] $cond — no ckpt dir at $CKPT"
    continue
  fi
  echo "=========================================="
  echo "Eval confusion :: $cond"
  echo "=========================================="
  if python -m sam.scripts.eval_confusion \
       --ckpt_dir   "$CKPT" \
       --model_name WideResnet28x10 \
       --batch_size 200 \
       --out_json   "$CKPT/confusion.json"; then
    PASS=$((PASS+1))
    echo "[ok]   $cond"
  else
    rc=$?
    FAIL=$((FAIL+1))
    FAILED_LIST+=("$cond (rc=$rc)")
    echo "[FAIL] $cond (rc=$rc) — see stderr above for traceback"
  fi
done

echo "=========================================="
echo "Summary:  ok=$PASS  fail=$FAIL"
if [ $FAIL -gt 0 ]; then
  echo "Failed conditions:"
  for c in "${FAILED_LIST[@]}"; do echo "  $c"; done
fi
echo ""
echo "JSON files written:"
ls -la /scratch1/$USER/sam_experiments/*/confusion.json 2>/dev/null
echo "=========================================="
