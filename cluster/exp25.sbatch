#!/bin/bash
#SBATCH --job-name=lcm-exp25
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Exp 25: the paper's Figure 4 claim at the paper's scale (D up to ~2e5), against a
# matched-parameter factor analyser. float64 throughout, so the `cpu` partition is both
# cheaper and faster per unit billing than a consumer GPU (fp64 is 1/64 rate there).
set -euo pipefail
PROJ="$HOME/laplace_lcm"
SIF=/shared/sets/singularity/pytorch_24.12-py3.sif
DATA=/shared/sets/datasets

hostname; nproc; free -g | head -2
cd "$PROJ"
git -C "$PROJ" rev-parse --short HEAD
git -C "$PROJ/Laplace" rev-parse --short HEAD

singularity exec -B "$HOME:$HOME" -B "$DATA:$DATA" "$SIF" bash -lc "
  set -euo pipefail
  cd $PROJ
  source .env/bin/activate
  export PYTHONPATH=$PROJ/experiments:$PROJ/Laplace
  export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-16}
  python -u experiments/exp25_paper_scale_baselines.py \
    --dataset cifar100 \
    --data-root $DATA \
    --channels ${CHANNELS:-16,64,192} \
    --n ${NIMG:-3000} \
    --epochs ${EPOCHS:-4} \
    --seeds ${SEEDS:-3} \
    --steps ${STEPS:-300} \
    --em-iters ${EM:-40} \
    --threads \${SLURM_CPUS_PER_TASK:-16} \
    --tag ${TAG:-exp25_paper_scale}
"
