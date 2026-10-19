#!/bin/bash
#SBATCH --job-name=lcm-exp25
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=10   # the `quick` QoS caps a job at 10 CPUs
#SBATCH --mem=48G   # the first attempt OOM-killed at 32G: with 3000 images the
                    # D=196608 feature matrix alone is 4.7 GB, and the factor-analysis
                    # initialisation adds a few GB of its own temporaries
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

singularity exec -B "$HOME:$HOME" -B "$DATA:$DATA" "$SIF" bash -c "
  set -euo pipefail
  cd $PROJ
  source .env/bin/activate
  export PYTHONPATH=$PROJ/experiments:$PROJ/Laplace
  export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-10}
  python -u experiments/exp25_paper_scale_baselines.py \
    --dataset cifar100 \
    --data-root $DATA \
    --channels ${CHANNELS:-16,64,192} \
    --n ${NIMG:-3000} \
    --epochs ${EPOCHS:-4} \
    --seeds ${SEEDS:-3} \
    --steps ${STEPS:-300} \
    --em-iters ${EM:-40} \
    --ml-max-d ${ML_MAX_D:-16384} \
    --threads \${SLURM_CPUS_PER_TASK:-10} \
    --tag ${TAG:-exp25_paper_scale}
"
