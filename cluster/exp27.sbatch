#!/bin/bash
#SBATCH --job-name=lcm-exp27
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=10   # the `quick` QoS caps a job at 10 CPUs
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
# Exp 27: do samples from each fitted covariance have the right spatial autocorrelation?
# This is what REMIX actually uses the covariance for. Capped at D=65536 by default: the
# AR(1) sampler loops over dimensions in Python, so D~2e5 would dominate the job.
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
  export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-10}
  python -u experiments/exp27_sample_structure.py \
    --dataset cifar100 \
    --data-root $DATA \
    --channels ${CHANNELS:-16,64} \
    --n ${NIMG:-3000} \
    --epochs ${EPOCHS:-4} \
    --seeds ${SEEDS:-5} \
    --steps ${STEPS:-200} \
    --n-samples ${NSAMP:-256} \
    --threads \${SLURM_CPUS_PER_TASK:-10} \
    --tag ${TAG:-exp27_sample_structure}"
