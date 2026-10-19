#!/bin/bash
#SBATCH --job-name=lcm-exp28
#SBATCH --partition=rtx4090
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8    # the `quick` QoS caps a job at 10 CPUs
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
# Exp 28: model inversion -- the application the covariance model was built for. Needs a
# 24 GB card: the authors' log-likelihood keeps a (batch, dim, 2, 2) float64 tensor per
# prefix-scan step under autograd, ~320 MB per step at layer1's 200 704 dims. Holding all
# four layers' graphs at once needs ~32 GB (job 541487, killed at 22.7 GB), and one layer at
# a time still needs ~20 GB (job 541566, killed at 22.9 GB). The script now recomputes each
# layer's log-likelihood in the backward pass AND splits the batch inside the likelihood,
# which is exact; NLL_CHUNK sets that split. Everything else still sees the full batch.
#
# Budget: 3 arms x ~3400 inversion steps, about 1.5-2.5 GPU-hours for one class and one
# seed -- deliberately under the 4-GPU-hour threshold that needs asking first.
set -euo pipefail
PROJ="$HOME/laplace_lcm"
SIF=/shared/sets/singularity/pytorch_24.12-py3.sif
DATA=/shared/sets/datasets
SHARED=/shared/results/common

hostname; nvidia-smi -L; free -g | head -2
cd "$PROJ"
echo "experiments commit: $(git -C "$PROJ" rev-parse --short HEAD)"
echo "library commit:     $(git -C "$PROJ/Laplace" rev-parse --short HEAD)"

# Pretrained weights are ~130 MB and must not land in the 60 GiB home quota.
mkdir -p "$SHARED/laplace_lcm/torch"

singularity exec --nv -B "$HOME:$HOME" -B "$DATA:$DATA" -B "$SHARED:$SHARED" "$SIF" bash -c "
  set -euo pipefail
  cd $PROJ
  source .env/bin/activate
  export PYTHONPATH=$PROJ/experiments:$PROJ/Laplace
  export TORCH_HOME=$SHARED/laplace_lcm/torch
  # The prefix scan allocates and frees ~320 MB blocks every step, which fragments the
  # 24 GB card; the last failure had 244 MB reserved-but-unusable.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
  python -c 'import torch, torchvision, scipy, PIL; print(\"torch\", torch.__version__, \"| torchvision\", torchvision.__version__, \"| cuda\", torch.cuda.is_available())'
  python -u experiments/exp28_model_inversion.py \
    --ref-dir $PROJ/reference/REMIX \
    --data-root $DATA/ImageNet/ILSVRC/Data/CLS-LOC/train \
    --wnid ${WNID:-n03457902} \
    --class-index ${CLASS_INDEX:-580} \
    --layers ${LAYERS:-layer1,layer2,layer3,layer4} \
    --arms ${ARMS:-none,frobenius,moments} \
    --batch ${BATCH:-50} \
    --n-real ${NREAL:-256} \
    --lcm-epochs ${LCM_EPOCHS:-50} \
    --iters-scale ${ITERS_SCALE:-1.0} \
    --nll-chunk ${NLL_CHUNK:-25} \
    --seeds ${SEEDS:-1} \
    --tag ${TAG:-exp28_inversion}
"
