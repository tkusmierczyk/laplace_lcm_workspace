#!/bin/bash
# Build the project venv on the GMUM cluster, inside the shared PyTorch container.
#
# torch/numpy come from the image via --system-site-packages, so nothing large is
# written to $HOME (quota is 60 GiB, soft == hard, no grace). Only the pure-python
# pieces laplace-torch needs are installed, plus asdfghjkl, which the laptop lacks
# and which is the only backend providing the library's `lowrank` structure.
#
# Run under singularity from 00_setup_env.sbatch, not directly.
set -euo pipefail

PROJ="${PROJ:-$HOME/laplace_lcm}"
cd "$PROJ"

echo "=== image python ==="
python -V
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| numpy', numpy.__version__)"
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")

if [ ! -d .env ]; then
  python -m venv --system-site-packages .env
fi
# shellcheck disable=SC1091
source .env/bin/activate

export PIP_NO_CACHE_DIR=1
pip install --quiet --upgrade pip
# Versions pinned to the laptop's, so a cluster result is comparable with a local one.
# curvlinops 3.x moved `curvlinops._base`, which laplace/curvature/curvlinops.py imports,
# so the fork's own ">= 2.0" is too loose: an unpinned install breaks `import laplace`.
pip install --quiet \
  "backpack-for-pytorch==1.7.1" \
  "curvlinops-for-pytorch==2.0.1" \
  "torchmetrics==1.8.2" \
  opt_einsum tqdm pandas scipy scikit-learn
# asdfghjkl is a 2022 alpha and the only backend providing the library's `lowrank`
# structure. It is optional: everything except that baseline runs without it, so a
# failure here is reported rather than fatal.
pip install "asdfghjkl==0.1a4" || echo "WARNING: asdfghjkl unavailable; lowrank stays blocked"

echo "=== after install ==="
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
echo "torch before=$TORCH_BEFORE after=$TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "WARNING: pip replaced the container's torch; the image's CUDA build is gone." >&2
fi

export PYTHONPATH="$PROJ/experiments:$PROJ/Laplace"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "| cuda available", torch.cuda.is_available(),
      "| threads", torch.get_num_threads())
for m in ("numpy", "pandas", "scipy", "sklearn", "torchmetrics", "tqdm",
          "curvlinops", "backpack", "asdfghjkl", "laplace", "common"):
    try:
        mod = importlib.import_module(m)
        print(f"  ok      {m:12} {getattr(mod, '__version__', '')}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  MISSING {m:12} {type(e).__name__}: {e}")
from laplace.utils.enums import HessianStructure
print("hessian structures:", [s.value for s in HessianStructure])
import common
print("provenance:", {k: v for k, v in common.provenance().items()
                      if k in ("experiments_git", "library_git", "host", "python", "torch")})
PY

echo "=== quota ==="
quota -s 2>&1 | tail -2
echo "=== venv size ==="
du -sh "$PROJ/.env"
echo "SETUP OK"
