"""Check that splitting the batch inside the authors' log-likelihood is exact.

Compares the whole-batch negative log-likelihood against a size-weighted mean over
chunks, in both value and gradient, using their module unmodified.
"""
import sys

import torch

REF = sys.argv[1]
sys.path.append(REF)          # append, never insert: their directory ships a utils.py
from lcm import LCM           # noqa: E402

torch.manual_seed(0)
DIM, BATCH = 97, 10
m = LCM(DIM).double()
with torch.no_grad():
    m.u_raw.normal_(0, 0.5)
    m.w.normal_(0, 0.3)
    m.a.copy_(torch.randn(DIM).cumsum(0) * 0.1)
m.mu = torch.randn(1, DIM, dtype=torch.float64)
x0 = torch.randn(BATCH, DIM, dtype=torch.float64)


def run(chunk):
    x = x0.clone().requires_grad_(True)
    for p in m.parameters():
        p.grad = None
    if chunk <= 0 or chunk >= BATCH:
        out = m.nll(x)
    else:
        out = 0.0
        for i in range(0, BATCH, chunk):
            part = x[i:i + chunk]
            out = out + m.nll(part) * (part.shape[0] / BATCH)
    out.backward()
    return float(out), x.grad.clone(), m.a.grad.clone(), m.w.grad.clone()


full = run(0)
print(f"whole batch of {BATCH}: nll = {full[0]:.12f}")
for chunk in (1, 3, 4, 7, 10):
    got = run(chunk)
    print(f"  chunks of {chunk:2d}: nll diff {abs(got[0] - full[0]):.3e}"
          f"   d/d-image {float((got[1] - full[1]).abs().max()):.3e}"
          f"   d/d-coord {float((got[2] - full[2]).abs().max()):.3e}"
          f"   d/d-weight {float((got[3] - full[3]).abs().max()):.3e}")
print("\nAll four columns should be ~1e-12 or smaller (float64 reassociation only).")
