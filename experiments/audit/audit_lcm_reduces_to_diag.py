"""Integration check: LCM with w=0 must reduce EXACTLY to diagonal Laplace."""
import torch, numpy as np, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"experiments")
import torch.nn as nn, common as C
from laplace import Laplace
from laplace.lcm import _inverse_softplus
C.set_seed(0)
rng=np.random.default_rng(0); n,d,k=400,6,3
X=rng.standard_normal((n,d)).astype("float32"); (Xs,),_=C.standardize(X)
y=(Xs@rng.standard_normal((d,k))+0.5*rng.standard_normal((n,k))).argmax(1).astype("int64")
net=nn.Sequential(nn.Linear(d,16),nn.Tanh(),nn.Linear(16,k))
net=C.train_map(net,C.loader(Xs,y,128,True),120,5e-3,1e-3,"classification")
ldr=C.loader(Xs,y,256)

la_d=Laplace(net,"classification",subset_of_weights="last_layer",hessian_structure="diag"); la_d.fit(ldr)
la_l=Laplace(net,"classification",subset_of_weights="last_layer",hessian_structure="lcm",lcm_steps=50); la_l.fit(ldr)

with torch.no_grad():
    ggn_diag=la_d.H.detach().double()
    la_l.lcm.w.zero_()
    la_l.lcm.u_raw.copy_(_inverse_softplus((ggn_diag-la_l.lcm.eps).clamp(min=1e-12)))
    la_l.H=la_l.lcm.dense().to(la_l._dtype).to(la_l._device)
    la_l._posterior_scale=None
la_l.prior_precision=la_d.prior_precision

Pl=la_l.posterior_precision.double(); Pd=la_d.posterior_precision.double()
print(f"diag of LCM precision == diag Laplace : {torch.allclose(torch.diag(Pl),Pd,atol=1e-7)}")
print(f"max |off-diagonal| of LCM precision   : {float((Pl-torch.diag(torch.diag(Pl))).abs().max()):.2e}")
print(f"log-det  lcm={float(la_l.log_det_posterior_precision):.6f}   diag={float(la_d.log_det_posterior_precision):.6f}")
p_l=C.classification_probs(la_l,Xs[:64]); p_d=C.classification_probs(la_d,Xs[:64])
print(f"predictive probability max abs diff   : {float(np.abs(p_l-p_d).max()):.2e}")
print(f"marglik  lcm={float(la_l.log_marginal_likelihood()):.6f}   diag={float(la_d.log_marginal_likelihood()):.6f}")
