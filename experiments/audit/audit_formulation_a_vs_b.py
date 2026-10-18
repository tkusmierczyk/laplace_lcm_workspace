import torch, numpy as np
torch.set_default_dtype(torch.float64)
from laplace.lcm import LCMCovariance
torch.manual_seed(0)

def ar1(n,d,rho,g):
    e=torch.randn(n,d,generator=g); X=torch.empty(n,d); X[:,0]=e[:,0]; s=(1-rho**2)**.5
    for j in range(1,d): X[:,j]=rho*X[:,j-1]+s*e[:,j]
    return X
def fit(target,rho,steps=600):
    n=target.shape[0]; L=LCMCovariance(n,dtype=torch.float64)
    with torch.no_grad(): L.a.copy_(torch.arange(n)*-np.log(rho))
    L.fit_dense(target,n_steps=steps,lr=5e-2); return L.dense().detach()
def joint_ll(mu,C,y):
    T=len(y); L=torch.linalg.cholesky(C)
    r=torch.linalg.solve_triangular(L,(y-mu).unsqueeze(1),upper=False).squeeze(1)
    return float(-0.5*(r@r+2*torch.log(torch.diagonal(L)).sum()+T*np.log(2*np.pi))/T)

d,n,rho,noise,prior,ntest=250,1500,0.9,0.5,1.0,300
g=torch.Generator().manual_seed(0)
X=ar1(n,d,rho,g); Xt=ar1(ntest,d,rho,g)
coord=torch.linspace(0,1,d)
beta=sum(torch.sin(2*np.pi*f*coord+p) for f,p in [(2,0.),(5,1.),(9,2.)])/d**0.5
y=X@beta+noise*torch.randn(n,generator=g); yt=Xt@beta+noise*torch.randn(ntest,generator=g)
GGN=X.T@X/noise**2; H=GGN+prior*torch.eye(d); Cov_full=torch.linalg.inv(H)
mu_t=Xt@(Cov_full@(X.T@y/noise**2)); I=noise**2*torch.eye(ntest)

covA=torch.linalg.inv(fit(GGN,rho)+prior*torch.eye(d))   # Formulation A (mine)
covB=fit(Cov_full,rho)                                   # Formulation B (fit covariance)
covD=torch.diag(1.0/(GGN.diagonal()+prior))

print("=== PREDICTIVE quality: does fitting the covariance instead of the precision help? ===")
print(f"{'method':22}{'marginal-LL':>13}{'joint-LL':>11}")
for nm,C in [("diagonal",covD),("A: fit precision (mine)",covA),("B: fit covariance",covB),("full (exact)",Cov_full)]:
    v=torch.einsum("ti,ij,tj->t",Xt,C,Xt)+noise**2
    m=float((-0.5*(torch.log(2*np.pi*v)+(yt-mu_t)**2/v)).mean())
    j=joint_ll(mu_t,Xt@C@Xt.T+I,yt)
    print(f"{nm:22}{m:>13.4f}{j:>11.4f}")
