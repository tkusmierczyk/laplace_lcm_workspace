"""Is real last-layer GGN Kronecker-structured (KFAC's home) rather than chain-structured?
If yes, no amount of scale/architecture makes a 1-D chain competitive -- it's structural."""
import torch, numpy as np, warnings; warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
import sys; sys.path.insert(0,"experiments")
import common as C
from laplace import Laplace
from laplace.lcm import LCMCovariance
import torch.nn as nn

def nearest_kron(M, m, n):
    """Best ||M - A kron B||_F via SVD of the rearranged matrix (Van Loan-Pitsianis)."""
    # M is (m*n, m*n); A is (m,m), B is (n,n)
    R = torch.zeros(m*m, n*n)
    for i in range(m):
        for j in range(m):
            R[i*m+j] = M[i*n:(i+1)*n, j*n:(j+1)*n].reshape(-1)
    U,S,Vh = torch.linalg.svd(R, full_matrices=False)
    A = (U[:,0]*S[0]).reshape(m,m); B = Vh[0].reshape(n,n)
    return torch.kron(A,B)

def fit_chain(target, steps=800):
    d=target.shape[0]; L=LCMCovariance(d,dtype=torch.float64)
    L.fit_dense(target, n_steps=steps, lr=3e-2); return L.dense().detach()

for width, k in [(8,3),(12,4)]:
    C.set_seed(0); rng=np.random.default_rng(0)
    n,dim=800,10
    B=rng.standard_normal((dim,dim)); Lc=np.linalg.cholesky(B@B.T/dim+0.3*np.eye(dim))
    X=(rng.standard_normal((n,dim))@Lc.T).astype("float32")
    (Xs,),_=C.standardize(X)
    yy=(Xs@rng.standard_normal((dim,k))+0.5*rng.standard_normal((n,k))).argmax(1).astype("int64")
    net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,width),nn.Tanh(),nn.Linear(width,k)).float()
    net=C.train_map(net,C.loader(Xs,yy,128,True),150,5e-3,1e-3,"classification")
    la=Laplace(net,"classification",subset_of_weights="last_layer",hessian_structure="full")
    la.fit(C.loader(Xs,yy,256))
    H=la.H.detach().double().cpu(); H=0.5*(H+H.T)
    P=H.shape[0]
    # last layer is Linear(width -> k): weights k*width, plus bias k  => drop bias block for kron test
    nw=k*width
    Hw=H[:nw,:nw]
    kron=nearest_kron(Hw, k, width)          # k x k  kron  width x width
    chain=fit_chain(Hw)
    diag=torch.diag(Hw.diagonal())
    nrm=Hw.norm()
    print(f"last layer Linear({width}->{k})  weight-GGN {nw}x{nw}   relative Frobenius error:")
    print(f"   nearest Kronecker : {float((kron-Hw).norm()/nrm):.4f}")
    print(f"   best LCM chain    : {float((chain-Hw).norm()/nrm):.4f}")
    print(f"   diagonal          : {float((diag-Hw).norm()/nrm):.4f}\n")
