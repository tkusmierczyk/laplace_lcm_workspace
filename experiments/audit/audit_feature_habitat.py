"""Habitat test: does LCM fit CONV FEATURE-MAP covariance (the paper's actual target)
well, in raster order -- versus the WEIGHT curvature I used throughout?"""
import torch, numpy as np, warnings; warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
import torch.nn as nn
from laplace.lcm import LCMCovariance
from torchvision import datasets, transforms
torch.manual_seed(0)

tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
ds=datasets.MNIST("/home/tom/data",train=True,download=False,transform=tf)
X=torch.stack([ds[i][0] for i in range(512)])

net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                  nn.Conv2d(8,8,3,padding=1),nn.ReLU()).double()
with torch.no_grad(): feat=net(X.double())           # (B, 8, 14, 14)
B,Cc,H,W=feat.shape
V=feat.reshape(B,-1)                                  # raster order: channel-major, then rows, cols
V=V-V.mean(0,keepdim=True)
S=(V.T@V)/B                                           # empirical feature covariance (D x D)
D=S.shape[0]
print(f"conv feature map: {Cc}x{H}x{W} -> D={D}, batch={B}")

def fit(target,init,steps=700):
    L=LCMCovariance(target.shape[0],dtype=torch.float64)
    with torch.no_grad(): L.a.copy_(init)
    L.fit_dense(target,n_steps=steps,lr=5e-2); return L.dense().detach()

nrm=S.norm()
raster=torch.arange(D,dtype=torch.float64)*0.15        # spatial raster order
lcm_raster=fit(S,raster)
lcm_free=fit(S,torch.linspace(0,D*0.15,D))             # learned from uninformed init
diag=torch.diag(S.diagonal())
print("\nrelative Frobenius error to the FEATURE covariance (the paper's target):")
print(f"   LCM, raster-order init : {float((lcm_raster-S).norm()/nrm):.4f}")
print(f"   LCM, uninformed init   : {float((lcm_free-S).norm()/nrm):.4f}")
print(f"   diagonal               : {float((diag-S).norm()/nrm):.4f}")
print("\ncompare: LCM on WEIGHT curvature (my experiments) plateaued at 0.66 vs diagonal 0.97")
