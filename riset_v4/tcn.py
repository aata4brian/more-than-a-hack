"""TCN offline kecil: status H/A/D dan enam arah batas pada 25 fps."""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .labels import boundary_targets,sample_weights,CLASS_SWAP,EDGE_SWAP
from .models import Calibration


class Residual(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),
            nn.ReLU(),nn.Dropout(.1),nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),nn.ReLU())
    def forward(self,x):
        return x+self.net(x)


class Network(nn.Module):
    halo=126 # Two 3-tap convolutions at each of dilations 1,2,4,8,16,32.
    def __init__(self,n_features,channels=48):
        super().__init__()
        self.input=nn.Conv1d(n_features,channels,1)
        self.blocks=nn.Sequential(*[Residual(channels,d) for d in (1,2,4,8,16,32)])
        self.state=nn.Conv1d(channels,3,1); self.edge=nn.Conv1d(channels,6,1)
    def forward(self,x):
        z=self.blocks(self.input(x))
        return self.state(z),self.edge(z)


class Temporal:
    def __init__(self):
        self.calibration=Calibration()

    def transform(self,x):
        observed=np.isfinite(x)
        z=np.where(observed,(x-self.center)/self.scale,0)
        return np.concatenate([np.clip(z,-10,10),observed.astype(float)],axis=1).astype(np.float32)

    def fit(self,cases,epochs=5,steps=180,device='auto',seed=331,progress=print):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(4)
        device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        sampled=np.concatenate([p[0][::25] for p,_,_ in cases])
        # Fitted on training cases only. All-missing columns retain a mask and zero value.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.center=np.nanmedian(sampled,axis=0)
            self.scale=np.nanstd(sampled,axis=0)
        self.center=np.nan_to_num(self.center)
        self.scale=np.nan_to_num(self.scale,nan=1.)
        self.scale[self.scale<.01]=1.
        self.net=Network(2*len(self.center)).to(device)
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.001)
        arrays=[]
        for p,y,w in cases:
            arrays.append((p,y,boundary_targets(y),sample_weights(y),w))
        rng=np.random.default_rng(seed)
        lengths=np.asarray([len(y)*w for _,y,_,_,w in arrays],float); lengths/=lengths.sum()
        pos_weight=torch.full((1,6,1),15.,device=device)
        self.history=[]; core=512; halo=self.net.halo; width=core+2*halo
        for epoch in range(epochs):
            self.net.train(); total=0.
            for _ in range(steps):
                xx,yy,ee,ww=[],[],[],[]
                for _ in range(8):
                    p,y,e,w,_=arrays[rng.choice(len(arrays),p=lengths)]
                    # Half the windows deliberately centre on a boundary.
                    candidates=np.flatnonzero(e.max(axis=1)>.9)
                    start=int(rng.choice(candidates)-core//2) if len(candidates) and rng.random()<.5 else int(rng.integers(max(1,len(y)-core)))
                    start=max(0,min(start,max(0,len(y)-core)))
                    side=int(rng.integers(2)); left=start-halo; right=start+core+halo
                    a,b=max(0,left),min(len(y),right)
                    x=np.pad(self.transform(p[side][a:b]),((a-left,right-b),(0,0)))
                    target=np.full(core,3,dtype=np.int64); target[:min(core,len(y)-start)]=y[start:start+core]
                    edge=np.zeros((core,6),np.float32); edge[:min(core,len(y)-start)]=e[start:start+core]
                    weight=np.zeros(core,np.float32); weight[:min(core,len(y)-start)]=w[start:start+core]
                    if side: target=CLASS_SWAP[target]; edge=edge[:,EDGE_SWAP]
                    xx.append(x.T); yy.append(target); ee.append(edge.T); ww.append(weight)
                x=torch.from_numpy(np.stack(xx)).to(device)
                y=torch.from_numpy(np.stack(yy).astype(np.int64)).to(device)
                e=torch.from_numpy(np.stack(ee)).to(device)
                w=torch.from_numpy(np.stack(ww)).to(device)
                s,b=self.net(x); s=s[:,:,halo:halo+core]; b=b[:,:,halo:halo+core]
                loss_state=F.cross_entropy(s,y,ignore_index=3,reduction='none')
                loss_edge=F.binary_cross_entropy_with_logits(b,e,pos_weight=pos_weight,reduction='none').mean(dim=1)
                loss=((loss_state+.4*loss_edge)*w).sum()/w.sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),3.)
                optimizer.step(); total+=float(loss.detach().cpu())
            self.history.append({'epoch':epoch+1,'loss':total/steps})
            progress({'tcn_epoch':epoch+1,'loss':total/steps,'device':device})
        self.net.cpu().eval(); self.device_used=device
        return self

    def raw(self,pair,chunk=4096,device='cpu'):
        self.net.to(device).eval(); torch.set_num_threads(4)
        probabilities=[]; boundaries=[]
        with torch.inference_mode():
            for features in pair:
                x=self.transform(features); ps=[]; qs=[]; n=len(x); halo=self.net.halo
                for first in range(0,n,chunk):
                    last=min(n,first+chunk); a=max(0,first-halo); b=min(n,last+halo)
                    tensor=torch.from_numpy(x[a:b].T.copy())[None].to(device)
                    p,q=self.net(tensor)
                    sl=slice(first-a,last-a)
                    ps.append(torch.softmax(p,dim=1)[0,:,sl].T.cpu().numpy())
                    qs.append(torch.sigmoid(q)[0,:,sl].T.cpu().numpy())
                probabilities.append(np.concatenate(ps)); boundaries.append(np.concatenate(qs))
        self.net.cpu()
        return ((probabilities[0]+probabilities[1][:,[1,0,2]])/2,
                (boundaries[0]+boundaries[1][:,EDGE_SWAP])/2)

    def predict(self,pair):
        p,q=self.raw(pair)
        return self.calibration.state(p),self.calibration.edge(q)
