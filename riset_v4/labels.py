"""Target transisi dan bobot episode, hanya dari label laga eksternal."""
import numpy as np

EDGES = ((0,1),(1,0),(0,2),(2,0),(1,2),(2,1))
EDGE_SWAP = np.array([1,0,4,5,2,3])
CLASS_SWAP = np.array([1,0,2,3])


def bounds(y):
    return np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]


def boundary_targets(y, radius=2):
    out = np.zeros((len(y),6),np.float32)
    valid = y<3
    for t in np.flatnonzero((y[1:]!=y[:-1])&(y[1:]<3)&(y[:-1]<3))+1:
        edge = EDGES.index((int(y[t-1]),int(y[t])))
        lo,hi=max(1,t-radius),min(len(y),t+radius+1)
        # Never supervise a transition over a ground-truth hole.
        holes=np.flatnonzero(~valid[lo:hi])+lo
        before=holes[holes<t];after=holes[holes>=t]
        if len(before):lo=int(before[-1]+1)
        if len(after):hi=int(after[0])
        out[lo:hi,edge] = np.maximum(out[lo:hi,edge],1-np.abs(np.arange(lo,hi)-t)/(radius+1))
    out[~valid]=0
    return out


def sample_weights(y, episode=True):
    w = np.ones(len(y),np.float32)
    if episode:
        b = bounds(y)
        for lo,hi in zip(b[:-1],b[1:]):
            if y[lo]<2:
                w[lo:hi]=np.clip(np.sqrt(125/max(hi-lo,1)),.6,3.)
        near = boundary_targets(y, radius=10).max(axis=1)
        w *= 1+1.5*near
    w[y>=3]=0
    if np.any(w):w /= w[w>0].mean()
    return w


def train_indices(y, step=5):
    near = boundary_targets(y, radius=6).max(axis=1)>0
    return np.flatnonzero((y<3)&(near|(np.arange(len(y))%step==0)))
