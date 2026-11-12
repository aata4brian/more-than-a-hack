"""Viterbi 25 fps dengan proposal batas baru di seluruh timeline."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import maximum_filter1d
from .labels import EDGES
try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda f:f


@dataclass(frozen=True)
class Settings:
    team: float = 2.
    dead: float = 4.
    boundary_gain: float = 2.
    emission_scale: float = .2
    dead_bias: float = -.18
    proposal_threshold: float = .1


def proposals(q,threshold=.1):
    peak = maximum_filter1d(q,size=5,axis=0,mode='nearest')
    return (q>=peak-1e-8)&(q>=threshold)


def decode(p,q=None,settings=Settings()):
    if len(p)==0:return np.empty(0,np.int8)
    logs=np.log(np.clip(p,1e-6,1))*settings.emission_scale
    logs[:,2]+=settings.dead_bias*settings.emission_scale
    base=np.array([[0,settings.team,settings.dead],
                   [settings.team,0,settings.dead],[settings.dead,settings.dead,0]],float)
    reductions=np.zeros((len(p),6),np.float32)
    if q is not None:
        if q.shape!=(len(p),6):raise ValueError('Boundary shape mismatch')
        active=proposals(q,settings.proposal_threshold)
        reductions=settings.boundary_gain*np.clip(q/max(settings.proposal_threshold,.01),0,2)*active
    return _viterbi(logs,base,reductions,np.asarray(EDGES,dtype=np.int64))


@njit(cache=True)
def _viterbi(logs,base,reductions,edges):
    score=logs[0].copy(); back=np.zeros((len(logs),3),np.int8)
    for t in range(1,len(logs)):
        cost=base.copy()
        for j in range(6):
            a,b=edges[j]
            cost[a,b]=max(.12,base[a,b]-reductions[t,j])
        following=np.zeros(3)
        for b in range(3):
            best=0; value=score[0]-cost[0,b]
            for a in range(1,3):
                alternative=score[a]-cost[a,b]
                if alternative>value:best=a;value=alternative
            back[t,b]=best;following[b]=logs[t,b]+value
        score=following
    result=np.empty(len(logs),np.int8);result[-1]=score.argmax()
    for t in range(len(logs)-1,0,-1):result[t-1]=back[t,result[t]]
    return result
