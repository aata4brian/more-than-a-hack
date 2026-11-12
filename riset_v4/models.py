"""Model status dan enam arah batas; kalibrasi memakai pertandingan terpisah."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .labels import CLASS_SWAP, EDGE_SWAP, boundary_targets, sample_weights, train_indices


def aligned(model, x, classes):
    result = np.zeros((len(x),classes),dtype=np.float32)
    with threadpool_limits(limits=4):
        result[:,model.classes_.astype(int)] = model.predict_proba(x)
    return result


def head(x,y,w,iterations=90):
    model = HistGradientBoostingClassifier(max_iter=iterations, max_leaf_nodes=25,
        min_samples_leaf=64,learning_rate=.09,l2_regularization=1.3,
        early_stopping=False,random_state=331)
    with threadpool_limits(limits=4):
        model.fit(x,y,sample_weight=w)
    return model


def switch_context(x):
    # The original switch head used +/-5 steps on a five-frame grid.
    # Keep that +/-1 second context at the new full-frame resolution.
    t=np.arange(len(x));before=x[np.maximum(t-25,0)];after=x[np.minimum(t+25,len(x)-1)]
    return np.column_stack([x,after-before]).astype(np.float32)


@dataclass
class Calibration:
    temperature: float = 1.
    slopes: object = None
    intercepts: object = None

    def state(self,p):
        return softmax(np.log(np.clip(p,1e-7,1))/self.temperature,axis=1).astype(np.float32)

    def edge(self,q):
        if self.slopes is None:
            return q
        return expit(logit(np.clip(q,1e-6,1-1e-6))*self.slopes+self.intercepts).astype(np.float32)


def calibrate(prob, edge, truth):
    p,q,y = np.concatenate(prob),np.concatenate(edge),np.concatenate(truth)
    valid = y < 3
    logs = np.log(np.clip(p[valid],1e-7,1)); actual=y[valid]
    def loss(t):
        z = softmax(logs/t,axis=1)
        return -np.log(np.clip(z[np.arange(len(z)),actual],1e-9,1)).mean()
    t=float(minimize_scalar(loss,bounds=(.4,3.),method='bounded').x)
    # Build targets per half, never invent an edge across half time.
    target = np.concatenate([boundary_targets(v,radius=2)>0 for v in truth])
    slopes, offsets = np.zeros(6),np.zeros(6)
    # Tie home/away counterparts so calibration preserves team symmetry.
    for group in ((0,1),(2,4),(3,5)):
        yy=target[valid][:,group].reshape(-1).astype(int)
        xx=logit(np.clip(q[valid][:,group].reshape(-1),1e-6,1-1e-6))[:,None]
        if len(np.unique(yy)) == 2:
            model=LogisticRegression(C=1.,max_iter=150).fit(xx,yy)
            slopes[list(group)]=max(0.,float(model.coef_[0,0]))
            offsets[list(group)]=float(model.intercept_[0])
        else:
            offsets[list(group)]=float(logit((yy.sum()+1)/(len(yy)+2)))
    return Calibration(t,np.asarray(slopes),np.asarray(offsets))


class Heads:
    def __init__(self,relative=True,weighted=True):
        self.relative=relative; self.weighted=weighted
        self.calibration=Calibration()

    def fit(self,cases,iterations=90):
        xs,ys,ws,es,ew,ex=[],[],[],[],[],[]
        for pair,y,scenario_weight in cases:
            idx=train_indices(y,step=10)
            weights=sample_weights(y) if self.weighted else (y<3).astype(float)
            targets=boundary_targets(y,radius=2)
            labels=np.where(targets.max(axis=1)>0,targets.argmax(axis=1)+1,0)
            edge_weight=np.where(labels>0,15.,1.)
            # Every sampled frame's inclusion probability is known. Dense positives
            # are intentional; independent dense calibration restores event priors.
            for view in range(2):
                xs.append(pair[view][idx]); ys.append((y if view==0 else CLASS_SWAP[y])[idx])
                ex.append(switch_context(pair[view])[idx])
                ws.append(weights[idx]*scenario_weight)
                e=labels.copy()
                if view:
                    positive=e>0; e[positive]=EDGE_SWAP[e[positive]-1]+1
                es.append(e[idx]); ew.append(edge_weight[idx]*scenario_weight)
        x=np.concatenate(xs)
        self.state_model=head(x,np.concatenate(ys),np.concatenate(ws),iterations)
        self.edge_model=head(np.concatenate(ex),np.concatenate(es),np.concatenate(ew),iterations)
        self.n_features=x.shape[1]
        return self

    def raw(self,pair):
        a,b=pair
        p=(aligned(self.state_model,a,3)+aligned(self.state_model,b,3)[:,[1,0,2]])/2
        q=(aligned(self.edge_model,switch_context(a),7)[:,1:]+aligned(self.edge_model,switch_context(b),7)[:,1:][:,EDGE_SWAP])/2
        return p,q

    def predict(self,pair,calibrated=True):
        p,q=self.raw(pair)
        return (self.calibration.state(p),self.calibration.edge(q)) if calibrated else (p,q)


def calibration_report(raw,calibrated,truth):
    p,q=np.concatenate(raw[0]),np.concatenate(raw[1])
    cp,cq=np.concatenate(calibrated[0]),np.concatenate(calibrated[1])
    y=np.concatenate(truth); valid=y<3
    target=np.concatenate([boundary_targets(t,radius=2)>0 for t in truth])
    report={}
    for name,pp,qq in [('raw',p,q),('calibrated',cp,cq)]:
        report[name]={'state_nll':float(-np.log(np.clip(pp[valid][np.arange(valid.sum()),y[valid]],1e-8,1)).mean()),
                      'edge_brier':float(((qq[valid]-target[valid])**2).mean()),
                      'edge_prevalence':float(target[valid].mean()),'edge_mean_probability':float(qq[valid].mean())}
    return report
