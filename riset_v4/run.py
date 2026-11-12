"""Eksperimen terpisah dari paket tim 1--15, dengan checkpoint dan CSV."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from . import VERSION
from .data import DEV,RETEST,SCENARIOS,discover,load,split,views,fingerprint
from .features import pair
from .models import Heads,calibrate,calibration_report,aligned
from .decode import decode,Settings
from .evaluate import atomic_json,write_csv,score,aggregate
from night_model import feature_pair,STRIDE
from explore_model import expand,fit_head,adaptive,Settings as LegacySettings


def log(root,stage,**details):
    event=dict(time=datetime.now(timezone.utc).isoformat(),stage=stage,**details)
    print(json.dumps(event),flush=True);atomic_json(root/'status.json',event)


def save_model(path,model):
    temp=path.with_suffix('.tmp');joblib.dump(model,temp,compress=3);temp.replace(path)


def legacy_features(d):return expand(d,feature_pair(d,True))


def legacy_prob(model,features):
    return (aligned(model,features[0],3)+aligned(model,features[1],3)[:,[1,0,2]])/2


def build_training(paths,train,scenarios,seed,root):
    cases=[];lx=[];ly=[];lw=[]
    cache=root/'feature_cache';cache.mkdir(exist_ok=True)
    for match in train:
        halves=load(paths[match])
        for scenario in scenarios:
            log(root,'training_features',match=match,scenario=scenario)
            for h,(d,y) in enumerate(views(halves,scenario,seed)):
                features,_=pair(d,True)
                disk=[]
                for side,x in enumerate(features):
                    path=cache/f'{match}_{scenario}_{seed}_{h}_{side}.npy'
                    np.save(path,x);disk.append(np.load(path,mmap_mode='r'))
                cases.append((tuple(disk),y,SCENARIOS[scenario]))
                del features
                old=legacy_features(d)
                grid=np.minimum(np.arange(0,len(y),STRIDE)+STRIDE//2,len(y)-1)
                idx=np.flatnonzero(y[grid]<3)[::2];truth=y[grid[idx]]
                for side in range(2):
                    lx.append(old[side][idx]);ly.append(truth if side==0 else np.array([1,0,2])[truth])
                    lw.append(np.full(len(idx),SCENARIOS[scenario]))
    return cases,(np.concatenate(lx),np.concatenate(ly),np.concatenate(lw))


def calibration_cases(path,scenarios,seed):
    halves=load(path)
    return [([(pair(d,True)[0],y) for d,y in views(halves,s,seed)],s) for s in scenarios]


def fit_calibration(model,cases,root,name):
    rawp,rawq,truth=[],[],[]
    for halves,_ in cases:
        for features,y in halves:
            p,q=model.raw(features);rawp.append(p);rawq.append(q);truth.append(y)
    model.calibration=calibrate(rawp,rawq,truth)
    p=[model.calibration.state(v) for v in rawp];q=[model.calibration.edge(v) for v in rawq]
    report=calibration_report((rawp,rawq),(p,q),truth)
    atomic_json(root/f'calibration_{name}.json',report)
    # Tiny predeclared search, confined to the calibration match.
    from night_benchmark import combine
    from evaluation.possession_metrics import evaluate
    best=None; candidates=[]
    for threshold in (.025,.075,.15):
        for gain in (1.,2.):
            setting=Settings(boundary_gain=gain,proposal_threshold=threshold)
            pred=[decode(a,b,setting) for a,b in zip(p,q)]
            value=evaluate(combine(truth),combine(pred),taus=(),bootstrap_n=0)['spell_f1_mean']
            candidates.append(dict(**asdict(setting),calibration_spellF1=value))
            if best is None or value>best[0]:best=(value,setting)
    atomic_json(root/f'decoder_{name}.json',{'chosen':asdict(best[1]),'candidates':candidates,
        'selection_source':'calibration match only; repeated scenarios are not independent matches'})
    model.decoder=best[1]
    return model


def benchmark(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    paths=discover(args.cache)
    files=[Path(__file__).parent/name for name in ('__init__.py','data.py','features.py','labels.py','models.py','decode.py','tcn.py','evaluate.py','run.py')]
    codehash=hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in files)).hexdigest()
    manifest=dict(version=VERSION,code_sha256=codehash,cache_sha256={m:fingerprint(p) for m,p in paths.items()},
                  folds=args.folds,seed=args.seed,train_scenarios=args.train_scenarios,
                  test_scenarios=list(SCENARIOS),iterations=args.iterations,epochs=args.epochs,steps=args.steps,
                  device=args.device,python=platform.python_version(),
                  note='All seven matches were inspected in earlier research. This is a controlled retest, not a new independent holdout.')
    existing=root/'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Output belongs to a different code/data/config version; use a new --output directory')
    atomic_json(existing,manifest)
    all_rows=[]
    for fold in args.folds:
        train,cal,tests=split(fold);folder=root/f'fold_{fold}';folder.mkdir(exist_ok=True)
        atomic_json(folder/'split.json',dict(train=train,calibration=cal,test=tests))
        model_path=folder/'models.joblib'
        if model_path.exists():
            log(root,'loading_checkpoint',fold=fold);models=joblib.load(model_path)
        else:
            cases,old=build_training(paths,train,args.train_scenarios,args.seed,root)
            log(root,'fitting_legacy_baseline',fold=fold)
            baseline=fit_head(*old,max_iter=args.iterations);del old;gc.collect()
            log(root,'fitting_relative_unweighted',fold=fold)
            plain=Heads(weighted=False).fit(cases,args.iterations)
            log(root,'fitting_relative_weighted',fold=fold)
            weighted=Heads(weighted=True).fit(cases,args.iterations)
            temporal=None
            if args.epochs:
                from .tcn import Temporal
                log(root,'training_tcn',fold=fold)
                temporal=Temporal().fit(cases,epochs=args.epochs,steps=args.steps,device=args.device,
                    seed=args.seed,progress=lambda v:log(root,'training_tcn',fold=fold,**v))
            del cases;gc.collect()
            log(root,'calibrating',fold=fold,match=cal)
            calibration=calibration_cases(paths[cal],args.train_scenarios,args.seed+17)
            weighted=fit_calibration(weighted,calibration,folder,'trees')
            if temporal is not None:temporal=fit_calibration(temporal,calibration,folder,'tcn')
            del calibration;gc.collect()
            models={'legacy_refit':baseline,'relative':plain,'weighted':weighted,'tcn':temporal}
            save_model(model_path,models)
            save_model(folder/'tree_candidate.joblib',weighted)
            if temporal is not None:save_model(folder/'tcn_candidate.joblib',temporal)
        baseline,plain,weighted,temporal=(models[k] for k in ('legacy_refit','relative','weighted','tcn'))
        for match in tests:
            path=folder/f'{match}_rows.json'
            if path.exists():all_rows.extend(json.loads(path.read_text()));continue
            halves=load(paths[match]);rows=[]
            for scenario in SCENARIOS:
                log(root,'evaluating',fold=fold,match=match,scenario=scenario)
                truth=[];predictions={};arrays={}
                for h,(d,y) in enumerate(views(halves,scenario,args.seed+101)):
                    truth.append(y);features,_=pair(d,True)
                    lp=legacy_prob(baseline,legacy_features(d))
                    p0,_=plain.raw(features);p,q=weighted.raw(features)
                    cp,cq=weighted.calibration.state(p),weighted.calibration.edge(q)
                    pred={'legacy_refit':adaptive(lp,len(y),LegacySettings(2,4,-.18)),
                          'relative_25fps':decode(p0), 'episode_weighted':decode(p),
                          'boundary_raw':decode(p,q,Settings()),
                          'boundary_calibrated':decode(cp,cq,weighted.decoder)}
                    if temporal is not None:
                        tp,tq=temporal.predict(features)
                        pred['tcn_dual']=decode(tp,tq,temporal.decoder)
                        pred['ensemble']=decode((cp+tp)/2,(cq+tq)/2,weighted.decoder)
                    for name,value in pred.items():
                        predictions.setdefault(name,[]).append(value);arrays[f'{name}_{h}']=value
                    arrays[f'truth_{h}']=y
                np.savez_compressed(folder/f'predictions_{match}_{scenario}.npz',**arrays)
                for name,pred in predictions.items():
                    row=score(truth,pred,fold=fold,split='development' if match in DEV else 'known_retest',
                              match=match,scenario=scenario,seed=args.seed+101,method=name)
                    rows.append(row);print(json.dumps(row),flush=True)
            atomic_json(path,rows);all_rows.extend(rows)
            write_csv(root/'performa_detail.csv',all_rows);write_csv(root/'ringkasan_performa.csv',aggregate(all_rows))
        del models;gc.collect()
    summary=aggregate(all_rows)
    atomic_json(root/'result.json',{'complete':True,'summary':summary,'rows':len(all_rows),
        'promoted':False,'reason':'Research candidates only. Existing team package and submission remain the reviewed baseline.'})
    log(root,'complete',rows=len(all_rows),best=summary[0])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--folds',type=int,nargs='+',default=[0])
    parser.add_argument('--seed',type=int,default=331)
    parser.add_argument('--train-scenarios',nargs='+',choices=list(SCENARIOS),default=['original','camera_gaps','drift'])
    parser.add_argument('--iterations',type=int,default=90)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--steps',type=int,default=180)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if any(f<0 or f>4 for f in args.folds) or len(set(args.folds))!=len(args.folds):parser.error('folds must be distinct values 0..4')
    if min(args.iterations,args.steps)<1 or args.epochs<0:parser.error('Invalid training budget')
    with threadpool_limits(limits=4):benchmark(args)


if __name__=='__main__':main()
