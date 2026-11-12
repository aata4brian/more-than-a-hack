"""Scorer panitia, diagnostik episode pendek, dan keluaran CSV mentah."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluation import possession_metrics as metrics
from night_benchmark import combine
from explore_benchmark import SCENARIOS


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def score(truth,prediction,**meta):
    y,p=combine(truth),combine(prediction)
    out=metrics.evaluate(y,p,taus=(),bootstrap_n=0)
    runs=metrics.valid_runs(y)
    gt=metrics.possession_spells(y,runs); pred=metrics.possession_spells(p,runs)
    overlaps=metrics.spell_overlaps(gt,pred)
    flags=metrics.matched_gt_flags(overlaps,len(gt),.5)
    duration=np.array([b-a for a,b,_ in gt])/25
    row=dict(meta,spellF1=out['spell_f1_mean'],macroF1=out['macro_f1'],
             deadF1=out['per_class']['D']['f1'],predicted_spells=len(pred),reference_spells=len(gt))
    for name,mask in [('under2s',duration<2),('2to5s',(duration>=2)&(duration<5)),('over5s',duration>=5)]:
        row['recall_'+name]=float(flags[mask].mean()) if mask.any() else ''
        row['count_'+name]=int(mask.sum())
    for threshold,value in out['spell_f1'].items():row[f'F1_IoU{threshold}']=value['f1']
    tt,pt=metrics.transitions(y,runs),metrics.transitions(p,runs)
    for frames in (12,25):
        matched,missed,spurious=metrics.match_transitions(tt,pt,frames,typed='pair')
        row[f'boundary_recall_{frames}frames']=len(matched)/len(tt) if tt else 0.
        row[f'boundary_precision_{frames}frames']=len(matched)/len(pt) if pt else 0.
    return row


def aggregate(rows):
    result=[]
    for method in sorted({r['method'] for r in rows}):
        part=[r for r in rows if r['method']==method]
        # Average scenarios within a match first; folds/seeds are repeated measures.
        by_match={}
        for match in sorted({r['match'] for r in part}):
            subset=[r for r in part if r['match']==match]
            by_match[match]={key:float(np.average([r[key] for r in subset],weights=[SCENARIOS[r['scenario']] for r in subset]))
                             for key in ('spellF1','macroF1','predicted_spells','reference_spells')}
        result.append(dict(method=method,matches=len(by_match),rows=len(part),
                           **{key:float(np.mean([m[key] for m in by_match.values()])) for key in ('spellF1','macroF1','predicted_spells','reference_spells')}))
    return sorted(result,key=lambda r:-r['spellF1'])
