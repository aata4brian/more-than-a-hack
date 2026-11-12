"""Inferensi kandidat tanpa membuka label target, dengan sidecar video opsional."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .features import pair
from .decode import decode
from .evaluate import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recovery',type=Path)
    args=parser.parse_args()
    import games
    games.DATA_DIR=str(args.data_dir)
    from games import GAME,GAMES
    from night_tracking import local_halves
    from starter.baseline import write_submission
    from evaluation.possession_metrics import validate_submission,read_label_csv
    from .data import fingerprint
    model=joblib.load(args.model)
    n=GAMES[GAME]['n_video_frames']
    result=np.full(n,2,np.int8);probs=np.tile([0.,0.,1.],(n,1)).astype(np.float32)
    qs=np.zeros((n,6),np.float32);recovered=0;sidecar=None
    if args.recovery:
        with np.load(args.recovery,allow_pickle=False) as a:
            sidecar={k:a[k] for k in ('ball','original_ball','recovered','confidence','fps')}
        if sidecar['ball'].shape!=(n,2) or int(sidecar['fps'])!=25:raise ValueError('Recovery alignment mismatch')
        if any(sidecar[k].shape!=(n,) for k in ('recovered','confidence')):raise ValueError('Invalid recovery mask')
    for first,last,data,_ in local_halves():
        if sidecar is not None:
            old=sidecar['original_ball'][first:last]
            if not np.allclose(old,data.ball,equal_nan=True):raise ValueError('Sidecar was generated from different tracking')
            mask=sidecar['recovered'][first:last]&(~np.isfinite(data.ball).all(axis=1))&(sidecar['confidence'][first:last]>=.2)
            ball=data.ball.copy();ball[mask]=sidecar['ball'][first:last][mask]
            data=data.with_ball(ball);recovered+=int(mask.sum())
        features,_=pair(data,True)
        p,q=model.predict(features)
        result[first:last]=decode(p,q,model.decoder)
        probs[first:last]=p;qs[first:last]=q
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_submission(str(args.output),np.array(['H','A','D'])[result])
    error=validate_submission(read_label_csv(str(args.output)),n)
    if error:raise ValueError(error)
    np.savez_compressed(args.output.with_suffix('.probabilities.npz'),probabilities=probs,boundaries=qs)
    atomic_json(args.output.with_suffix('.json'),dict(model_sha256=fingerprint(args.model),frames=n,
        labels={name:int((result==i).sum()) for i,name in enumerate('HAD')},recovered_frames_used=recovered,
        target_labels_used=False,smart_tag_hard_override=False,promoted=False,
        note='Candidate inference only. No target spellF1 is claimed without authorized evaluation labels.'))
    print(json.dumps(dict(output=str(args.output),frames=n,recovered_frames_used=recovered,valid=True)))


if __name__=='__main__':main()
