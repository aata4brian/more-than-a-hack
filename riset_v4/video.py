"""Pemulihan konservatif dari piksel video, dengan dua jangkar dan uji bolak-balik.

Tidak mengisi celah dengan garis lurus. Bola yang tertutup/tidak tampak boleh
tetap hilang. Hasil berupa sidecar; tracking asli tidak pernah ditulis ulang.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .evaluate import atomic_json,write_csv


def project(matrix,point):
    z=matrix@np.r_[point,1.]
    return z[:2]/z[2] if abs(z[2])>1e-10 else np.full(2,np.nan)


def patch(frame,point,radius=6):
    h,w=frame.shape
    if not np.isfinite(point).all() or not (radius<=point[0]<w-radius and radius<=point[1]<h-radius):return None
    return cv2.getRectSubPix(frame,(radius*2+1,radius*2+1),tuple(map(float,point))).astype(np.float32)


def similarity(a,b):
    if a is None or b is None:return -1.
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a*b).sum()/den) if den>1e-5 else -1.


def follow(frames,start):
    points=[np.asarray(start,np.float32)];quality=[1.]
    template=patch(frames[0],start)
    if template is None or template.std()<4:return None,None,'weak_anchor'
    params=dict(winSize=(15,15),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
    for prev,cur in zip(frames[:-1],frames[1:]):
        p=points[-1].reshape(1,1,2)
        q,status,error=cv2.calcOpticalFlowPyrLK(prev,cur,p,None,**params)
        if q is None or not status[0,0]:return None,None,'flow_lost'
        back,ok,_=cv2.calcOpticalFlowPyrLK(cur,prev,q,None,**params)
        cycle=float(np.linalg.norm(back-p)) if back is not None else float('inf')
        ncc=similarity(template,patch(cur,q[0,0]))
        if not ok[0,0] or cycle>1.5 or ncc<.35:return None,None,'appearance_or_cycle'
        points.append(q[0,0]);quality.append(float(np.exp(-cycle)*max(0,ncc)))
    return np.asarray(points),np.asarray(quality),''


def recover_gap(frames,matrices,valid,reprojection,start,end,pitch):
    n=len(frames)
    if n<3 or len(matrices)!=n:return None,None,'length'
    if not np.all(valid) or not np.isfinite(matrices).all() or np.max(reprojection)>3.:
        return None,None,'homography'
    a=project(matrices[0],start/pitch);b=project(matrices[-1],end/pitch)
    forward,fq,why=follow(frames,a)
    if forward is None:return None,None,'forward_'+why
    reverse,rq,why=follow(frames[::-1],b)
    if reverse is None:return None,None,'reverse_'+why
    backward=reverse[::-1];rq=rq[::-1]
    discrepancy=np.linalg.norm(forward-backward,axis=1)
    if discrepancy.max()>4.:return None,None,'two_anchor_disagreement'
    # Choose independently tracked pixel positions; this is not pitch interpolation.
    pixels=np.where((fq>=rq)[:,None],forward,backward)
    try:xy=np.asarray([project(np.linalg.inv(h),p)*pitch for h,p in zip(matrices,pixels)])
    except np.linalg.LinAlgError:return None,None,'singular_homography'
    xy[0]=start;xy[-1]=end
    if not np.isfinite(xy).all() or (xy < -3).any() or (xy>pitch+3).any():return None,None,'outside_pitch'
    if np.linalg.norm(np.diff(xy,axis=0),axis=1).max()*25>45.:return None,None,'implausible_speed'
    confidence=np.exp(-(discrepancy/3.)**2)*np.minimum(fq,rq)
    if np.min(confidence[1:-1])<.2:return None,None,'low_confidence'
    return xy,confidence,'accepted'


def gaps(ball,first,last):
    missing=~np.isfinite(ball[first:last]).all(axis=1)
    edges=np.diff(np.r_[False,missing,False].astype(np.int8))
    for a,b in zip(np.flatnonzero(edges==1)+first,np.flatnonzero(edges==-1)+first):
        yield int(a),int(b)


class RejectedGap(Exception):pass


def read_frames(cap,first,last,anchor=None):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES,first):raise RuntimeError('Video seek failed')
    frames=[]
    for index in range(first,last):
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Video decode failed at {index}')
        actual=cap.get(cv2.CAP_PROP_POS_FRAMES)
        if actual and abs(actual-(index+1))>.5:raise RuntimeError('Video frame alignment failed')
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        # Fail early on bad pixels instead of buffering ten seconds of unusable
        # video on a laptop. The complete anchored test still runs afterwards.
        if anchor is not None:
            if len(frames)==1:
                template=patch(gray,anchor)
                if template is None or template.std()<4:raise RejectedGap('forward_weak_anchor')
                point=anchor
            else:
                tracked,_,reason=follow(frames[-2:],point)
                if tracked is None:raise RejectedGap('forward_'+reason)
                point=tracked[-1]
            if len(frames)*gray.nbytes>384*1024*1024:raise RejectedGap('video_memory_budget')
    return frames


def recover_window(cap,a,b,matrices,valid,error,ball,pitch):
    if not np.all(valid[a-1:b+1]) or np.max(error[a-1:b+1])>3:
        return None,None,'homography'
    try:
        anchor=project(matrices[a-1],ball[a-1]/pitch)
        frames=read_frames(cap,a-1,b+1,anchor)
    except RejectedGap as rejection:return None,None,str(rejection)
    return recover_gap(frames,matrices[a-1:b+1],valid[a-1:b+1],error[a-1:b+1],ball[a-1],ball[b],pitch)


def run(args):
    # Set before importing games: that module resolves DATA_DIR on import.
    import games
    games.DATA_DIR=str(args.data_dir)
    from games import GAMES,GAME,half_video_blocks
    from night_tracking import local_halves
    from starter.load_homography import load
    from .data import fingerprint
    args.output.mkdir(parents=True,exist_ok=True)
    n=GAMES[GAME]['n_video_frames'];pitch=np.asarray(GAMES[GAME]['pitch'])
    ball=np.full((n,2),np.nan,np.float32)
    for first,last,data,_ in local_halves():ball[first:last]=data.ball
    original=ball.copy();confidence=np.zeros(n,np.float32);accepted=np.zeros(n,bool)
    matrices,error,valid=load(data_dir=str(args.data_dir))
    from games import asset_paths
    hp=Path(asset_paths(data_dir=str(args.data_dir))['homography'])
    import pandas as pd
    indices=pd.read_csv(hp,usecols=['frame'])['frame'].to_numpy()
    if len(matrices)!=n or not np.array_equal(indices,np.arange(n)):raise ValueError('Homography frame alignment mismatch')
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise FileNotFoundError(f'Cannot open video {args.video}')
    if abs(cap.get(cv2.CAP_PROP_FPS)-25)>.01 or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=n:
        cap.release();raise ValueError('Expected tactical video: 25 fps, 199319 frames')
    candidates=[]
    entropy=None
    if args.probabilities:
        p=np.load(args.probabilities,allow_pickle=False)['probabilities']
        if p.shape!=(n,3):raise ValueError('Probability frame alignment mismatch')
        entropy=-(p*np.log(np.clip(p,1e-7,1))).sum(axis=1)
    for _,first,last in half_video_blocks(GAME):
        for a,b in gaps(ball,first,last+1):
            duration=b-a
            priority=duration*(1+float(entropy[a:b].mean()) if entropy is not None else 1.)
            reason='pending'
            if a<=first or b>last:reason='missing_anchor'
            elif duration>args.max_gap_frames:reason='too_long_for_reliable_flow'
            candidates.append(dict(start=a,end=b,length=duration,priority=priority,status=reason,recovered_frames=0))
    candidates.sort(key=lambda v:-v['priority'])
    processed=0;selftest=[]
    try:
        # Mask observed coordinates only for this diagnostic; never use them as possession labels.
        for _,first,last in half_video_blocks(GAME):
            for a in range(first+50,last-20,487):
                b=a+5
                if len(selftest)>=args.selftest:break
                if not np.isfinite(original[a-1:b+1]).all():continue
                xy,c,why=recover_window(cap,a,b,matrices,valid,error,original,pitch)
                row=dict(start=a,end=b,status=why,mean_error_m='',max_error_m='')
                if xy is not None:
                    distance=np.linalg.norm(xy[1:-1]-original[a:b],axis=1)
                    row.update(mean_error_m=float(distance.mean()),max_error_m=float(distance.max()))
                selftest.append(row)
                print(json.dumps(dict(stage='video_selftest',**row)),flush=True)
        for row in candidates:
            if row['status']!='pending':continue
            if processed>=args.top:row['status']='not_attempted_budget';continue
            processed+=1;a,b=row['start'],row['end']
            xy,c,why=recover_window(cap,a,b,matrices,valid,error,ball,pitch)
            row['status']=why
            if xy is not None:
                ball[a:b]=xy[1:-1];confidence[a:b]=c[1:-1];accepted[a:b]=True
                row['recovered_frames']=b-a
            print(json.dumps(dict(stage='video_gap',**row)),flush=True)
            atomic_json(args.output/'status.json',dict(stage='recovering',processed=processed,recovered_frames=int(accepted.sum())))
    finally:cap.release()
    if not np.array_equal(ball[np.isfinite(original).all(1)],original[np.isfinite(original).all(1)]):
        raise AssertionError('Observed tracking was changed')
    np.savez_compressed(args.output/'ball_recovery.npz',ball=ball,confidence=confidence,recovered=accepted,fps=25,
                        original_ball=original)
    if candidates:write_csv(args.output/'gap_audit.csv',candidates)
    if selftest:write_csv(args.output/'selftest_tracking_agreement.csv',selftest)
    result=dict(complete=True,attempted_gaps=processed,total_gaps=len(candidates),recovered_frames=int(accepted.sum()),
                accepted_gaps=sum(r['status']=='accepted' for r in candidates),
                selftest_attempts=len(selftest),selftest_accepted=sum(r['status']=='accepted' for r in selftest),
                homography_sha256=fingerprint(hp),video_name=args.video.name,video_bytes=args.video.stat().st_size,
                max_gap_frames=args.max_gap_frames,top=args.top,
                limitation='Agreement with supplied tracking is a diagnostic, not human ground truth. Recovery is not automatically promoted to submission.')
    atomic_json(args.output/'result.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--top',type=int,default=40)
    parser.add_argument('--max-gap-frames',type=int,default=250);parser.add_argument('--selftest',type=int,default=20)
    parser.add_argument('--probabilities',type=Path)
    args=parser.parse_args()
    if min(args.top,args.max_gap_frames,args.selftest)<0:parser.error('Budgets must be nonnegative')
    run(args)


if __name__=='__main__':main()
