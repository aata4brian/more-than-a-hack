"""Fitur rapat 25 fps, simetris terhadap pertukaran tim."""
import warnings
import numpy as np
from night_model import rolling, mean_valid
from night_model import feature_pair as legacy_pair, STRIDE
from explore_model import expand

FPS = 25
SCHEMA = 'legacy-plus-relative-contact-25fps-v2'


def velocity(points, lag=3, limit=45.):
    out = np.full_like(points, np.nan, dtype=np.float32)
    out[lag:] = (points[lag:] - points[:-lag]) * FPS / lag
    valid = np.isfinite(points).all(axis=-1)
    # Do not differentiate across a missing observation or a large slot jump.
    continuous = valid.copy()
    for k in range(1, lag + 1):
        continuous[k:] &= valid[:-k]
        continuous[:k] = False
    out[~continuous | (np.linalg.norm(out, axis=-1) > limit)] = np.nan
    return out


def pair(data, relative=True):
    ball = data.ball.astype(np.float32)
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    bv = velocity(ball)
    speed = np.linalg.norm(bv, axis=1)
    acceleration = velocity(bv, lag=3, limit=150.)
    fixed = [seen.astype(float), np.minimum(idx-prev, 750)/25,
             np.minimum(nxt-idx, 750)/25,
             np.abs(ball[:, 0]/data.pitch[0]-.5),
             np.abs(ball[:, 1]/data.pitch[1]-.5), speed,
             np.maximum(-ball, ball-data.pitch).max(axis=1)]
    names_fixed = ['ball_seen','since_ball','until_ball','abs_ball_x','abs_ball_y','ball_speed','pitch_margin']
    for window in (5,25,75):
        fixed.extend([rolling(speed,window),rolling(seen,window)])
        names_fixed.extend([f'ball_speed_mean_{window}',f'visibility_{window}'])
    if relative:
        prev_v = np.roll(bv, 3, axis=0); prev_v[:3] = np.nan
        denom = np.linalg.norm(prev_v,axis=1)*np.linalg.norm(bv,axis=1)
        turn = np.divide((prev_v*bv).sum(axis=1),denom,
                         out=np.full(n,np.nan),where=denom>1)
        fixed.extend([np.linalg.norm(acceleration,axis=1),turn])
        names_fixed += ['ball_acceleration','ball_turn_cosine']
    teams = []
    team_names = None
    for players in data.players:
        pv = velocity(players,lag=5,limit=12.)
        ps = np.linalg.norm(pv,axis=2)
        distances = np.linalg.norm(players-ball[:,None,:],axis=2)
        distances[~np.isfinite(distances)] = np.inf
        valid = np.isfinite(players).all(axis=2)
        nearest = distances.min(axis=1)
        nearest[~np.isfinite(nearest)] = np.nan
        near = np.exp(-nearest/2.5)
        active = valid.sum(axis=1)
        center = np.stack([mean_valid(players[:,:,i]) for i in (0,1)],axis=1)
        spread = mean_valid(np.linalg.norm(players-center[:,None,:],axis=2))
        cols = [nearest, active/11., np.linalg.norm(center-ball,axis=1),spread,
                mean_valid(ps),mean_valid(np.where(np.isfinite(ps),ps<1,np.nan))]
        names = ['nearest','visible_share','center_ball','spread','speed','static_share']
        for w in (5,25,75):
            cols.extend([rolling(near,w),rolling(nearest,w)])
            names.extend([f'contact_mean_{w}',f'distance_mean_{w}'])
        if relative:
            k = min(3,players.shape[1])
            order = np.argsort(distances,axis=1)[:,:k]
            pos = np.take_along_axis(players,order[:,:,None],axis=1)
            vel = np.take_along_axis(pv,order[:,:,None],axis=1)
            delta = ball[:,None,:]-pos
            d = np.linalg.norm(delta,axis=2)
            rel = bv[:,None,:]-vel
            radial = np.sum(delta*rel,axis=2)/np.maximum(d,.1)
            rel_speed = np.linalg.norm(rel,axis=2)
            for j in range(3):
                if j < k:
                    cols.extend([d[:,j],radial[:,j],rel_speed[:,j]])
                else:
                    cols.extend([np.full(n,np.nan)]*3)
                names.extend([f'near{j}_distance',f'near{j}_radial',f'near{j}_relative_speed'])
            contact = ((nearest<2.5)&seen).astype(float)
            same_motion = np.exp(-rel_speed[:,0]/3.)*np.exp(-d[:,0]/2.5)
            # Run duration belongs to a team, never to a supposedly persistent slot ID.
            last_break = np.maximum.accumulate(np.where(contact==0,idx,-1))
            cols.extend([np.minimum(idx-last_break,250)/25.,rolling(same_motion,9),
                         rolling(same_motion,25),rolling(contact,9),
                         ((distances<5)&np.isfinite(distances)).sum(axis=1)/np.maximum(active,1)])
            names += ['contact_run','comotion_short','comotion_long','contact_short','near_density']
        teams.append(cols); team_names = names
    shared = np.column_stack(fixed)
    h,a = (np.column_stack(t) for t in teams)
    forward = np.column_stack([shared,h,a,h-a]).astype(np.float32)
    mirror = np.column_stack([shared,a,h,a-h]).astype(np.float32)
    names = names_fixed + ['home_'+x for x in team_names] + ['away_'+x for x in team_names] + ['diff_'+x for x in team_names]
    for x in (forward,mirror):
        x[~np.isfinite(x)] = np.nan
    # Preserve every v3 feature. Its five-frame grid is expanded without
    # interpolation; the added geometry/contact features above remain at 25 fps.
    old=expand(data,legacy_pair(data,True))
    forward=np.column_stack([np.repeat(old[0],STRIDE,axis=0)[:n],forward]).astype(np.float32)
    mirror=np.column_stack([np.repeat(old[1],STRIDE,axis=0)[:n],mirror]).astype(np.float32)
    names=[f'legacy_{i}' for i in range(old[0].shape[1])]+names
    return (forward,mirror), names
