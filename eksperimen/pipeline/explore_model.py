"""Expanded team-symmetric observations and an adaptive sequence decoder."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from night_model import FPS, STRIDE, Decoder, decode, emissions, mean_valid, motion, rolling


LABEL_SWAP = [1, 0, 2]


def extra_features(data):
    ball, pitch = data.ball, data.pitch
    frame = np.arange(len(ball))
    seen = np.isfinite(ball).all(axis=1)
    last = np.maximum.accumulate(np.where(seen, frame, -1))
    following = np.minimum.accumulate(np.where(seen, frame, len(ball))[::-1])[::-1]
    prev_id = np.maximum(last, 0)
    next_id = np.minimum(following, len(ball) - 1)
    prev_ball = ball[prev_id].copy()
    next_ball = ball[next_id].copy()
    prev_ball[(last < 0) | (frame - last > FPS * 15)] = np.nan
    next_ball[(following == len(ball)) | (following - frame > FPS * 15)] = np.nan
    both = np.isfinite(prev_ball).all(axis=1) & np.isfinite(next_ball).all(axis=1)
    path_gap = np.linalg.norm(next_ball - prev_ball, axis=1)
    path_gap[~both] = np.nan
    fixed = [path_gap, rolling(seen, 125), rolling(seen, 251)]
    signed = []
    teams = [[], []]

    for side, players in enumerate(data.players):
        valid = np.isfinite(players).all(axis=2)
        active = valid.sum(axis=1)
        teams[side].extend([active / 11, rolling(active / 11, 125)])
        for point in (ball, prev_ball, next_ball):
            distances = np.linalg.norm(players - point[:, None], axis=2)
            distances[~np.isfinite(distances)] = np.inf
            near = distances.min(axis=1, initial=np.inf)
            near[~np.isfinite(near)] = np.nan
            density = np.divide((distances <= 7).sum(axis=1), active,
                                out=np.full(len(ball), np.nan), where=active >= 4)
            density[~np.isfinite(point).all(axis=1)] = np.nan
            teams[side].extend((near, rolling(near, 25), rolling(density, 25)))
        for lag in (5, 25):
            velocity = np.full(players.shape[:2], np.nan, dtype=np.float32)
            velocity[lag:] = np.linalg.norm(players[lag:] - players[:-lag], axis=2) * FPS / lag
            velocity[velocity > 12] = np.nan
            valid_motion = np.isfinite(velocity)
            moving = mean_valid(np.where(valid_motion, velocity >= 3, np.nan))
            stationary = mean_valid(np.where(valid_motion, velocity < .65, np.nan))
            teams[side].extend((rolling(moving, 25), rolling(stationary, 75)))

    own, opp = teams
    for a, b in zip(own, opp):
        signed.append(a - b)
    grid = np.minimum(np.arange(0, len(ball), STRIDE) + STRIDE // 2, len(ball) - 1)
    direct = np.column_stack([col[grid] for col in fixed + signed + own + opp]).astype(np.float32)
    mirror = np.column_stack([col[grid] for col in fixed + [-x for x in signed] + opp + own]).astype(np.float32)
    return direct, mirror


def expand(data, pair):
    return tuple(np.column_stack([base, new]) for base, new in zip(pair, extra_features(data)))


def fit_head(x, y, weights, max_iter=125):
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, max_leaf_nodes=25, learning_rate=.09, min_samples_leaf=64,
        l2_regularization=1.3, early_stopping=False, random_state=331,
    )
    with threadpool_limits(limits=4):
        model.fit(x, y, sample_weight=weights)
    return model


def fit_models(x, y, weights):
    flat = fit_head(x, y, weights)
    dead = fit_head(x, (y == 2).astype(np.int8), weights)
    alive = y < 2
    owner = fit_head(x[alive], y[alive], weights[alive])
    return flat, dead, owner


def probabilities(models, pair):
    flat, dead, owner = models
    flat_views, split_views = [], []
    with threadpool_limits(limits=4):
        for data in pair:
            flat_views.append(flat.predict_proba(data))
            prob_dead = dead.predict_proba(data)[:, 1]
            who = owner.predict_proba(data)
            split_views.append(np.column_stack([who * (1 - prob_dead[:, None]), prob_dead]))
    return tuple((views[0] + views[1][:, LABEL_SWAP]) / 2 for views in (flat_views, split_views))


@dataclass(frozen=True)
class Settings:
    team: float
    dead: float
    bias: float = 0.0
    adapt: float = 0.0
    merge: int = 0


def adaptive(scores, n, s):
    if not s.adapt and not s.merge:
        return decode(emissions(scores), n, Decoder(s.team, s.dead, s.bias))
    logs = emissions(scores)
    odds_team = logs[:, 0] - logs[:, 1]
    odds_dead = logs[:, 2] - np.maximum(logs[:, 0], logs[:, 1])
    jump = lambda odds: np.minimum(np.abs(np.diff(odds, prepend=odds[0])) / 3, 1)
    cost_team = s.team * (1 - s.adapt * jump(odds_team))
    cost_dead = s.dead * (1 - s.adapt * jump(odds_dead))
    a, b, c = logs[0]
    c += s.bias
    back = np.empty((len(logs), 3), dtype=np.int8)
    for i in range(1, len(logs)):
        h, w, d = logs[i]
        tc, dc = cost_team[i], cost_dead[i]
        from_h = (a, b - tc, c - dc)
        from_a = (a - tc, b, c - dc)
        from_d = (a - dc, b - dc, c)
        ih = 0 if from_h[0] >= from_h[1] and from_h[0] >= from_h[2] else (1 if from_h[1] >= from_h[2] else 2)
        ia = 0 if from_a[0] >= from_a[1] and from_a[0] >= from_a[2] else (1 if from_a[1] >= from_a[2] else 2)
        id_ = 0 if from_d[0] >= from_d[1] and from_d[0] >= from_d[2] else (1 if from_d[1] >= from_d[2] else 2)
        a, b, c = h + from_h[ih], w + from_a[ia], d + from_d[id_] + s.bias
        back[i] = ih, ia, id_
    states = np.empty(len(logs), dtype=np.int8)
    states[-1] = np.argmax([a, b, c])
    for i in range(len(states) - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    if s.merge:
        bounds = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
        for start, end in zip(bounds[1:-2], bounds[2:-1]):
            if end - start <= s.merge and states[start - 1] == states[end]:
                incumbent = states[end]
                island = states[start]
                if (logs[start:end, island] - logs[start:end, incumbent]).mean() <= 1.3:
                    states[start:end] = incumbent
    return np.repeat(states, STRIDE)[:n]


def choose_probabilities(method, v2, flat, split):
    if method == "v2_redecode":
        return v2
    if method == "flat":
        return flat
    if method == "two_head":
        return split
    if method == "blend":
        return (flat + split) / 2
    return .4 * v2 + .3 * flat + .3 * split
