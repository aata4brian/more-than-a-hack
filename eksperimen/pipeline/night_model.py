"""Feature extraction and offline decoding for one football half."""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d

from possession import classify, dead_ball, motion, pitch_margin, runs


FPS = 25
STRIDE = 5
EPS = 1e-6


@dataclass(frozen=True)
class Decoder:
    team_switch: float
    dead_switch: float
    dead_bias: float


def mean_valid(values, axis=1):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float32), where=count > 0)


def rolling(values, window):
    data = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(data)
    numerator = uniform_filter1d(np.where(valid, data, 0), window, mode="nearest")
    denominator = uniform_filter1d(valid.astype(np.float32), window, mode="nearest")
    return np.divide(numerator, denominator, out=np.full_like(data, np.nan), where=denominator > EPS)


def team_geometry(players, pitch, direction):
    x = (players[:, :, 0] / pitch[0] - .5) * direction
    y = players[:, :, 1] / pitch[1] - .5
    x_mean, y_mean = mean_valid(x), mean_valid(y)
    x_width = np.sqrt(mean_valid((x - x_mean[:, None]) ** 2))
    y_width = np.sqrt(mean_valid((y - y_mean[:, None]) ** 2))
    too_few = np.isfinite(x).sum(axis=1) < 5
    for feature in (x_mean, y_mean, x_width, y_width):
        feature[too_few] = np.nan

    delta = np.full_like(players, np.nan)
    delta[10:] = (players[10:] - players[:-10]) * (FPS / 10)
    speed = np.linalg.norm(delta, axis=2)
    speed[speed > 12] = np.nan
    moving = np.isfinite(speed)
    static = np.where(moving, speed < 1.0, np.nan)
    mean_speed = mean_valid(speed)
    static_share = mean_valid(static)
    vx = mean_valid(np.where(moving, delta[:, :, 0], np.nan)) * direction
    return [x_mean, np.abs(y_mean), x_width, y_width, mean_speed, static_share, vx]


def directional_axis(data):
    # Put the home team's own end on the left. The detector uses no match IDs.
    home, away = data.players
    offset = mean_valid(away[:, :, 0]) - mean_valid(home[:, :, 0])
    finite = offset[np.isfinite(offset)]
    return -1 if len(finite) and np.median(finite) < 0 else 1


def feature_pair(data, include_players, prepared_geometry=None):
    """Produce original and team-swapped features on the same five-frame grid."""
    ball, pitch = data.ball, data.pitch
    n = len(ball)
    grid = np.minimum(np.arange(0, n, STRIDE) + STRIDE // 2, n - 1)
    direction = directional_axis(data) if prepared_geometry is None else prepared_geometry[0]
    dx = (ball[:, 0] / pitch[0] - .5) * direction
    ay = np.abs(ball[:, 1] / pitch[1] - .5)
    pos = np.arange(n)
    seen = np.isfinite(ball).all(axis=1)
    prev = np.maximum.accumulate(np.where(seen, pos, -1))
    nxt = np.minimum.accumulate(np.where(seen, pos, n)[::-1])[::-1]
    back = np.where(prev >= 0, np.minimum(pos - prev, 750) / FPS, np.nan)
    ahead = np.where(nxt < n, np.minimum(nxt - pos, 750) / FPS, np.nan)
    previous_x = np.where(prev >= 0, dx[np.maximum(prev, 0)], np.nan)
    next_x = np.where(nxt < n, dx[np.minimum(nxt, n - 1)], np.nan)

    base = data.basic()
    hd, ad = base.distance.T
    hd, ad = (np.where(np.isfinite(d), d.clip(0, 15), np.nan) for d in (hd, ad))
    close = np.fmin(hd, ad)
    speed = motion(ball, FPS)
    margin = np.clip(pitch_margin(ball, pitch), -20, 12)
    fixed = [ay, speed, margin, close, back, ahead, seen.astype(np.float32),
             dead_ball(base, speed, pitch, FPS).astype(np.float32)]
    signed = [dx, previous_x, next_x]
    team_pairs = [(hd, ad)]
    for window in (9, 25, 75):
        fixed.extend(rolling(x, window) for x in (speed, margin, close, seen))
        signed.append(rolling(hd - ad, window))
        team_pairs.append((rolling(np.exp(-hd / 2.5), window),
                           rolling(np.exp(-ad / 2.5), window)))
    if include_players:
        geometry = (prepared_geometry[1] if prepared_geometry is not None else
                    [team_geometry(team, pitch, direction) for team in data.players])
        for i, (home, away) in enumerate(zip(*geometry)):
            for window in (9, 75):
                h, a = rolling(home, window), rolling(away, window)
                # X centroid and X velocity reverse when the teams are exchanged.
                if i in (0, 6):
                    signed.append(h + a)
                    fixed.append(h - a)
                else:
                    team_pairs.append((h, a))

    forward, mirror = list(fixed), list(fixed)
    for x in signed:
        forward.append(x)
        mirror.append(-x)
    for home, away in team_pairs:
        forward.extend((home, away))
        mirror.extend((away, home))
    return tuple(np.column_stack([col[grid] for col in version]).astype(np.float32)
                 for version in (forward, mirror))


def prepare_geometry(data):
    direction = directional_axis(data)
    return direction, [team_geometry(team, data.pitch, direction) for team in data.players]


def smooth_scores(predictor, pair):
    direct, swapped = (predictor.predict_proba(x) for x in pair)
    return (direct + swapped[:, [1, 0, 2]]) / 2


def emissions(probability):
    return np.log(np.clip(probability, 1e-6, 1)).astype(np.float64)


def decode(log_probabilities, n_frames, settings):
    """Viterbi over H/A/D. A live team change and a stoppage have separate costs."""
    prior = log_probabilities[0].copy()
    prior[2] += settings.dead_bias
    a, b, c = prior
    back = np.empty((len(log_probabilities), 3), dtype=np.int8)
    hcost, dcost = settings.team_switch, settings.dead_switch
    for i, (h, w, d) in enumerate(log_probabilities[1:], start=1):
        h_sources = (a, b - hcost, c - dcost)
        a_sources = (a - hcost, b, c - dcost)
        d_sources = (a - dcost, b - dcost, c)
        ih = 0 if h_sources[0] >= h_sources[1] and h_sources[0] >= h_sources[2] else (1 if h_sources[1] >= h_sources[2] else 2)
        ia = 0 if a_sources[0] >= a_sources[1] and a_sources[0] >= a_sources[2] else (1 if a_sources[1] >= a_sources[2] else 2)
        id_ = 0 if d_sources[0] >= d_sources[1] and d_sources[0] >= d_sources[2] else (1 if d_sources[1] >= d_sources[2] else 2)
        a, b, c = h + h_sources[ih], w + a_sources[ia], d + d_sources[id_] + settings.dead_bias
        back[i] = ih, ia, id_
    path = np.empty(len(log_probabilities), dtype=np.int8)
    path[-1] = np.argmax([a, b, c])
    for i in range(len(path) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return np.repeat(path, STRIDE)[:n_frames]


def bridge(data, events=()):
    """Interpolate brief gaps only when endpoints and implied speed are plausible."""
    ball = data.ball.copy()
    unseen = ~np.isfinite(ball).all(axis=1)
    for first, end in runs(unseen):
        if first == 0 or end == len(ball) or end - first > FPS:
            continue
        left, right = ball[first - 1], ball[end]
        pace = np.linalg.norm(right - left) * FPS / (end - first + 1)
        if pace <= 19:
            fraction = np.arange(1, end - first + 1)[:, None] / (end - first + 1)
            ball[first:end] = left + fraction * (right - left)
    return classify(data.with_ball(ball).basic(), data.pitch, FPS, events)
