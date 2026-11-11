"""Estimate changes of possession from context around each five-frame block."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from night_model import STRIDE, emissions

BASE = "flat/play_bias"
BIAS = -.18
CONTEXT = {"context": 2., "context_gap": 3.}
LEARNED = {"switch": 2., "switch_gap": 3.}
LONG = ("gap_fit", "gap_blend", "gap_context", "gap_switch")


def grid_for(n):
    return np.minimum(np.arange(0, n, STRIDE) + STRIDE // 2, n - 1)


def gap_lengths(ball):
    missing = ~np.isfinite(ball).all(axis=1)
    edges = np.flatnonzero(np.diff(missing, prepend=False, append=False))
    lengths = np.zeros(len(ball), dtype=np.int32)
    for first, end in zip(edges[::2], edges[1::2]):
        lengths[first:end] = end - first
    return lengths


def switch_features(x):
    grid = np.arange(len(x))
    past = x[np.maximum(grid - 5, 0)]
    future = x[np.minimum(grid + 5, len(x) - 1)]
    return np.column_stack((x, future - past)).astype(np.float32, copy=False)


def event_labels(truth):
    """Class 1 is H to A, 2 is A to H, 0 is no direct team switch."""
    n = len(truth)
    labels = np.zeros(len(grid_for(n)), dtype=np.int8)
    frames = np.flatnonzero((truth[1:] != truth[:-1]) &
                            (truth[1:] < 2) & (truth[:-1] < 2)) + 1
    for frame in frames:
        step = frame // STRIDE
        labels[max(0, step - 1):step + 2] = 1 if truth[frame] == 1 else 2
    return labels


def fit_switch(items, weights):
    xs, ys, ws = [], [], []
    swap = np.array([0, 2, 1], dtype=np.int8)
    for item in items:
        for pair, truth in zip(item.features, item.truth):
            y = event_labels(truth)
            idx = np.flatnonzero((truth[grid_for(len(truth))] < 3) &
                                 ((y > 0) | (np.arange(len(y)) % 3 == 0)))
            target = y[idx]
            xs.extend((switch_features(pair[0])[idx], switch_features(pair[1])[idx]))
            ys.extend((target, swap[target]))
            w = weights[item.scenario] * np.where(target > 0, 1.6, 1.)
            ws.extend((w, w))
    model = HistGradientBoostingClassifier(
        max_iter=90, max_leaf_nodes=19, learning_rate=.08, min_samples_leaf=45,
        l2_regularization=2., early_stopping=False, random_state=711,
    )
    with threadpool_limits(limits=4):
        model.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
    return model


def switch_probability(model, pair):
    with threadpool_limits(limits=4):
        p, q = (model.predict_proba(switch_features(x)) for x in pair)
    return ((p + q[:, [0, 2, 1]]) / 2)[:, 1:]


def crossing_cost(scores, gaps, method, switches=None):
    length = len(scores)
    costs = np.full((length, 2), 2.)
    gap = gaps[grid_for(len(gaps))]
    if method in CONTEXT or method == "gap_context":
        logits = emissions(scores)
        odds = np.clip(logits[:, 1] - logits[:, 0], -4., 4.)
        grid = np.arange(length)
        local = uniform_filter1d(odds, 5, mode="nearest")
        before = local[np.maximum(grid - 3, 0)]
        after = local[np.minimum(grid + 2, length - 1)]
        strength = np.column_stack((np.minimum(-before, after), np.minimum(before, -after)))
        scale = CONTEXT.get(method, 3.)
        if method != "context":
            scale = np.where(gap > 0, scale, .65)[:, None]
        costs -= scale * np.clip(strength / 2.5, 0, 1)
    if method in LEARNED or method == "gap_switch":
        scale = LEARNED.get(method, 3.)
        if method != "switch":
            scale = np.where(gap > 0, scale, 1.)[:, None]
        costs -= scale * np.clip((switches - .12) / .45, 0., 1.)
    return np.maximum(costs, .25)


def decode_context(scores, gaps, method, switches=None):
    logs = emissions(scores)
    a, b, d = logs[0]
    d += BIAS
    back = np.zeros((len(logs), 3), dtype=np.int8)
    costs = crossing_cost(scores, gaps, method, switches)
    for step in range(1, len(logs)):
        ha, ah = costs[step]
        to_h = (a, b - ah, d - 4.)
        to_a = (a - ha, b, d - 4.)
        to_d = (a - 4., b - 4., d)
        ih = 0 if to_h[0] >= to_h[1] and to_h[0] >= to_h[2] else (1 if to_h[1] >= to_h[2] else 2)
        ia = 0 if to_a[0] >= to_a[1] and to_a[0] >= to_a[2] else (1 if to_a[1] >= to_a[2] else 2)
        iz = 0 if to_d[0] >= to_d[1] and to_d[0] >= to_d[2] else (1 if to_d[1] >= to_d[2] else 2)
        h, away, dead = logs[step]
        a, b, d = h + to_h[ih], away + to_a[ia], dead + to_d[iz] + BIAS
        back[step] = ih, ia, iz
    path = np.empty(len(logs), dtype=np.int8)
    path[-1] = np.argmax((a, b, d))
    for step in range(len(logs) - 1, 0, -1):
        path[step - 1] = back[step, path[step]]
    return np.repeat(path, STRIDE)[:len(gaps)]


def transitions(y):
    idx = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < 3) & (y[:-1] < 3)) + 1
    return idx, y[idx - 1] * 3 + y[idx]


def boundary_audit(truth, pred, gaps):
    """Match boundaries one-to-one by direction and distance, up to 3 seconds."""
    refs, kinds = transitions(truth)
    guess = pred.copy()
    guess[truth == 3] = 3
    finds, found_kinds = transitions(guess)
    match = np.full(len(refs), -1, dtype=np.int32)
    for kind in np.unique(kinds):
        rid = np.flatnonzero(kinds == kind)
        pid = np.flatnonzero(found_kinds == kind)
        pairs = []
        positions = finds[pid]
        for i in rid:
            start, end = np.searchsorted(positions, [refs[i] - 75, refs[i] + 76])
            for j in pid[start:end]:
                pairs.append((abs(int(finds[j]) - int(refs[i])), int(i), int(j)))
        used = set()
        for distance, i, j in sorted(pairs):
            if match[i] == -1 and j not in used:
                match[i] = distance
                used.add(j)
    group = np.where(gaps[refs] == 0, "visible", np.where(gaps[refs] >= 375, "gap15plus", "gap_under15"))
    result = {}
    for name in ("visible", "gap_under15", "gap15plus"):
        for transition, selected in (("team", np.isin(kinds, (1, 3))), ("stop_restart", ~np.isin(kinds, (1, 3)))):
            d = match[(group == name) & selected]
            result[f"{name}_{transition}"] = [int(len(d)), int(((d >= 0) & (d <= 25)).sum()),
                                               int((d > 25).sum()), int((d < 0).sum())]
    result["unmatched_predictions"] = int(len(finds) - (match >= 0).sum())
    return result
