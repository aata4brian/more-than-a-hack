"""Data eksternal saja untuk pelatihan; split selalu per pertandingan."""
from pathlib import Path
import hashlib
import numpy as np
from night_tracking import load_external_cache
from explore_benchmark import corrupt, SCENARIOS
from .features import pair

DEV = ('J03WMX','J03WN1','J03WOH','J03WOY','J03WPY')
RETEST = ('J03WQQ','J03WR9')


def split(fold):
    test = DEV[fold % 5]
    calibration = DEV[(fold + 1) % 5]
    train = tuple(x for x in DEV if x not in (test, calibration))
    validate_split(train, (calibration,), (test,) + RETEST)
    return train, calibration, (test,) + RETEST


def validate_split(train, calibration, test):
    groups = [set(x) for x in (train, calibration, test)]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, calibration and test must contain disjoint matches')
    if not set.union(*groups) <= set(DEV + RETEST):
        raise ValueError('Only the declared public IDSSE matches may provide labels')


def discover(folders):
    result = {}
    for match in DEV + RETEST:
        choices = [Path(p) / (match + '.npz') for p in folders]
        found = next((p for p in choices if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f'External cache missing: {match}.npz in {folders}')
        result[match] = found
    return result


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    halves = load_external_cache(path)
    for data, y in halves:
        if len(data.ball) != len(y) or not np.isin(y, [0,1,2,3]).all():
            raise ValueError('Malformed labels or tracking length')
        if data.ball.shape != (len(y),2) or len(y) < 100:
            raise ValueError('Malformed ball array')
        if any(p.shape[0] != len(y) or p.shape[-1] != 2 for p in data.players):
            raise ValueError('Malformed player array')
    return halves


def views(halves, scenario, seed):
    rng = np.random.default_rng(seed)
    return [(corrupt(d,scenario,rng),y) for d,y in halves]


def features(halves, scenario, seed, relative=True):
    return [(pair(d,relative)[0], y) for d,y in views(halves,scenario,seed)]
