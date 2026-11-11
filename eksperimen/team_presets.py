"""Reuse the competition baseline and the frozen team pipelines."""
import numpy as np
import joblib
from threadpoolctl import threadpool_limits

from explore_model import adaptive, choose_probabilities, expand, probabilities
from games import source_frames
from night_model import feature_pair, smooth_scores
from night_submission import build as build_v2
from night_tracking import local_halves
from possession import dead_ball, motion
from run import HERE, PRESETS
from starter.baseline import build as baseline_codes, smooth
from transition_model import decode_context, gap_lengths

from explore_benchmark import DECODE
from explore_submission import build as build_v3
from make_submission import build as build_rules

FROZEN = HERE / "frozen"


def experimental(numbers):
    """Share the expensive feature extraction across the requested settings."""
    base_model, extended = joblib.load(FROZEN / "v3" / "new_models.joblib")
    results = {number: np.full(source_frames(), "X", dtype="<U1") for number in numbers}
    for first, end, half, tags in local_halves():
        basic = feature_pair(half, True)
        with threadpool_limits(limits=4):
            base = smooth_scores(base_model, basic)
        flat, split = probabilities(extended, expand(half, basic))
        gaps = gap_lengths(half.ball)
        track = half.basic()
        speed = motion(track.ball, 25)
        tagged_stop = dead_ball(track, speed, half.pitch, 25, tags) & ~dead_ball(track, speed, half.pitch, 25)
        for number in numbers:
            _, _, family, setting = PRESETS[number - 1]
            if family == "context":
                codes = decode_context(flat, gaps, setting)
            else:
                p = choose_probabilities(family, base, flat, split)
                codes = adaptive(p, end - first, DECODE[setting])
            codes[tagged_stop] = 2
            results[number][first:end] = np.asarray(list("HAD"))[codes]
    return results


def generate(numbers):
    official = None
    variants = None
    for number in numbers:
        _, title, family, setting = PRESETS[number - 1]
        print(f"\n[{number:02d}] {title}", flush=True)
        if family == "official":
            if official is None:
                official = baseline_codes()
            labels = smooth(official, setting)
        elif family == "rules":
            labels = build_rules()
        elif family == "v2":
            labels = build_v2(FROZEN / "v2" / "selection.json")
        elif family == "v3":
            labels = build_v3(FROZEN / "v3" / "decision.json")
        else:
            if variants is None:
                variants = experimental([n for n in numbers if n >= 8])
            labels = variants[number]
        yield number, labels
